import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from lxml.etree import XPath
from playwright.async_api import Page, async_playwright
from trafilatura import extract, utils, xpaths

from docreader.config import CONFIG
from docreader.models.document import Document
from docreader.parser.base_parser import BaseParser
from docreader.parser.chain_parser import PipelineParser
from docreader.parser.markdown_parser import MarkdownParser
from docreader.utils import endecode

logger = logging.getLogger(__name__)

_GOTO_TIMEOUT_MS = 30_000
_NETWORK_IDLE_TIMEOUT_MS = 10_000
_SPA_WAIT_TIMEOUT_MS = 15_000
# Minimum visible characters before treating an SPA shell as "rendered".
_SPA_MIN_TEXT_LEN = 80
# Minimum visible characters for Playwright text fallback when trafilatura fails.
_MIN_FALLBACK_TEXT_LEN = 50
# A body this short means trafilatura returned metadata and page furniture but
# no article — worth a second attempt before falling back to plain visible text.
_MIN_EXTRACTED_BODY_LEN = 200
_FRONT_MATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_SECTION_OPEN = re.compile(r"<section\b", re.IGNORECASE)
_SECTION_CLOSE = re.compile(r"</section\s*>", re.IGNORECASE)
# Headless Chromium advertises itself as `HeadlessChrome`, which WeChat
# Official Account pages answer with a "环境异常，完成验证后即可继续访问" stub
# instead of the article. Presenting a plain desktop Chrome UA is enough to get
# the real page back.
_DESKTOP_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)
# Since mid-September 2026 WeChat 302s every long-form article link
# (`/s?__biz=…&mid=…&idx=…&sn=…`) that lacks a non-empty `chksm` to the
# `wappoc_appmsgcaptcha` page — regardless of IP, UA or browser. Any non-empty
# value gets the real article back; short links (`/s/<token>`) are unaffected.
# A `poc_token` copied from someone's browser also forces the captcha.
_WECHAT_HOST = "mp.weixin.qq.com"
_WECHAT_PLACEHOLDER_CHKSM = "chksm=0"
_WECHAT_CAPTCHA_PATH = "/mp/wappoc_appmsgcaptcha"
_WECHAT_CAPTCHA_TEXT = "当前环境异常，完成验证后即可继续访问"

# Monkey-patch trafilatura internals to better support WeChat Official Account
# articles, whose images live on `mmbiz.qpic.cn` without a standard file
# extension and whose main content sits inside `#js_content` /
# `.rich_media_content`. Trafilatura's `utils.IMAGE_EXTENSION` and
# `xpaths.BODY_XPATH` are internal APIs, so we guard the patch and skip
# silently if they are renamed/removed in a future release.
try:
    _WECHAT_IMAGE_EXTENSION = re.compile(
        r"[^\s]+\.(avif|bmp|gif|hei[cf]|jpe?g|png|webp)(\b|$)|"  # Standard extensions
        r"mmbiz\.qpic\.cn/[^\s]*wx_fmt=(jpeg|jpg|png|gif|webp)"  # WeChat query format
    )
    utils.IMAGE_EXTENSION = _WECHAT_IMAGE_EXTENSION

    _WECHAT_BODY_XPATH = XPath(
        '(.//*[@id="js_content" or contains(@class, "rich_media_content")])[1]'
    )
    _wechat_xpath_str = str(_WECHAT_BODY_XPATH)
    if not any(str(x) == _wechat_xpath_str for x in xpaths.BODY_XPATH):
        xpaths.BODY_XPATH.insert(0, _WECHAT_BODY_XPATH)
except (AttributeError, ImportError) as e:
    logger.warning(
        "Failed to patch trafilatura internals for WeChat support: %s", e
    )


@dataclass(frozen=True)
class _ScrapeResult:
    html: str
    visible_text: str
    page_title: str
    page_url: str = ""


def normalize_wechat_article_url(url: str) -> str:
    """Rewrite a long-form WeChat article link so it is not sent to the captcha.

    Drops `poc_token` and adds a placeholder `chksm` when it is missing or
    empty. The query string is edited as raw `k=v` pieces so `__biz`'s
    trailing `==` keeps its original encoding. Other URLs pass through.
    """
    parts = urlsplit(url)
    if parts.hostname != _WECHAT_HOST or parts.path != "/s" or not parts.query:
        return url

    pieces = [p for p in parts.query.split("&") if p]
    keys = [p.split("=", 1)[0] for p in pieces]
    if "__biz" not in keys or "sn" not in keys:
        return url

    kept = [
        p
        for p in pieces
        if p.split("=", 1)[0] != "poc_token"
        and not (p.split("=", 1)[0] == "chksm" and not p.partition("=")[2])
    ]
    if not any(p.split("=", 1)[0] == "chksm" for p in kept):
        kept.append(_WECHAT_PLACEHOLDER_CHKSM)
    return urlunsplit(parts._replace(query="&".join(kept)))


def is_wechat_captcha_page(page_url: str, visible_text: str) -> bool:
    """True when WeChat served its "环境异常" verification page instead of content."""
    return _WECHAT_CAPTCHA_PATH in (page_url or "") or _WECHAT_CAPTCHA_TEXT in (
        visible_text or ""
    )


def browser_context_options() -> dict:
    """Browser context options that make the headless browser look like a desktop one."""
    return {
        "user_agent": _DESKTOP_UA,
        "locale": "zh-CN",
        "viewport": {"width": 1280, "height": 900},
    }


def _run_trafilatura(html: str) -> Optional[str]:
    md_text = extract(
        html,
        output_format="markdown",
        with_metadata=True,
        include_images=True,
        include_tables=True,
        include_links=True,
    )
    if not md_text or not md_text.strip():
        return None
    return md_text


def _markdown_body_len(md: Optional[str]) -> int:
    """Length of the markdown minus the YAML front matter trafilatura emits."""
    if not md:
        return 0
    match = _FRONT_MATTER.match(md)
    return len(md) - match.end() if match else len(md)


def sections_to_divs(html: str) -> str:
    """Rewrite `<section>` to `<div>`.

    WeChat Official Account articles nest their whole body in `<section>`
    elements, which trafilatura's tag catalogue skips — the extraction then
    yields only the front matter and page furniture. `<div>` carries no
    semantics trafilatura treats differently, so this is a safe repair.
    """
    return _SECTION_CLOSE.sub("</div>", _SECTION_OPEN.sub("<div", html))


def extract_markdown_from_html(html: str) -> Optional[str]:
    """Run trafilatura on HTML; return markdown or None if nothing extracted."""
    if not html or not html.strip():
        return None
    md_text = _run_trafilatura(html)
    if _markdown_body_len(md_text) >= _MIN_EXTRACTED_BODY_LEN:
        return md_text

    # Nothing but front matter came back: retry on section-free HTML before
    # giving up (see sections_to_divs). Costs a second pass only in this case.
    retried = _run_trafilatura(sections_to_divs(html))
    if _markdown_body_len(retried) > _markdown_body_len(md_text):
        logger.info(
            "Trafilatura body was %d chars; <section> rewrite recovered %d",
            _markdown_body_len(md_text),
            _markdown_body_len(retried),
        )
        return retried
    return md_text


def build_visible_text_fallback(visible_text: str, page_title: str = "") -> Optional[str]:
    """Build markdown from Playwright-visible text when trafilatura finds no article body."""
    text = (visible_text or "").strip()
    if len(text) < _MIN_FALLBACK_TEXT_LEN:
        return None
    title = (page_title or "").strip()
    if title and not text.startswith(title):
        return f"# {title}\n\n{text}"
    return text


async def wait_for_rendered_content(page: Page) -> None:
    """Wait for SPA/JS pages beyond the initial HTML shell."""
    try:
        await page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS)
        logger.info("Network idle after navigation")
    except Exception:
        logger.info("Network idle wait timed out, continuing")

    try:
        await page.wait_for_function(
            """(minLen) => {
                const root = document.querySelector('#app')
                    || document.querySelector('main')
                    || document.body;
                return ((root?.innerText || '').trim().length >= minLen);
            }""",
            arg=_SPA_MIN_TEXT_LEN,
            timeout=_SPA_WAIT_TIMEOUT_MS,
        )
        logger.info("SPA/root visible text reached minimum length")
    except Exception:
        logger.info("SPA text wait timed out, using current DOM")


async def read_visible_text(page: Page) -> str:
    """Prefer #app/main innerText, then fall back to body."""
    return await page.evaluate(
        """() => {
            const root = document.querySelector('#app')
                || document.querySelector('main')
                || document.querySelector('[role="main"]')
                || document.body;
            return (root?.innerText || '').trim();
        }"""
    )


class StdWebParser(BaseParser):
    """Standard web page parser using Playwright and Trafilatura.

    This parser scrapes web pages using Playwright's Chromium browser and extracts
    clean content using Trafilatura library. It supports proxy configuration and
    converts HTML content to markdown format.
    """

    def __init__(self, title: str, **kwargs):
        """Initialize the web parser.

        Args:
            title: Title of the web page to be used as file name
            **kwargs: Additional arguments passed to BaseParser
        """
        self.title = title
        # Get proxy configuration from config if available
        self.proxy = CONFIG.external_https_proxy
        super().__init__(file_name=title, **kwargs)
        logger.info(f"Initialized WebParser with title: {title}")

    async def scrape(self, url: str) -> _ScrapeResult:
        """Scrape web page content using Playwright.

        Args:
            url: The URL of the web page to scrape

        Returns:
            HTML, visible text, and document title; empty fields on hard failure
        """
        logger.info(f"Starting web page scraping for URL: {url}")
        empty = _ScrapeResult(html="", visible_text="", page_title="")
        target_url = normalize_wechat_article_url(url)
        if target_url != url:
            logger.info("Rewrote WeChat article URL to avoid captcha: %s", target_url)
        try:
            async with async_playwright() as p:
                kwargs = {}
                # Configure proxy if available
                if self.proxy:
                    kwargs["proxy"] = {"server": self.proxy}
                logger.info("Launching Chromium browser")
                browser = await p.chromium.launch(**kwargs)
                context = await browser.new_context(**browser_context_options())
                page = await context.new_page()

                logger.info(f"Navigating to URL: {target_url}")
                try:
                    await page.goto(
                        target_url,
                        timeout=_GOTO_TIMEOUT_MS,
                        wait_until="domcontentloaded",
                    )
                    logger.info("Initial page load complete")
                except Exception as e:
                    logger.error(f"Error navigating to URL: {str(e)}")
                    await browser.close()
                    return empty

                await wait_for_rendered_content(page)

                page_title = await page.title()
                visible_text = await read_visible_text(page)
                content = await page.content()
                page_url = str(page.url)
                logger.info(
                    "Retrieved %d bytes HTML, %d chars visible text, title=%r",
                    len(content),
                    len(visible_text),
                    page_title[:80] if page_title else "",
                )

                await browser.close()
                logger.info("Browser closed")

            logger.info("Successfully retrieved HTML content")
            return _ScrapeResult(
                html=content,
                visible_text=visible_text,
                page_title=page_title or "",
                page_url=page_url,
            )

        except Exception as e:
            logger.error(f"Failed to scrape web page: {str(e)}")
            return empty

    def parse_into_text(self, content: bytes) -> Document:
        """Parse web page content into a Document object.

        Args:
            content: URL encoded as bytes

        Returns:
            Document object containing the parsed markdown content
        """
        url = endecode.decode_bytes(content)

        logger.info(f"Scraping web page: {url}")
        scrape_result = asyncio.run(self.scrape(url))
        if not scrape_result.html and not scrape_result.visible_text:
            logger.error("Failed to scrape web page (no HTML or visible text)")
            return Document(content=f"Error parsing web page: {url}")

        # Fail the import rather than index the verification page as the article.
        if is_wechat_captcha_page(scrape_result.page_url, scrape_result.visible_text):
            raise RuntimeError(
                f"微信返回了「环境异常」验证页，未取到文章正文: {url}"
            )

        md_text = extract_markdown_from_html(scrape_result.html)
        if not md_text:
            md_text = build_visible_text_fallback(
                scrape_result.visible_text,
                scrape_result.page_title,
            )
            if md_text:
                logger.info(
                    "Trafilatura empty; using Playwright visible-text fallback (%d chars)",
                    len(md_text),
                )

        if not md_text:
            logger.error("Failed to parse web page")
            return Document(content=f"Error parsing web page: {url}")

        metadata = {}
        title_match = re.search(r"^title:\s*(.+)", md_text, re.MULTILINE)
        if title_match:
            extracted_title = title_match.group(1).strip()
            if extracted_title:
                metadata["title"] = extracted_title
                logger.info(
                    f"Extracted article title from trafilatura: {extracted_title}"
                )
        elif scrape_result.page_title:
            metadata["title"] = scrape_result.page_title.strip()
            logger.info(
                "Using page title from Playwright: %s", metadata["title"]
            )
        else:
            logger.info(
                "No title found in trafilatura output, first 200 chars: %r",
                md_text[:200],
            )
        return Document(content=md_text, metadata=metadata)


class WebParser(PipelineParser):
    """Web parser using pipeline pattern.

    This parser chains StdWebParser (for web scraping and HTML to markdown conversion)
    with MarkdownParser (for markdown processing). The pipeline processes content
    sequentially through both parsers.
    """

    # Parser classes to be executed in sequence
    _parser_cls = (StdWebParser, MarkdownParser)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    url = sys.argv[1] if len(sys.argv) > 1 else "https://cloud.tencent.com/document/product/457/6759"
    print(f"\n{'='*60}")
    print(f"URL: {url}")
    print(f"{'='*60}\n")

    parser = WebParser(title="")
    doc = parser.parse_into_text(url.encode())

    print(f"--- metadata ---")
    for k, v in doc.metadata.items():
        print(f"  {k}: {v}")

    print(f"\n--- images ({len(doc.images)}) ---")
    for path in list(doc.images.keys())[:10]:
        print(f"  {path}  ({len(doc.images[path])} chars base64)")

    print(f"\n--- content ({len(doc.content)} chars) ---")
    print(doc.content[:300000])
    if len(doc.content) > 300000:
        print(f"\n... (truncated, total {len(doc.content)} chars)")

    print(f"\n--- chunks ({len(doc.chunks)}) ---")
    for i, chunk in enumerate(doc.chunks[:5]):
        print(f"  [{i}] seq={chunk.seq} range=[{chunk.start}:{chunk.end}] len={len(chunk.content)}")
        print(f"      {chunk.content[:120]}{'...' if len(chunk.content) > 120 else ''}")
    if len(doc.chunks) > 5:
        print(f"  ... ({len(doc.chunks) - 5} more chunks)")
