import asyncio
import pathlib
import unittest
from unittest import mock

from docreader.parser.web_parser import (
    StdWebParser,
    browser_context_options,
    build_visible_text_fallback,
    extract_markdown_from_html,
    is_wechat_captcha_page,
    normalize_wechat_article_url,
    sections_to_divs,
)


class TestWebParserHelpers(unittest.TestCase):
    def test_extract_markdown_empty_html(self):
        self.assertIsNone(extract_markdown_from_html(""))
        self.assertIsNone(extract_markdown_from_html("   "))

    def test_extract_markdown_article_html(self):
        html = """
        <html><head><title>Demo</title></head><body>
        <article><h1>Hello</h1><p>World paragraph with enough text for extraction.</p></article>
        </body></html>
        """
        md = extract_markdown_from_html(html)
        self.assertIsNotNone(md)
        self.assertIn("Hello", md)

    def test_extract_markdown_wechat_article(self):
        """Regression: trafilatura returns only front matter for this page.

        Fixture is a trimmed capture of a real WeChat Official Account article
        (mp.weixin.qq.com/s/fooVR2WJ2pP1g7xxy4TJ_Q). Its body lives in nested
        <section> elements; without the rewrite the extraction is 139 chars of
        page furniture and none of the article.
        """
        html = (
            pathlib.Path(__file__).resolve().parents[1]
            / "testdata"
            / "wechat_article_sections.html"
        ).read_text(encoding="utf-8")
        md = extract_markdown_from_html(html)
        self.assertIsNotNone(md)
        self.assertIn("今天开始按单元整理辽宁省博物馆", md)

    def test_sections_to_divs_rewrites_both_tags(self):
        html = '<section class="a"><SECTION>x</SECTION></section >'
        self.assertNotIn("section", sections_to_divs(html).lower())

    def test_build_fallback_too_short(self):
        self.assertIsNone(build_visible_text_fallback("short"))
        self.assertIsNone(build_visible_text_fallback(""))

    def test_build_fallback_with_title(self):
        text = "A" * 60
        md = build_visible_text_fallback(text, page_title="WeKnora")
        self.assertIsNotNone(md)
        self.assertTrue(md.startswith("# WeKnora"))
        self.assertIn(text, md)

    def test_build_fallback_without_title(self):
        text = "B" * 60
        md = build_visible_text_fallback(text, page_title="")
        self.assertEqual(md, text)


class TestWechatArticleUrl(unittest.TestCase):
    _LONG = (
        "https://mp.weixin.qq.com/s?__biz=MzU1NzkxNjc4NQ==&mid=2247488223"
        "&idx=1&sn=3dc1b3b72c5d9b88274259f42c4c38c4"
    )

    def test_long_link_gets_placeholder_chksm(self):
        """WeChat sends long links without `chksm` to the captcha page."""
        self.assertEqual(normalize_wechat_article_url(self._LONG), self._LONG + "&chksm=0")

    def test_poc_token_and_empty_chksm_are_dropped(self):
        url = self._LONG + "&chksm=&poc_token=HBucq2qjKE7_43aQ&scene=21#rd"
        self.assertEqual(
            normalize_wechat_article_url(url),
            self._LONG + "&scene=21&chksm=0#rd",
        )

    def test_existing_chksm_is_kept(self):
        url = self._LONG + "&chksm=fd3a1c2b"
        self.assertEqual(normalize_wechat_article_url(url), url)

    def test_other_urls_untouched(self):
        for url in (
            "https://mp.weixin.qq.com/s/fooVR2WJ2pP1g7xxy4TJ_Q",
            "https://example.com/s?__biz=a&sn=b",
            "https://cloud.tencent.com/document/product/457/6759",
        ):
            self.assertEqual(normalize_wechat_article_url(url), url)

    def test_captcha_page_detection(self):
        self.assertTrue(
            is_wechat_captcha_page(
                "https://mp.weixin.qq.com/mp/wappoc_appmsgcaptcha?poc_token=x", ""
            )
        )
        self.assertTrue(
            is_wechat_captcha_page("", "环境异常\n当前环境异常，完成验证后即可继续访问。")
        )
        self.assertFalse(is_wechat_captcha_page(self._LONG, "正文"))


class TestBrowserContextOptions(unittest.TestCase):
    def test_user_agent_is_not_headless(self):
        """WeChat answers a `HeadlessChrome` UA with a verification stub."""
        ua = browser_context_options()["user_agent"]
        self.assertNotIn("Headless", ua)
        self.assertIn("Chrome/", ua)


class TestScrapeBrowserSetup(unittest.IsolatedAsyncioTestCase):
    async def test_scrape_uses_chromium_with_spoofed_user_agent(self):
        """Only chromium is deployed, and it must not use its default UA."""
        page = mock.AsyncMock()
        page.title.return_value = "T"
        page.evaluate.return_value = "text"
        page.content.return_value = "<html></html>"
        context = mock.AsyncMock()
        context.new_page.return_value = page
        browser = mock.AsyncMock()
        browser.new_context.return_value = context
        chromium = mock.AsyncMock()
        chromium.launch.return_value = browser
        webkit = mock.AsyncMock()

        pw = mock.MagicMock()
        pw.chromium = chromium
        pw.webkit = webkit
        ctxmgr = mock.AsyncMock()
        ctxmgr.__aenter__.return_value = pw

        with mock.patch(
            "docreader.parser.web_parser.async_playwright", return_value=ctxmgr
        ):
            await StdWebParser(title="t").scrape("https://example.com")

        chromium.launch.assert_awaited_once()
        webkit.launch.assert_not_awaited()
        ua = browser.new_context.await_args.kwargs["user_agent"]
        self.assertNotIn("Headless", ua)


class TestParseRejectsCaptcha(unittest.TestCase):
    def test_captcha_page_fails_instead_of_being_indexed(self):
        """Regression: the "环境异常" stub was stored as the article body."""
        from docreader.parser.web_parser import _ScrapeResult

        stub = _ScrapeResult(
            html="<html><title>环境异常</title><body>当前环境异常，完成验证后即可继续访问。</body></html>",
            visible_text="环境异常\n当前环境异常，完成验证后即可继续访问。\n去验证",
            page_title="微信公众平台",
            page_url="https://mp.weixin.qq.com/mp/wappoc_appmsgcaptcha?poc_token=x",
        )
        parser = StdWebParser(title="t")
        with mock.patch.object(StdWebParser, "scrape", mock.AsyncMock(return_value=stub)):
            with self.assertRaisesRegex(RuntimeError, "环境异常"):
                parser.parse_into_text(b"https://mp.weixin.qq.com/s?__biz=a&sn=b")


if __name__ == "__main__":
    unittest.main()
