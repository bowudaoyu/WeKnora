import asyncio
import pathlib
import unittest
from unittest import mock

from docreader.parser.web_parser import (
    StdWebParser,
    browser_context_options,
    build_visible_text_fallback,
    extract_markdown_from_html,
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


if __name__ == "__main__":
    unittest.main()
