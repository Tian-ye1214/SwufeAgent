from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable

import logger
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright


class PlaywrightBrowserSession:
    """持久 Chromium 会话；内部仅在单后台线程上触碰 Playwright 对象。"""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")
        self._run_lock = threading.Lock()
        self._stopped = False
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._headless: bool | None = None


    def _run(self, fn: Callable[[], Any], timeout: float = 120.0) -> Any:
        if self._stopped:
            raise RuntimeError("Playwright session has been shut down")
        with self._run_lock:
            future = self._executor.submit(fn)
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError:
                future.cancel()
                raise

    def close(self) -> None:
        """关闭浏览器（不停止线程池）。从未启动过时直接返回，避免为空操作白白拉起后台线程。"""
        if self._stopped:
            return
        if self._pw is None and self._browser is None and self._context is None and self._page is None:
            return
        try:
            self._run(self._close_impl, timeout=60.0)
        except Exception as e:
            logger.warning(f"[browser] close: {e}")

    def shutdown(self) -> None:
        """关闭浏览器并停止后台线程池（进程退出时调用）。"""
        if self._stopped:
            return
        self.close()
        self._stopped = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _close_impl(self) -> None:
        for attr, closer in (
            ("_page", lambda o: o.close()),
            ("_context", lambda o: o.close()),
            ("_browser", lambda o: o.close()),
        ):
            obj = getattr(self, attr)
            if obj is not None:
                try:
                    closer(obj)
                except Exception as e:
                    logger.warning(f"[browser] 关闭 {attr} 时: {e}")

        if self._pw is not None:
            try:
                self._pw.stop()
            except Exception as e:
                logger.warning(f"[browser] stop playwright: {e}")

        self._page = None
        self._context = None
        self._browser = None
        self._pw = None
        self._headless = None

    def _ensure_started_impl(self, headless: bool) -> str | None:
        if self._page is not None:
            if self._headless is not None and self._headless != headless:
                self._close_impl()
            else:
                return None
        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=headless)
            self._context = self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                locale="zh-CN",
            )
            self._page = self._context.new_page()
            self._page.set_default_timeout(30_000)
            self._headless = headless
            return None
        except Exception as e:
            self._close_impl()
            return f"启动浏览器失败: {type(e).__name__}: {e}"

    def navigate(self, url: str, headless: bool, wait_until: str = "domcontentloaded") -> str:
        def _work() -> str:
            err = self._ensure_started_impl(headless)
            if err:
                return err
            assert self._page is not None
            try:
                self._page.goto(url, wait_until=wait_until, timeout=60_000)
                msg = f"OK\nURL: {self._page.url}\nTitle: {self._page.title()}"
                if headless:
                    msg += (
                        "\n\n说明: 当前为无头 Chromium，页面在 Agent 进程内打开，"
                        "不会出现在您日常使用的 Chrome/Edge 窗口里。"
                        "若要弹出可見窗口，请设置环境变量 BROWSER_HEADLESS=0（仍为独立浏览器，非系统默认）。"
                    )
                return msg
            except Exception as e:
                return f"导航失败: {type(e).__name__}: {e}"
        return self._run(_work)

    def get_content(self, headless: bool) -> str:
        def _work() -> str:
            err = self._ensure_started_impl(headless)
            if err:
                return err
            assert self._page is not None
            try:
                text = self._page.evaluate("""() => document.body ? document.body.innerText : ''""")
                if not isinstance(text, str):
                    text = str(text)
                text = text.strip()
                return f"URL: {self._page.url}\n---\n{text or '(无文本内容)'}"
            except Exception as e:
                return f"读取页面文本失败: {type(e).__name__}: {e}"
        return self._run(_work)

    def screenshot(self, headless: bool, filename: str, full_page: bool = False) -> str:
        def _work() -> str:
            err = self._ensure_started_impl(headless)
            if err:
                return err
            assert self._page is not None
            try:
                self._page.screenshot(path=filename, full_page=full_page)
                return f"截图已保存: {filename}"
            except Exception as e:
                return f"截图失败: {type(e).__name__}: {e}"
        return self._run(_work)

    def click(self, headless: bool, selector: str) -> str:
        def _work() -> str:
            err = self._ensure_started_impl(headless)
            if err:
                return err
            assert self._page is not None
            try:
                self._page.click(selector, timeout=30_000)
                return f"已点击: {selector}"
            except Exception as e:
                return f"点击失败: {type(e).__name__}: {e}"
        return self._run(_work)

    def fill(self, headless: bool, selector: str, text: str) -> str:
        def _work() -> str:
            err = self._ensure_started_impl(headless)
            if err:
                return err
            assert self._page is not None
            try:
                self._page.fill(selector, text, timeout=30_000)
                return f"已填入 {selector}"
            except Exception as e:
                return f"填充失败: {type(e).__name__}: {e}"
        return self._run(_work)

    def press(self, headless: bool, key: str) -> str:
        def _work() -> str:
            err = self._ensure_started_impl(headless)
            if err:
                return err
            assert self._page is not None
            try:
                self._page.keyboard.press(key)
                return f"已按键: {key}"
            except Exception as e:
                return f"按键失败: {type(e).__name__}: {e}"
        return self._run(_work)

    def wait_for_selector(self, headless: bool, selector: str, timeout_ms: int = 30_000) -> str:
        timeout_ms = max(1, min(int(timeout_ms), 110_000))

        def _work() -> str:
            err = self._ensure_started_impl(headless)
            if err:
                return err
            assert self._page is not None
            try:
                self._page.wait_for_selector(selector, timeout=timeout_ms)
                return f"已出现元素: {selector}"
            except Exception as e:
                return f"等待元素超时或失败: {type(e).__name__}: {e}"
        return self._run(_work)

    def run_javascript(self, headless: bool, expression: str) -> str:
        def _work() -> str:
            err = self._ensure_started_impl(headless)
            if err:
                return err
            assert self._page is not None
            try:
                result = self._page.evaluate(expression)
                return f"结果: {repr(result)}"
            except Exception as e:
                return f"执行脚本失败: {type(e).__name__}: {e}"
        return self._run(_work)
