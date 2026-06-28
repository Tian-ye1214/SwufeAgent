from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any, Callable

from redlotus.infra import logger

if TYPE_CHECKING:  # playwright 为可选依赖（extra: browser），仅在实际使用浏览器时才需要
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright


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
                self._discard_poisoned_executor()
                raise

    def _discard_poisoned_executor(self) -> None:
        poisoned = self._executor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._headless = None
        try:
            poisoned.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

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
            try:
                from playwright.sync_api import sync_playwright
            except ModuleNotFoundError:
                return "未安装 playwright；请先 `pip install redlotus[browser]` 再 `playwright install chromium`。"
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

    def _page_action(
        self,
        headless: bool,
        fn: Callable[[Page], str],
        error_prefix: str,
    ) -> str:
        def _work() -> str:
            err = self._ensure_started_impl(headless)
            if err:
                return err
            assert self._page is not None
            try:
                return fn(self._page)
            except Exception as e:
                return f"{error_prefix}: {type(e).__name__}: {e}"
        return self._run(_work)

    def navigate(self, url: str, headless: bool, wait_until: str = "domcontentloaded") -> str:
        def _action(page: Page) -> str:
            page.goto(url, wait_until=wait_until, timeout=60_000)
            msg = f"OK\nURL: {page.url}\nTitle: {page.title()}"
            if headless:
                msg += (
                    "\n\n说明: 当前为无头 Chromium，页面在 Agent 进程内打开，"
                    "不会出现在您日常使用的 Chrome/Edge 窗口里。"
                    "若要弹出可見窗口，请设置环境变量 BROWSER_HEADLESS=0（仍为独立浏览器，非系统默认）。"
                )
            return msg
        return self._page_action(headless, _action, "导航失败")

    def get_content(self, headless: bool) -> str:
        def _action(page: Page) -> str:
            text = page.evaluate("""() => document.body ? document.body.innerText : ''""")
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            return f"URL: {page.url}\n---\n{text or '(无文本内容)'}"
        return self._page_action(headless, _action, "读取页面文本失败")

    def screenshot(self, headless: bool, filename: str, full_page: bool = False) -> str:
        def _action(page: Page) -> str:
            page.screenshot(path=filename, full_page=full_page)
            return f"截图已保存: {filename}"
        return self._page_action(headless, _action, "截图失败")

    def click(self, headless: bool, selector: str) -> str:
        def _action(page: Page) -> str:
            page.click(selector, timeout=30_000)
            return f"已点击: {selector}"
        return self._page_action(headless, _action, "点击失败")

    def fill(self, headless: bool, selector: str, text: str) -> str:
        def _action(page: Page) -> str:
            page.fill(selector, text, timeout=30_000)
            return f"已填入 {selector}"
        return self._page_action(headless, _action, "填充失败")

    def press(self, headless: bool, key: str) -> str:
        def _action(page: Page) -> str:
            page.keyboard.press(key)
            return f"已按键: {key}"
        return self._page_action(headless, _action, "按键失败")

    def wait_for_selector(self, headless: bool, selector: str, timeout_ms: int = 30_000) -> str:
        timeout_ms = max(1, min(int(timeout_ms), 110_000))

        def _action(page: Page) -> str:
            page.wait_for_selector(selector, timeout=timeout_ms)
            return f"已出现元素: {selector}"
        return self._page_action(headless, _action, "等待元素超时或失败")

    def run_javascript(self, headless: bool, expression: str) -> str:
        def _action(page: Page) -> str:
            result = page.evaluate(expression)
            return f"结果: {repr(result)}"
        return self._page_action(headless, _action, "执行脚本失败")
