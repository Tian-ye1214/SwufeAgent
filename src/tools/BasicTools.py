from pathlib import Path
import asyncio
import inspect
import os
import re
import subprocess
import threading
import time
import mimetypes
from ddgs import DDGS
from infra import logger
import shlex
import platform as _platform
from pydantic_ai import BinaryContent, ImageUrl, ToolReturn
from tools.ExtractFileContent import extract_text
from tools.ImageGeneration import generate_image_from_flux
from config.app_config import get_agent_run_policy, get_env
from skills.SkillsManager import SkillsManager
from skills.SkillsTools import SkillsToolkit
from infra.path_sandbox import resolve_readable_path, runtime_repo_root, work_database_root
from infra.subprocess_runner import run_subprocess
from tools.browser_session import PlaywrightBrowserSession
from cli.render import show_file_diff
from cli.pending_review import PendingReviewStore

_REPO_ROOT = runtime_repo_root()


class BasicToolkit:
    _WORK_DATABASE_ROOT = work_database_root()

    def __init__(
        self,
        skills_manager: SkillsManager,
        *,
        extra_worker_tools: list | None = None,
    ):
        self._base_dir: Path = self._WORK_DATABASE_ROOT
        self._file_lock = threading.Lock()
        self._command_lock = asyncio.Lock()
        self._review_store = PendingReviewStore(self._file_lock)
        self._extra_worker_tools: list = list(extra_worker_tools or [])
        self._ask_user_handler = None
        self._skills_manager = skills_manager
        self._skills_toolkit = SkillsToolkit(skills_manager)
        self._browser_session = PlaywrightBrowserSession()
        self._dangerous_patterns = [
            'rm -rf /',
            'rm -rf /*',
            'mkfs.',
            'dd if=',
            ':(){:|:&};:',
            '> /dev/sda',
            'chmod -R 777 /',
            '| sh',
            '| bash',
        ]
        self._dangerous_start_patterns = [
            'eval ',
            'exec ',
        ]
        self._confirm_patterns = [
            (re.compile(r'\brm\s+-\w*r', re.I), "rm 递归删除"),
            (re.compile(r'\brd\s+/s', re.I), "rd /s 递归删除目录"),
            (re.compile(r'\brmdir\s+/s', re.I), "rmdir /s 递归删除目录"),
            (re.compile(r'\bdel\s+/s', re.I), "del /s 递归删除文件"),
            (re.compile(r'\bRemove-Item\b.*-Recurse', re.I), "Remove-Item -Recurse 递归删除"),
        ]

    @property
    def skills_manager(self) -> SkillsManager:
        return self._skills_manager

    @property
    def review_store(self) -> PendingReviewStore:
        return self._review_store

    def close(self) -> None:
        """进程退出时关闭 Playwright 等资源。"""
        self._browser_session.shutdown()

    def set_task_directory(self, task_name: str) -> Path:
        """
        Set a dedicated work directory for the current task.

        Parameters:
            task_name: Task name; used to create a subdirectory under WorkDatabase.

        Returns:
            Path to the task work directory.
        """
        safe_name = "".join(
            c if c.isalnum() or c in ('_', '-', ' ') else '_' for c in task_name
        )
        safe_name = safe_name.strip()[:50]

        if not safe_name:
            safe_name = "default_task"

        task_dir = self._WORK_DATABASE_ROOT / safe_name
        task_dir.mkdir(parents=True, exist_ok=True)

        self._browser_session.close()
        self._base_dir = task_dir
        logger.info(f"📁 任务工作目录已设置: {task_dir}")

        return task_dir

    def reset_task_directory(self):
        self._browser_session.close()
        self._base_dir = self._WORK_DATABASE_ROOT
        logger.info(f"📁 工作目录已重置为: {self._base_dir}")

    async def _run_blocking(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    def _resolve_path_candidate(self, name: str) -> Path:
        return resolve_readable_path(name, work_base=self._base_dir, repo_root=_REPO_ROOT)

    def _safe_path(self, name: str) -> Path:
        path = self._resolve_path_candidate(name)
        root = self._WORK_DATABASE_ROOT.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Path not under WorkDatabase: {path}")
        return path

    def _readable_path(self, name: str) -> Path:
        return self._resolve_path_candidate(name)

    def _browser_headless_from_env(self) -> bool:
        v = (get_env("BROWSER_HEADLESS", warn=False) or "").strip().lower()
        if v in ("0", "false", "no"):
            return False
        return True

    async def _browser_call(self, fn, **kw) -> str:
        h = self._browser_headless_from_env()
        return await self._run_blocking(fn, headless=h, **kw)

    async def browser_navigate(self, url: str, wait_until: str = "domcontentloaded") -> str:
        """
        Open a URL in a real Chromium browser. Works for static and dynamic pages: navigate first,
        then use browser_get_content, screenshots, etc. as needed.

        Requires: pip install playwright && playwright install chromium
        Show the browser window: set env BROWSER_HEADLESS=0

        Parameters:
            url: Full URL (https://...)
            wait_until: Load wait strategy; one of domcontentloaded, load, networkidle (default domcontentloaded)
        """
        return await self._browser_call(
            self._browser_session.navigate, url=url, wait_until=wait_until
        )

    async def browser_get_content(self) -> str:
        """
        Return visible text from the current page (body innerText), for reading dynamically rendered content.
        Returns the full text with no length truncation.
        """
        return await self._browser_call(self._browser_session.get_content)

    async def browser_screenshot(self, name: str, full_page: bool = False) -> str:
        """
        Capture a screenshot of the current page and save it to the given path.
        Relative paths use the same rules as read_file (WorkDatabase or src/ under repo).

        Parameters:
            name: File path, e.g. page.png or an absolute path
            full_page: If True, capture the full scrollable page
        """
        try:
            path = self._resolve_path_candidate(name)
            path.parent.mkdir(parents=True, exist_ok=True)
        except ValueError as e:
            return str(e)
        return await self._browser_call(
            self._browser_session.screenshot,
            filename=str(path),
            full_page=full_page,
        )

    async def browser_click(self, selector: str) -> str:
        """
        Click an element on the current page. selector uses Playwright syntax (CSS, text=..., etc.).

        Parameters:
            selector: e.g. #submit, text=Sign in
        """
        return await self._browser_call(self._browser_session.click, selector=selector)

    async def browser_fill(self, selector: str, text: str) -> str:
        """
        Fill an input field with text (clears the field first, then types the value).

        Parameters:
            selector: CSS or other Playwright selector for the input
            text: Text to enter
        """
        return await self._browser_call(
            self._browser_session.fill, selector=selector, text=text
        )

    async def browser_press_key(self, key: str) -> str:
        """
        Send a keyboard key to the current page (e.g. Enter, Tab).

        Parameters:
            key: Playwright key name, e.g. Enter, ArrowDown
        """
        return await self._browser_call(self._browser_session.press, key=key)

    async def browser_wait_for_selector(self, selector: str, timeout_ms: int = 30000) -> str:
        """
        Wait until an element appears in the DOM (useful for SPAs and async-loaded UI).

        Parameters:
            selector: Playwright selector
            timeout_ms: Timeout in milliseconds
        """
        return await self._browser_call(
            self._browser_session.wait_for_selector,
            selector=selector,
            timeout_ms=timeout_ms,
        )

    async def browser_evaluate(self, javascript_expression: str) -> str:
        """
        Evaluate a JavaScript expression in the page context and return the result (page.evaluate).
        Example: document.querySelector('h1')?.innerText

        Parameters:
            javascript_expression: A single-line expression to evaluate
        """
        return await self._browser_call(
            self._browser_session.run_javascript, expression=javascript_expression
        )

    async def browser_close(self) -> str:
        """Close the Playwright browser process and release resources; the next action will start a new browser."""
        await self._run_blocking(self._browser_session.close)
        return "Browser closed"

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if command contains dangerous patterns"""
        command_lower = command.lower().strip()
        for pattern in self._dangerous_patterns:
            if pattern.lower() in command_lower:
                return False, f"Dangerous command pattern detected: '{pattern}'"
        for pattern in self._dangerous_start_patterns:
            if command_lower.startswith(pattern.lower()):
                return False, f"Dangerous command pattern detected: '{pattern}'"
        if "npx" in command_lower and "clawhub" in command_lower:
            if re.search(r"\bclawhub\s+install(?:\s|$)", command_lower):
                return False, (
                    "Blocked bare `npx clawhub install`. Use: "
                    f"`npx clawhub --dir src/skills install <slug>`. Repo root: {_REPO_ROOT}"
                )
            if re.search(r'--dir(?:=|\s+)["\']?[a-zA-Z]:', command):
                return False, (
                    "Blocked `--dir` with a drive letter; use `--dir src/skills` relative to repo root. "
                    f"Repo root: {_REPO_ROOT}"
                )
        return True, ""

    def _command_needs_confirm(self, command: str) -> str | None:
        """Return a short reason if the command performs a recursive delete, else None."""
        for pattern, reason in self._confirm_patterns:
            if pattern.search(command):
                return reason
        return None

    def set_ask_user_handler(self, handler):
        """
        Replace the underlying implementation of ask_user (for non-terminal use, e.g. QQ Bot, Web API).
        handler may be sync or async: (question: str) -> str | None, or async (question: str) -> str | None.
        Pass None to restore the default terminal input.
        """
        self._ask_user_handler = handler

    async def ask_user(self, question: str) -> str:
        """
        Ask the user a question and return their reply.
        Parameters:
            question: The question to ask
        Returns:
            The user's answer
        """
        if self._ask_user_handler is not None:
            if inspect.iscoroutinefunction(self._ask_user_handler):
                result = await self._ask_user_handler(question)
            else:
                result = await asyncio.to_thread(self._ask_user_handler, question)
            return result if result is not None else "(User did not reply)"

        logger.info("=" * 50)
        logger.info("🤔 Agent 需要您的帮助")
        logger.info("=" * 50)
        logger.info(f"问题: {question}")

        user_response = (await asyncio.to_thread(input, "📝 您的回复: ")).strip()
        logger.info(f"用户回答: {user_response}")

        return user_response

    def read_file(self, name: str) -> str:
        """
        Read file contents.
        Parameters:
            name: File name/path
        """
        try:
            file_path = self._readable_path(name)
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return content if content else "File is empty"
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"Read error: {e}"

    def list_files(self, directory: str = "") -> str:
        """
        List all files and folders in a directory.
        Parameters:
            directory: Optional, subdirectory path, defaults to root directory
        """
        try:
            target_dir = (
                self._readable_path(directory) if directory else self._base_dir
            )
            if not target_dir.exists():
                return f"Error: Directory '{directory}' does not exist"

            items = []
            base_r = self._base_dir.resolve()
            for item in sorted(target_dir.iterdir()):
                try:
                    rel_path = str(item.relative_to(base_r))
                except ValueError:
                    rel_path = str(item)
                if item.is_dir():
                    items.append(f"{rel_path}/")
                else:
                    size = item.stat().st_size
                    items.append(f"{rel_path} ({size} bytes)")

            return "\n".join(items) if items else "Directory is empty"
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"Error listing files: {e}"

    def write_file(self, name: str, content: str) -> str:
        """
        Create or overwrite a file with SHORT content only.

        Parameters:
            name: File name/path (relative to WorkDatabase directory)
            content: Content to write
        """
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._safe_path(name)
            os.makedirs(file_path.parent, exist_ok=True)
            with self._file_lock:
                try:
                    old_content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
                except Exception:
                    old_content = ""
                file_path.write_text(content, encoding="utf-8")
            show_file_diff(old_content, content, path=name)
            self._review_store.register(file_path, name=name, baseline=old_content, snapshot=content)
            content_len = len(content)
            return f"File '{name}' written successfully ({content_len} characters)"
        except ValueError as e:
            return f"Security error: {e}"
        except PermissionError as e:
            return f"Permission error: Cannot write to '{name}' - {e}"
        except Exception as e:
            return f"Write error: {type(e).__name__} - {e}"

    def edit_file(self, name: str, old_string: str, new_string: str) -> str:
        """
        Edit an existing file by replacing an EXACT, UNIQUE snippet (string replace).
        Prefer this over write_file when modifying an existing file: it makes a precise,
        local change and shows a colored diff instead of rewriting the whole file.

        Parameters:
            name: File path (relative to WorkDatabase directory).
            old_string: Exact text to replace. Must occur EXACTLY ONCE in the file —
                include enough surrounding context (indentation, neighboring lines) to be unique.
            new_string: Replacement text.
        """
        try:
            file_path = self._safe_path(name)
            if not file_path.exists():
                return f"Edit error: file '{name}' does not exist (use write_file to create it)"
            with self._file_lock:
                old_content = file_path.read_text(encoding="utf-8")
                count = old_content.count(old_string)
                if count == 0:
                    return f"Edit error: old_string not found in '{name}'"
                if count > 1:
                    return f"Edit error: old_string is not unique in '{name}' (found {count}×); add more surrounding context"
                new_content = old_content.replace(old_string, new_string, 1)
                if new_content == old_content:
                    return "Edit error: new_string is identical to old_string; nothing to change"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            added, deleted, modified = show_file_diff(old_content, new_content, path=name)
            self._review_store.register(file_path, name=name, baseline=old_content, snapshot=new_content)
            return f"Edited '{name}' successfully (+{added} -{deleted} ~{modified})"
        except ValueError as e:
            return f"Security error: {e}"
        except PermissionError as e:
            return f"Permission error: Cannot edit '{name}' - {e}"
        except Exception as e:
            return f"Edit error: {type(e).__name__} - {e}"

    def search_in_files(self, keyword: str, file_extension: str = None) -> str:
        """
        Search for a keyword in files.
        Parameters:
            keyword: Keyword to search for
            file_extension: Optional, limit search to specific file types, e.g., ".py", ".txt"
        """
        results = []
        try:
            for file_path in self._base_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_extension and file_path.suffix != file_extension:
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if keyword.lower() in line.lower():
                                rel_path = file_path.relative_to(self._base_dir)
                                results.append(f"{rel_path}:{line_num}: {line.strip()}")
                except Exception:
                    continue

            if results:
                return f"Found {len(results)} matches:\n" + "\n".join(results)
            return "No matches found"
        except Exception as e:
            return f"Search error: {e}"

    def search_web(self, query: str, max_results: int = 5) -> str:
        """
        Search web pages. Returns a list of search results (title, link, summary).
        Parameters:
            query: Search keywords
            max_results: Maximum number of results to return, defaults to 5
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results, region='cn-zh'))

            if not results:
                logger.warning("⚠️ 没有找到相关搜索结果")
                return "No relevant search results found."

            output = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                link = result.get('href', 'No link')
                snippet = result.get('body', 'No summary')
                output.append(f"{i}. {title}\n   Link: {link}\n   Summary: {snippet}\n")

            result_text = "\n".join(output)
            return result_text
        except Exception as e:
            logger.error(f"❌ 搜索出错: {e}")
            return f"Error during search: {e}"

    async def run_command(self, command: str, timeout: int = 60) -> str:
        """
        Execute a Shell/terminal command.
        Parameters:
            command: Command to execute
            timeout: Timeout in seconds, defaults to 60
        """
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            return f"Security error: {reason}"

        danger = self._command_needs_confirm(command)
        if danger:
            answer = (await self.ask_user(f"⚠ 该命令将递归删除文件:\n{command}\n确认执行？(y/N)")).strip().lower()
            if answer not in ("y", "yes", "是", "确认"):
                return f"已取消执行（用户未确认）: {danger}"

        try:
            async with self._command_lock:
                policy = get_agent_run_policy()
                timeout = policy.clamp_command_timeout(timeout)
                use_shell = any(c in command for c in ['|', '>', '<', '&&', '||', ';', '*', '?'])
                if use_shell and re.search(r"\b(start|nohup|setsid)\b|&\s*$", command, re.I):
                    return "Security error: background shell processes are not allowed"
                cwd = str(self._base_dir.resolve())
                env = None
                if re.search(r"\bclawhub\b", command, re.I):
                    cwd = str(_REPO_ROOT)
                    if "--workdir" not in command:
                        env = os.environ | {"CLAWHUB_WORKDIR": cwd}

                shell = use_shell or _platform.system() == "Windows"
                args = command if shell else shlex.split(command)
                stdout, stderr, return_code = await run_subprocess(
                    args, shell=shell, cwd=cwd, env=env, timeout=timeout
                )
                output = stdout + stderr
                return f"Return code: {return_code}\nOutput:\n{output}" if output else f"Execution completed, return code: {return_code}"
        except subprocess.TimeoutExpired:
            return f"Error: Command execution timed out ({timeout} seconds)"
        except Exception as e:
            return f"Execution error: {e}"

    def read_image(self, image_path: str) -> ToolReturn | str:
        """
        Read an image from a local file path or URL, and return its content for visual analysis by the AI model.
        Use this tool whenever you need to VIEW, ANALYZE, DESCRIBE, or UNDERSTAND an image.

        This is the ONLY tool that can make you actually SEE image content. Other file reading
        tools (read_file, extract_text) cannot process image data for visual understanding.

        Keywords that should trigger this tool:
        - "看图" / "查看图片" / "分析图片" / "描述图片"
        - "view image" / "analyze image" / "describe image"
        - Any task requiring understanding of visual content in an image

        Supported formats: jpg, jpeg, png, gif, webp, bmp

        Parameters:
            image_path: Local file path OR image URL.
                - Local: absolute (C:\\Users\\PC\\Desktop\\photo.jpg) or relative path within WorkDatabase.
                - URL: http/https link to an image (e.g., https://example.com/photo.jpg).
        """
        is_url = image_path.startswith(('http://', 'https://'))

        try:
            if is_url:
                return self._read_image_from_url(image_path)
            else:
                return self._read_image_from_file(image_path)
        except Exception as e:
            return f"Error reading image: {type(e).__name__} - {e}"

    def _read_image_from_file(self, image_path: str) -> ToolReturn | str:
        """Load an image from a local file path."""
        try:
            path = self._readable_path(image_path)
        except ValueError as e:
            return f"Security error: {e}"

        if not path.exists():
            return f"Error: Image file not found: {image_path}"

        supported_ext = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        if path.suffix.lower() not in supported_ext:
            return f"Error: Unsupported image format '{path.suffix}'. Supported: {', '.join(supported_ext)}"

        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = 'image/png'

        with open(path, 'rb') as f:
            image_data = f.read()

        return ToolReturn(
            return_value=f"Image loaded successfully: {path.name} ",
            content=[
                f"Image file: {path.name}",
                BinaryContent(data=image_data, media_type=mime_type),
            ]
        )

    def _read_image_from_url(self, url: str) -> ToolReturn:
        """Pass the image URL to the model for analysis without downloading the file."""
        return ToolReturn(
            return_value=f"Image URL passed to model: {url}",
            content=[
                f"Image from URL: {url}",
                ImageUrl(url=url),
            ]
        )

    async def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, max_wait_time: int = 300) -> ToolReturn | str:
        """
        Generate images using AI model. Use this tool whenever the user asks to create, generate, make, or produce an image, picture, photo, illustration, artwork, or visual content.

        This is the PRIMARY tool for ALL image generation requests. Keywords that should trigger this tool:
        - "create image" / "generate image" / "make a picture" / "draw" / "paint" / "illustrate"
        - "give me an image" / "produce image"
        - Any request involving creating visual content, artwork, diagrams, or images (any language)

        Parameters:
            prompt: The text description of what image to generate. Be detailed and specific about the visual content, style, composition, colors, mood, etc. This is the most important parameter.
            width: Image width in pixels. Default: 1024. Common values: 512, 768, 1024, 1536, etc.
            height: Image height in pixels. Default: 1024. Common values: 512, 768, 1024, 1536, etc.
            max_wait_time: Maximum wait time in seconds. Default: 300 (5 minutes).

        Returns:
            Success: The generated image displayed inline plus generation details.
            Failure: Returns an error message.
        """
        result = await generate_image_from_flux(prompt, width=width, height=height, max_wait_time=max_wait_time)
        if not isinstance(result, tuple):
            return result

        image_bytes, mime_type, info_text = result
        ext = mimetypes.guess_extension(mime_type) or ".jpg"
        name = f"generated_{int(time.time())}{ext}"
        try:
            path = self._resolve_path_candidate(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_bytes, image_bytes)
        except ValueError as e:
            return f"Security error: {e}"

        return ToolReturn(
            return_value=f"{info_text}\nSaved to: {path.name}",
            content=[
                info_text,
                BinaryContent(data=image_bytes, media_type=mime_type),
            ]
        )

    def worker_tool_groups(self, *, include_browser: bool) -> dict[str, list]:
        """Worker tools grouped for resident tools and deferred capabilities."""
        browser = [
            self.browser_navigate,
            self.browser_get_content,
            self.browser_screenshot,
            self.browser_click,
            self.browser_fill,
            self.browser_press_key,
            self.browser_wait_for_selector,
            self.browser_evaluate,
            self.browser_close,
        ] if include_browser else []
        groups = {
            "core": [
                self.list_files,
                self.read_file,
                self.search_in_files,
                self.search_web,
                self.ask_user,
            ],
            "file_mutation": [
                self.write_file,
                self.edit_file,
            ],
            "execution": [
                self.run_command,
            ],
            "media": [
                self.generate_image,
                self.read_image,
                extract_text,
            ],
            "memory": list(self._extra_worker_tools),
            "skills": list(self._skills_toolkit.tools),
        }
        if browser:
            groups["browser"] = browser
        return {name: tools for name, tools in groups.items() if tools}

    def _worker_tools(self, *, include_browser: bool) -> list:
        """扁平工具集（Coordinator 直用）：由 worker_tool_groups 拍平，单一真相源。"""
        tools: list = []
        for group in self.worker_tool_groups(include_browser=include_browser).values():
            tools.extend(group)
        return tools

    @property
    def workers_tools(self) -> list:
        """List of tool callables exposed to the Worker Agent."""
        return self._worker_tools(include_browser=True)
