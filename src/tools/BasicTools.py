from pathlib import Path
import asyncio
import inspect
import os
import re
import subprocess
import mimetypes
from ddgs import DDGS
import logger
import shlex
import platform as _platform
from pydantic_ai import BinaryContent, ImageUrl, ToolReturn
from tools.ExtractFileContent import extract_text
from app_config import get_env
from skills.SkillsManager import SkillsManager
from skills.SkillsTools import SkillsToolkit
from tools.browser_session import PlaywrightBrowserSession

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class BasicToolkit:
    _WORK_DATABASE_ROOT = _REPO_ROOT / "WorkDatabase"

    def __init__(
        self,
        skills_manager: SkillsManager,
        *,
        extra_worker_tools: list | None = None,
    ):
        self._base_dir: Path = self._WORK_DATABASE_ROOT
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

    @property
    def skills_manager(self) -> SkillsManager:
        return self._skills_manager

    @property
    def base_dir(self) -> Path:
        return self._base_dir

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

    def _browser_headless_from_env(self) -> bool:
        v = (get_env("BROWSER_HEADLESS", warn=False) or "").strip().lower()
        if v in ("0", "false", "no"):
            return False
        return True

    def browser_navigate(self, url: str, wait_until: str = "domcontentloaded") -> str:
        """
        Open a URL in a real Chromium browser. Works for static and dynamic pages: navigate first,
        then use browser_get_content, screenshots, etc. as needed.

        Requires: pip install playwright && playwright install chromium
        Show the browser window: set env BROWSER_HEADLESS=0

        Parameters:
            url: Full URL (https://...)
            wait_until: Load wait strategy; one of domcontentloaded, load, networkidle (default domcontentloaded)
        """
        h = self._browser_headless_from_env()
        return self._browser_session.navigate(url, headless=h, wait_until=wait_until)

    def browser_get_content(self) -> str:
        """
        Return visible text from the current page (body innerText), for reading dynamically rendered content.
        Returns the full text with no length truncation.
        """
        h = self._browser_headless_from_env()
        return self._browser_session.get_content(headless=h)

    def browser_screenshot(self, name: str, full_page: bool = False) -> str:
        """
        Capture a screenshot of the current page and save it to the given path.
        Relative paths use the same rules as read_file (WorkDatabase or src/ under repo).

        Parameters:
            name: File path, e.g. page.png or an absolute path
            full_page: If True, capture the full scrollable page
        """
        h = self._browser_headless_from_env()
        try:
            path = self._resolve_path_candidate(name)
            path.parent.mkdir(parents=True, exist_ok=True)
        except ValueError as e:
            return str(e)
        return self._browser_session.screenshot(headless=h, filename=str(path), full_page=full_page)

    def browser_click(self, selector: str) -> str:
        """
        Click an element on the current page. selector uses Playwright syntax (CSS, text=..., etc.).

        Parameters:
            selector: e.g. #submit, text=Sign in
        """
        h = self._browser_headless_from_env()
        return self._browser_session.click(headless=h, selector=selector)

    def browser_fill(self, selector: str, text: str) -> str:
        """
        Fill an input field with text (clears the field first, then types the value).

        Parameters:
            selector: CSS or other Playwright selector for the input
            text: Text to enter
        """
        h = self._browser_headless_from_env()
        return self._browser_session.fill(headless=h, selector=selector, text=text)

    def browser_press_key(self, key: str) -> str:
        """
        Send a keyboard key to the current page (e.g. Enter, Tab).

        Parameters:
            key: Playwright key name, e.g. Enter, ArrowDown
        """
        h = self._browser_headless_from_env()
        return self._browser_session.press(headless=h, key=key)

    def browser_wait_for_selector(self, selector: str, timeout_ms: int = 30000) -> str:
        """
        Wait until an element appears in the DOM (useful for SPAs and async-loaded UI).

        Parameters:
            selector: Playwright selector
            timeout_ms: Timeout in milliseconds
        """
        h = self._browser_headless_from_env()
        return self._browser_session.wait_for_selector(
            headless=h, selector=selector, timeout_ms=timeout_ms
        )

    def browser_evaluate(self, javascript_expression: str) -> str:
        """
        Evaluate a JavaScript expression in the page context and return the result (page.evaluate).
        Example: document.querySelector('h1')?.innerText

        Parameters:
            javascript_expression: A single-line expression to evaluate
        """
        h = self._browser_headless_from_env()
        return self._browser_session.run_javascript(headless=h, expression=javascript_expression)

    def browser_close(self) -> str:
        """Close the Playwright browser process and release resources; the next action will start a new browser."""
        self._browser_session.close()
        return "Browser closed"

    def _resolve_path_candidate(self, name: str) -> Path:
        """Resolve path (same rules as _safe_path) without workspace/repo boundary check — for read/list only."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Path name must not be empty")
        # Align with system prompt: bare "skills" means src/skills under the repo
        if name.replace("\\", "/").strip("/").rstrip("/") == "skills":
            name = "src/skills"

        base_r = self._base_dir.resolve()
        repo_r = _REPO_ROOT.resolve()
        p_in = Path(name).expanduser()
        if p_in.is_absolute():
            return p_in.resolve()
        norm = name.replace("\\", "/")
        anchor = repo_r if norm == "src" or norm.startswith("src/") else base_r
        return (anchor / name).resolve()

    def _safe_path(self, name: str) -> Path:
        """Resolve path and confine to task workspace or Agent repo root (write/delete/move/execute)."""
        path = self._resolve_path_candidate(name)
        base_r = self._base_dir.resolve()
        repo_r = _REPO_ROOT.resolve()
        if not (path.is_relative_to(base_r) or path.is_relative_to(repo_r)):
            raise ValueError(
                "Path traversal detected: path must be inside the task workspace "
                f"({base_r}) or the Agent project directory ({repo_r})"
            )
        return path

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
            file_path = self._resolve_path_candidate(name)
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
                self._resolve_path_candidate(directory) if directory else self._base_dir
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
            content: Content to write (must be a string)

        CRITICAL LIMITATIONS - READ CAREFULLY:
        - For larger content: Use append_to_file() to write in chunks
        - For code files: Keep under 200 lines per file
        - For long documents: Split into multiple files or write a Python script to generate the file

        DO NOT use this for:
        - Large code files (>200 lines)
        - Long documents or reports
        - Generated content that might be lengthy

        INSTEAD, for large content:
        1. Write a Python script that generates the file using standard file I/O
        2. Use append_to_file() to write content in multiple chunks
        3. Split content into multiple smaller files
        """
        content_len = len(content) if content else 0
        try:
            if content is None:
                return "Write error: content cannot be None"
            if not isinstance(content, str):
                content = str(content)

            self._base_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._safe_path(name)
            os.makedirs(file_path.parent, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            result = f"File '{name}' written successfully ({content_len} characters)"
            return result
        except ValueError as e:
            return f"Security error: {e}"
        except PermissionError as e:
            return f"Permission error: Cannot write to '{name}' - {e}"
        except Exception as e:
            return f"Write error: {type(e).__name__} - {e}"

    def append_to_file(self, name: str, content: str) -> str:
        """
        Append content to an existing file (or create new file if it doesn't exist).
        Use this for writing large content in chunks.

        Parameters:
            name: File name/path (relative to WorkDatabase directory)
            content: Content to append (keep each chunk under 5000 characters)

        Usage Pattern for Large Files:
        1. First chunk: write_file("myfile.txt", "first part...")
        2. Next chunks: append_to_file("myfile.txt", "second part...")
        3. Continue until done
        """
        content_len = len(content) if content else 0
        try:
            if content is None:
                return "Append error: content cannot be None"
            if not isinstance(content, str):
                content = str(content)

            self._base_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._safe_path(name)
            os.makedirs(file_path.parent, exist_ok=True)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)

            total_size = file_path.stat().st_size
            return f"Content appended to '{name}' successfully ({content_len} chars added, total file size: {total_size} bytes)"
        except ValueError as e:
            return f"Security error: {e}"
        except PermissionError as e:
            return f"Permission error: Cannot append to '{name}' - {e}"
        except Exception as e:
            return f"Append error: {type(e).__name__} - {e}"

    def create_directory(self, name: str) -> str:
        """
        Create a directory.
        Parameters:
            name: Directory name/path
        """
        try:
            dir_path = self._safe_path(name)
            os.makedirs(dir_path, exist_ok=True)
            return f"Directory '{name}' created successfully"
        except ValueError as e:
            return f"Security error: {e}"
        except Exception as e:
            return f"Error creating directory: {e}"

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
                except:
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

    def execute_file(self, name: str, args: str = "") -> str:
        """
        Execute a file (supports Python, Shell scripts, etc.).
        Parameters:
            name: File name/path to execute
            args: Optional, command-line arguments to pass to the script
        """
        try:
            file_path = self._safe_path(name)
            if not file_path.exists():
                return f"Error: File '{name}' does not exist"

            ext = file_path.suffix.lower()
            executors = {
                ".py": ["python"],
                ".sh": ["bash"],
                ".bat": ["cmd", "/c"],
                ".ps1": ["powershell", "-File"],
            }

            if ext not in executors:
                return f"Error: Unsupported file type '{ext}'. Supported: {', '.join(executors.keys())}"

            cmd = executors[ext] + [str(file_path)]
            if args:
                cmd.extend(args.split())

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                cwd=str(self._base_dir)
            )
            output = result.stdout + result.stderr
            return_code = result.returncode
            return f"Return code: {return_code}\nOutput:\n{output}" if output else f"Execution completed, return code: {return_code}"
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out (60 seconds)"
        except ValueError as e:
            return f"Security error: {e}"
        except Exception as e:
            return f"Execution error: {e}"

    def run_command(self, command: str, timeout: int = 60) -> str:
        """
        Execute a Shell/terminal command.
        Parameters:
            command: Command to execute
            timeout: Timeout in seconds, defaults to 60
        """
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            return f"Security error: {reason}"

        try:
            use_shell = any(c in command for c in ['|', '>', '<', '&&', '||', ';', '*', '?'])
            cwd = str(self._base_dir.resolve())
            sub_kw: dict = {}
            if re.search(r"\bclawhub\b", command, re.I):
                cwd = str(_REPO_ROOT)
                if "--workdir" not in command:
                    sub_kw["env"] = os.environ | {"CLAWHUB_WORKDIR": cwd}
            sub_kw.update(
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=cwd,
            )

            if use_shell:
                result = subprocess.run(command, shell=True, **sub_kw)
            elif _platform.system() == "Windows":
                result = subprocess.run(command, shell=True, **sub_kw)
            else:
                result = subprocess.run(shlex.split(command), **sub_kw)

            output = result.stdout + result.stderr
            return_code = result.returncode
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
        path = Path(image_path)
        if not path.is_absolute():
            path = (self._base_dir / image_path).resolve()

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

    @property
    def workers_tools(self) -> list:
        """List of tool callables exposed to the Worker Agent."""
        return [
            # Read / list
            self.list_files,
            self.read_file,
            self.read_image,
            # Write
            self.write_file,
            self.append_to_file,
            # Directories
            self.create_directory,
            # Search
            self.search_in_files,
            self.search_web,
            # Browser automation (Playwright / Chromium)
            self.browser_navigate,
            self.browser_get_content,
            self.browser_screenshot,
            self.browser_click,
            self.browser_fill,
            self.browser_press_key,
            self.browser_wait_for_selector,
            self.browser_evaluate,
            self.browser_close,
            # Execute
            self.run_command,
            self.execute_file,
            # Images and user input
            self.ask_user,
            extract_text,
            *self._extra_worker_tools,
            # Agent Skills tools
            *self._skills_toolkit.tools,
        ]