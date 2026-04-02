from pathlib import Path
import os
import subprocess
import time
import mimetypes
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import logger
import shlex
import platform as _platform
from pydantic_ai import BinaryContent, ImageUrl, ToolReturn
from tools.ExtractFileContent import extract_text
from skills.SkillsTools import SkillsToolkit


class BasicToolkit:
    _WORK_DATABASE_ROOT = Path("./WorkDatabase")

    def __init__(self):
        self._base_dir: Path = self._WORK_DATABASE_ROOT
        self._ask_user_handler = None
        self._skills_toolkit = SkillsToolkit()
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
    def base_dir(self) -> Path:
        return self._base_dir

    def set_task_directory(self, task_name: str) -> Path:
        """
        为当前任务设置独立的工作目录

        Parameters:
            task_name: 任务名称，将用于创建子文件夹

        Returns:
            Path: 任务工作目录的路径
        """
        safe_name = "".join(
            c if c.isalnum() or c in ('_', '-', ' ') else '_' for c in task_name
        )
        safe_name = safe_name.strip()[:50]

        if not safe_name:
            safe_name = "default_task"

        task_dir = self._WORK_DATABASE_ROOT / safe_name
        task_dir.mkdir(parents=True, exist_ok=True)

        self._base_dir = task_dir
        logger.info(f"📁 任务工作目录已设置: {task_dir}")

        return task_dir

    def reset_task_directory(self):
        self._base_dir = self._WORK_DATABASE_ROOT
        logger.info(f"📁 工作目录已重置为: {self._base_dir}")

    def _safe_path(self, name: str) -> Path:
        """Ensure path is within base_dir to prevent path traversal attacks"""
        path = (self._base_dir / name).resolve()
        path_str = str(path)
        base_str = str(self._base_dir.resolve())
        if _platform.system() == "Windows":
            path_str = path_str.lower()
            base_str = base_str.lower()
        if not path_str.startswith(base_str):
            raise ValueError("Path traversal detected: access outside base_dir is not allowed")
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
        return True, ""

    def set_ask_user_handler(self, handler):
        """
        替换 ask_user 的底层实现（用于非终端场景，如 QQ Bot、Web API 等）。
        handler 签名：(question: str) -> str
        传入 None 可恢复默认终端输入。
        """
        self._ask_user_handler = handler

    def ask_user(self, question: str) -> str:
        """
        主动询问用户问题并获取回答
        Parameters:
            question: 要询问用户的问题
        Returns:
            用户的回答
        """
        if self._ask_user_handler is not None:
            result = self._ask_user_handler(question)
            return result if result is not None else "(用户未回复)"

        logger.info("=" * 50)
        logger.info("🤔 Agent 需要您的帮助")
        logger.info("=" * 50)
        logger.info(f"问题: {question}")

        user_response = input("📝 您的回复: ").strip()
        logger.info(f"用户回答: {user_response}")

        return user_response

    def read_file(self, name: str) -> str:
        """
        Read file contents.
        Parameters:
            name: File name/path
        """
        logger.debug(f"(read_file {name})")
        try:
            file_path = self._safe_path(name)
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return content if content else "File is empty"
        except ValueError as e:
            return f"Security error: {e}"
        except Exception as e:
            return f"Read error: {e}"

    def list_files(self, directory: str = "") -> str:
        """
        List all files and folders in a directory.
        Parameters:
            directory: Optional, subdirectory path, defaults to root directory
        """
        logger.debug(f"(list_files {directory})")
        try:
            target_dir = self._safe_path(directory) if directory else self._base_dir
            if not target_dir.exists():
                return f"Error: Directory '{directory}' does not exist"

            items = []
            for item in sorted(target_dir.iterdir()):
                rel_path = str(item.relative_to(self._base_dir))
                if item.is_dir():
                    items.append(f"{rel_path}/")
                else:
                    size = item.stat().st_size
                    items.append(f"{rel_path} ({size} bytes)")

            return "\n".join(items) if items else "Directory is empty"
        except ValueError as e:
            return f"Security error: {e}"
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
        logger.debug(f"(write_file {name}, content_length={content_len})")
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
        logger.debug(f"(append_to_file {name}, content_length={content_len})")
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
        logger.debug(f"(create_directory {name})")
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
        logger.debug(f"(search_in_files keyword='{keyword}', ext={file_extension})")
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
                                results.append(f"{rel_path}:{line_num}: {line.strip()[:100]}")
                except:
                    continue

            if results:
                output = f"Found {len(results)} matches:\n" + "\n".join(results[:50])
                if len(results) > 50:
                    output += f"\n... {len(results) - 50} more matches not shown"
                return output
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
        logger.debug(f"(search_web query='{query}', max_results={max_results})")
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

    def fetch_webpage(self, url: str, extract_text_only: bool = True) -> str:
        """
        Fetch webpage content. Can return plain text or HTML content.
        Parameters:
            url: The URL of the webpage to fetch
            extract_text_only: If True, returns the extracted plain text; if False, returns the raw HTML
        """
        logger.debug(f"(fetch_webpage url='{url}', extract_text={extract_text_only})")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            if extract_text_only:
                soup = BeautifulSoup(response.text, 'html.parser')

                for script in soup(['script', 'style', 'meta', 'link']):
                    script.decompose()

                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)

                return f"Page Title: {soup.title.string if soup.title else 'No title'}\n\nContent:\n{text[:10000]}{'...' if len(text) > 10000 else ''}"
            else:
                return response.text[:10000] + ('...' if len(response.text) > 10000 else '')

        except requests.exceptions.RequestException as e:
            return f"Error fetching webpage: {e}"
        except Exception as e:
            return f"Error processing webpage content: {e}"

    def execute_file(self, name: str, args: str = "") -> str:
        """
        Execute a file (supports Python, Shell scripts, etc.).
        Parameters:
            name: File name/path to execute
            args: Optional, command-line arguments to pass to the script
        """
        logger.debug(f"(execute_file {name} {args})")
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
        logger.debug(f"(run_command: {command})")
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            return f"Security error: {reason}"

        try:
            use_shell = any(c in command for c in ['|', '>', '<', '&&', '||', ';', '*', '?'])

            if use_shell:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    cwd=str(self._base_dir)
                )
            else:
                if _platform.system() == "Windows":
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout,
                        cwd=str(self._base_dir)
                    )
                else:
                    cmd_parts = shlex.split(command)
                    result = subprocess.run(
                        cmd_parts,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout,
                        cwd=str(self._base_dir)
                    )

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
        logger.debug(f"(read_image {image_path})")

        is_url = image_path.startswith(('http://', 'https://'))

        try:
            if is_url:
                return self._read_image_from_url(image_path)
            else:
                return self._read_image_from_file(image_path)
        except Exception as e:
            return f"Error reading image: {type(e).__name__} - {e}"

    def _read_image_from_file(self, image_path: str) -> ToolReturn | str:
        """从本地文件路径读取图片"""
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

    @staticmethod
    def _read_image_from_url(url: str) -> ToolReturn:
        """直接将图片 URL 传递给模型进行分析，无需下载"""
        return ToolReturn(
            return_value=f"Image URL passed to model: {url}",
            content=[
                f"Image from URL: {url}",
                ImageUrl(url=url),
            ]
        )

    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, max_wait_time: int = 300) -> str:
        """
        Generate images using AI model. Use this tool whenever the user asks to create, generate, make, or produce an image, picture, photo, illustration, artwork, or visual content.

        This is the PRIMARY tool for ALL image generation requests. Keywords that should trigger this tool:
        - "生成图片" / "生成图像" / "create image" / "generate image" / "make a picture"
        - "画一张" / "draw" / "paint" / "illustrate"
        - "给我一张" / "给我一个图片" / "give me an image"
        - "创建图片" / "制作图片" / "produce image"
        - Any request involving creating visual content, artwork, diagrams, or images

        Parameters:
            prompt: The text description of what image to generate. Be detailed and specific about the visual content, style, composition, colors, mood, etc. This is the most important parameter.
            width: Image width in pixels. Default: 1024. Common values: 512, 768, 1024, 1536, etc.
            height: Image height in pixels. Default: 1024. Common values: 512, 768, 1024, 1536, etc.
            max_wait_time: Maximum wait time in seconds. Default: 300 (5 minutes).

        Returns:
            Success: Returns the image URL and generation details.
            Failure: Returns an error message.
        """
        logger.debug(f"(generate_image prompt='{prompt}', width={width}, height={height})")

        bfl_base_url = os.environ.get('BFL_BASE_URL')
        bfl_api_key = os.environ.get("BFL_API_KEY")
        if not bfl_api_key:
            return "Error: BFL_API_KEY environment variable is not set. Please set it before using image generation."

        try:
            logger.info(f"正在提交图像生成请求: {prompt[:50]}...")
            response = requests.post(
                bfl_base_url,
                headers={
                    "accept": "application/json",
                    "x-key": bfl_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                },
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()

            request_id = response_data.get("id")
            polling_url = response_data.get("polling_url")

            if not polling_url:
                return f"Error: No polling_url received from API. Response: {response_data}"

            logger.info(f"请求已提交，Request ID: {request_id}")
            logger.info("正在等待图像生成完成...")

            start_time = time.time()
            poll_count = 0

            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    return f"Error: Image generation timed out after {max_wait_time} seconds. Request ID: {request_id}"

                poll_count += 1
                if poll_count % 10 == 0:
                    logger.info(f"仍在等待中... (已等待 {elapsed_time:.1f} 秒)")

                result_response = requests.get(
                    polling_url,
                    headers={
                        "accept": "application/json",
                        "x-key": bfl_api_key
                    },
                    timeout=30
                )
                result_response.raise_for_status()
                result = result_response.json()

                status = result.get("status", "Unknown")

                if status == "Ready":
                    image_url = result.get("result", {}).get("sample")
                    if image_url:
                        logger.info("图像生成成功！")
                        logger.info(f"图像URL: {image_url}")
                        return f"Image generated successfully!\nImage URL: {image_url}\nPrompt: {prompt}\nDimensions: {width}x{height}"
                    else:
                        return f"Error: Image generation completed but no image URL found in response. Response: {result}"

                elif status == "Failed":
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"图像生成失败: {error_msg}")
                    return f"Error: Image generation failed - {error_msg}"

                time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            logger.error(f"API请求错误: {e}")
            return f"Error: API request failed - {e}"
        except Exception as e:
            logger.error(f"图像生成异常: {e}")
            return f"Error: Image generation exception - {type(e).__name__}: {e}"


    @property
    def workers_tools(self) -> list:
        """返回所有可供 Worker Agent 使用的工具函数列表"""
        return [
            # 查
            self.list_files,
            self.read_file,
            self.read_image,
            # 增
            self.write_file,
            self.append_to_file,
            # 目录操作
            self.create_directory,
            # 搜索操作
            self.search_in_files,
            self.search_web,
            # 网络操作
            self.fetch_webpage,
            # 执行操作
            self.run_command,
            self.execute_file,
            # 图像生成与识别
            self.generate_image,
            self.ask_user,
            extract_text,
            # Agent Skills 工具
            *self._skills_toolkit.tools,
        ]