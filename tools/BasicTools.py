from pathlib import Path
import os
import subprocess
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from tools import MultimodalTools
import logger
import shlex
import platform as _platform
from tools.ExtraceFileContent import extract_text
from skills.SkillsTools import skills_tools

_DANGEROUS_PATTERNS = [
    'rm -rf /',
    'rm -rf /*',
    'mkfs.',
    'dd if=',
    ':(){:|:&};:',
    '> /dev/sda',
    'chmod -R 777 /',
    'chown -R',
    '| sh',
    '| bash',
    '`',
    '$(',
    'eval ',
    'exec ',
]

base_dir = Path("./WorkDatabase")


def _safe_path(name: str) -> Path:
    """Ensure path is within base_dir to prevent path traversal attacks"""
    path = (base_dir / name).resolve()
    if not str(path).startswith(str(base_dir.resolve())):
        raise ValueError("Path traversal detected: access outside base_dir is not allowed")
    return path


def _is_command_safe(command: str) -> tuple[bool, str]:
    """Check if command contains dangerous patterns"""
    command_lower = command.lower().strip()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.lower() in command_lower:
            return False, f"Dangerous command pattern detected: '{pattern}'"
    return True, ""


def read_file(name: str, max_lines: int = None) -> str:
    """
    Read file contents.
    Parameters:
        name: File name/path
        max_lines: Optional, maximum number of lines to read (prevents context overflow for large files)
    """
    logger.debug(f"(read_file {name}, max_lines={max_lines})")
    try:
        file_path = _safe_path(name)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if max_lines:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... File truncated, read {max_lines} lines ...")
                        break
                    lines.append(line)
                content = "".join(lines)
            else:
                content = f.read()
        return content if content else "File is empty"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Read error: {e}"


def list_files(directory: str = "") -> str:
    """
    List all files and folders in a directory.
    Parameters:
        directory: Optional, subdirectory path, defaults to root directory
    """
    logger.debug(f"(list_files {directory})")
    try:
        target_dir = _safe_path(directory) if directory else base_dir
        if not target_dir.exists():
            return f"Error: Directory '{directory}' does not exist"
        
        items = []
        for item in sorted(target_dir.iterdir()):
            rel_path = str(item.relative_to(base_dir))
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


def delete_file(name: str) -> str:
    """
    Delete a file.
    Parameters:
        name: File name/path to delete
    """
    logger.debug(f"(delete_file {name})")
    try:
        file_path = _safe_path(name)
        if not file_path.exists():
            return f"File '{name}' does not exist"
        os.remove(file_path)
        return f"File '{name}' has been deleted"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Delete error: {e}"


def write_file(name: str, content: str) -> str:
    """
    Create or overwrite a file with SHORT content only.
    
    Parameters:
        name: File name/path (relative to WorkDatabase directory)
        content: Content to write (must be a string)
    
    ⚠CRITICAL LIMITATIONS - READ CAREFULLY:
    - Maximum content length: 6000 characters (STRICTLY ENFORCED)
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

        if content_len > 6000:
            logger.warning(f"write_file: content length ({content_len}) exceeds recommended limit of 6000 characters")

        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = _safe_path(name)
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


def append_to_file(name: str, content: str) -> str:
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

        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = _safe_path(name)
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


def execute_file(name: str, args: str = "") -> str:
    """
    Execute a file (supports Python, Shell scripts, etc.).
    Parameters:
        name: File name/path to execute
        args: Optional, command-line arguments to pass to the script
    """
    logger.debug(f"(execute_file {name} {args})")
    try:
        file_path = _safe_path(name)
        if not file_path.exists():
            return f"Error: File '{name}' does not exist"

        ext = file_path.suffix.lower()
        executors = {
            ".py": ["python"],
            ".sh": ["bash"],
            ".bat": ["cmd", "/c"],
            ".ps1": ["powershell", "-File"],
            # ".js": ["node"],
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
            cwd=str(base_dir)
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


def search_web(query: str, max_results: int = 5) -> str:
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


def fetch_webpage(url: str, extract_text: bool = True) -> str:
    """
    Fetch webpage content. Can return plain text or HTML content.
    Parameters:
        url: The URL of the webpage to fetch
        extract_text: If True, returns the extracted plain text; if False, returns the raw HTML
    """
    logger.debug(f"(fetch_webpage url='{url}', extract_text={extract_text})")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        
        if extract_text:
            soup = BeautifulSoup(response.text, 'html.parser')

            for script in soup(['script', 'style', 'meta', 'link']):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return f"Page Title: {soup.title.string if soup.title else 'No title'}\n\nContent:\n{text[:5000]}{'...' if len(text) > 5000 else ''}"
        else:
            return response.text[:10000] + ('...' if len(response.text) > 10000 else '')
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching webpage: {e}"
    except Exception as e:
        return f"Error processing webpage content: {e}"


def run_command(command: str, timeout: int = 60) -> str:
    """
    Execute a Shell/terminal command.
    Parameters:
        command: Command to execute
        timeout: Timeout in seconds, defaults to 60
    """
    logger.debug(f"(run_command: {command})")
    is_safe, reason = _is_command_safe(command)
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
                cwd=str(base_dir)
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
                    cwd=str(base_dir)
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
                    cwd=str(base_dir)
                )
        
        output = result.stdout + result.stderr
        return_code = result.returncode
        return f"Return code: {return_code}\nOutput:\n{output}" if output else f"Execution completed, return code: {return_code}"
    except subprocess.TimeoutExpired:
        return f"Error: Command execution timed out ({timeout} seconds)"
    except Exception as e:
        return f"Execution error: {e}"


def search_in_files(keyword: str, file_extension: str = None) -> str:
    """
    Search for a keyword in files.
    Parameters:
        keyword: Keyword to search for
        file_extension: Optional, limit search to specific file types, e.g., ".py", ".txt"
    """
    logger.debug(f"(search_in_files keyword='{keyword}', ext={file_extension})")
    results = []
    try:
        for file_path in base_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_extension and file_path.suffix != file_extension:
                continue
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if keyword.lower() in line.lower():
                            rel_path = file_path.relative_to(base_dir)
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


def create_directory(name: str) -> str:
    """
    Create a directory.
    Parameters:
        name: Directory name/path
    """
    logger.debug(f"(create_directory {name})")
    try:
        dir_path = _safe_path(name)
        os.makedirs(dir_path, exist_ok=True)
        return f"Directory '{name}' created successfully"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error creating directory: {e}"


def delete_directory(name: str, force: bool = False) -> str:
    """
    Delete a directory.
    Parameters:
        name: Directory name/path
        force: Whether to force delete non-empty directories
    """
    logger.debug(f"(delete_directory {name}, force={force})")
    try:
        import shutil
        dir_path = _safe_path(name)
        if not dir_path.exists():
            return f"Error: Directory '{name}' does not exist"
        if not dir_path.is_dir():
            return f"Error: '{name}' is not a directory"
        
        if force:
            shutil.rmtree(dir_path)
        else:
            os.rmdir(dir_path)  # Can only delete empty directories
        return f"Directory '{name}' has been deleted"
    except OSError as e:
        if "not empty" in str(e).lower() or "目录不是空的" in str(e):
            return f"Error: Directory is not empty, set force=True to force delete"
        return f"Delete error: {e}"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Delete error: {e}"


def ask_user(question: str) -> str:
    """
    主动询问用户问题并获取回答
    Parameters:
        question: 要询问用户的问题
    Returns:
        用户的回答
    """
    logger.info("=" * 50)
    logger.info("🤔 Agent 需要您的帮助")
    logger.info("=" * 50)
    logger.info(f"问题: {question}")

    user_response = input("📝 您的回复: ").strip()
    logger.info(f"用户回答: {user_response}")

    return user_response


workers_tools = [
    # 查
    list_files,
    read_file,
    # 增
    write_file,
    append_to_file,
    # 删
    delete_file,
    # 目录操作
    create_directory,
    delete_directory,
    # 搜索操作
    search_in_files,
    search_web,
    # 网络操作
    fetch_webpage,
    # 执行操作
    run_command,
    execute_file,
    # 多模态图像理解
    MultimodalTools.analyze_local_image,
    MultimodalTools.analyze_image_url,
    MultimodalTools.analyze_multiple_images,
    MultimodalTools.analyze_videos_url,
    ask_user,
    extract_text,
    # Agent Skills 工具
    *skills_tools,
]

workers_parameter = {
    "temperature": 0.6,
    "top_p": 0.8,
    "max_tokens": 65536,
}
