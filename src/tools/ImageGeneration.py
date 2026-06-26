from infra import logger
import asyncio
import requests
import time

from config.app_config import get_env


async def generate_image_from_flux(prompt: str, width: int = 1024, height: int = 1024, max_wait_time: int = 300):
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
        Success: Returns a tuple (image_bytes, mime_type, info_text).
        Failure: Returns an error message string.
    """
    bfl_base_url = get_env("BFL_BASE_URL", warn=False)
    bfl_api_key = get_env("BFL_API_KEY", warn=False)
    if not bfl_api_key:
        return "Error: BFL_API_KEY environment variable is not set. Please set it before using image generation."

    try:
        logger.info(f"正在提交图像生成请求: {prompt[:50]}...")
        response = await asyncio.to_thread(
            requests.post,
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

            result_response = await asyncio.to_thread(
                requests.get,
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
                    img_response = await asyncio.to_thread(requests.get, image_url, timeout=30)
                    img_response.raise_for_status()
                    image_bytes = img_response.content
                    mime_type = img_response.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
                    info_text = f"Image generated successfully!\nImage URL: {image_url}\nPrompt: {prompt}\nDimensions: {width}x{height}"
                    return image_bytes, mime_type, info_text
                else:
                    return f"Error: Image generation completed but no image URL found in response. Response: {result}"

            elif status == "Failed":
                error_msg = result.get("error", "Unknown error")
                logger.error(f"图像生成失败: {error_msg}")
                return f"Error: Image generation failed - {error_msg}"

            await asyncio.sleep(0.5)

    except requests.exceptions.RequestException as e:
        logger.error(f"API请求错误: {e}")
        return f"Error: API request failed - {e}"
    except Exception as e:
        logger.error(f"图像生成异常: {e}")
        return f"Error: Image generation exception - {type(e).__name__}: {e}"
