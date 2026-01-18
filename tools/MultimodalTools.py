import os
import mimetypes
from pydantic_ai import Agent, BinaryContent, ImageUrl, VideoUrl
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import logger
from dotenv import load_dotenv
load_dotenv()

_provider = OpenAIProvider(
    base_url=os.environ.get('BASE_URL'),
    api_key=os.environ.get('API_KEY')
)

MULTIMODAL_MODEL_NAME = os.environ.get('MULTIMODAL_MODEL', 'gpt-5-mini')


def _create_vision_agent():
    model = OpenAIChatModel(
        MULTIMODAL_MODEL_NAME,
        provider=_provider,
    )
    return Agent(model)


def _encode_image_file_to_bytes(image_path: str) -> tuple[bytes, str]:
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/png'
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    return image_bytes, mime_type


def analyze_local_image(image_path: str, prompt: str = "Please describe the content of this image in detail") -> str:
    """
    Analyze local image content.

    Analyzes local image files using a multimodal model, which can be used for:
        - Describing image content
        - Recognizing text in images (OCR)
        - Analyzing objects, scenes, and people in images
        - Answering questions about images

    Parameters:
        image_path: Local image file path (supports both relative and absolute paths)
                    Supported formats: jpg, jpeg, png, gif, webp
        prompt: A question or analysis instruction for the image, for example:
                - "Please describe the image content"
                - "What text is in the image?"
                - "What scene is this?"
                - "How many people are in the image?"

    Returns:
        str: Image analysis result

    Example:
        >>> analyze_local_image("./screenshot.png", "Please recognize the text in the image")
        >>> analyze_local_image("C:/images/photo.jpg", "Describe the content of this photo")
    """
    logger.debug(f"(analyze_local_image), image_path={image_path}, prompt={prompt}")
    try:
        image_bytes, media_type = _encode_image_file_to_bytes(image_path)
        
        vision_agent = _create_vision_agent()
        result = vision_agent.run_sync([
            prompt,
            BinaryContent(data=image_bytes, media_type=media_type),
        ])
        
        return f"Image analysis result:\n{result.output}"
    
    except FileNotFoundError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Image analysis failed: {str(e)}"


def analyze_image_url(image_url: str, prompt: str = "Please describe the content of this image in detail") -> str:
    """
    Analyze web image content.

    Analyzes online images via URL, which can be used for:
        - Describing image content
        - Recognizing text in images (OCR)
        - Analyzing objects, scenes, and people in images
        - Answering questions about images

    Parameters:
        image_url: URL address of the image, must be a publicly accessible link
            Supported formats: jpg, jpeg, png, gif, webp
        prompt: A question or analysis instruction for the image, for example:
            - "Please describe the image content"
            - "What text is in the image?"
            - "What scene is this?"
            - "How many people are in the image?"

    Returns:
        str: Image analysis result

    Example:
        >>> analyze_image_url("https://example.com/image.jpg", "Please describe this image")
        >>> analyze_image_url("https://example.com/chart.png", "Analyze what data this chart shows")
    """
    logger.debug(f"(analyze_image_url), image_path={image_url}, prompt={prompt}")
    try:
        if not image_url.startswith(('http://', 'https://')):
            return "Error: Image URL must start with http:// or https://"
        
        vision_agent = _create_vision_agent()
        result = vision_agent.run_sync([
            prompt,
            ImageUrl(url=image_url),
        ])
        
        return f"Image analysis result:\n{result.output}"
    
    except Exception as e:
        return f"Image analysis failed: {str(e)}"


def analyze_videos_url(video_url: str, prompt: str = "Please describe the content of this Video in detail") -> str:
    """
    Analyze web Video content.

    Analyzes online Video via URL, which can be used for:
        - Describing Video content
        - Analyzing objects, scenes, and people in images
        - Answering questions about Video

    Parameters:
        Video_url: URL address of the Video, must be a publicly accessible link
        prompt: A question or analysis instruction for the image, for example:
            - "Please describe the Video content"
            - "What text is in the Video?"
            - "How many people are in the Video?"

    Returns:
        str: Video analysis result
    """
    logger.debug(f"(analyze_videos_url), image_path={video_url}, prompt={prompt}")
    try:
        if not video_url.startswith(('http://', 'https://')):
            return "Error: Video URL must start with http:// or https://"

        vision_agent = _create_vision_agent()
        result = vision_agent.run_sync([
            prompt,
            VideoUrl(url=video_url),
        ])

        return f"Video analysis result:\n{result.output}"

    except Exception as e:
        return f"Video analysis failed: {str(e)}"


def analyze_multiple_images(image_sources: list, prompt: str = "Please analyze these images") -> str:
    """
    Analyze multiple images.

    Simultaneously analyzes multiple images (local or URL), which can be used for:
        - Comparing differences between multiple images
        - Analyzing image sequences
        - Comprehensively understanding content across multiple images

    Parameters:
        image_sources: List of image sources, where each element can be:
                           - {"type": "local", "path": "image path"}
                           - {"type": "url", "url": "image URL"}
        prompt: Analysis requirements for the images

    Returns:
            str: Image analysis result

    Example:
        >>> analyze_multiple_images([
        ...     {"type": "local", "path": "./img1.png"},
        ...     {"type": "url", "url": "https://example.com/img2.jpg"}
        ... ], "Compare the differences between these two images")
    """
    try:
        messages = [prompt]
        
        for i, source in enumerate(image_sources):
            if source.get("type") == "local":
                path = source.get("path")
                if path:
                    image_bytes, media_type = _encode_image_file_to_bytes(path)
                    messages.append(BinaryContent(data=image_bytes, media_type=media_type))
            elif source.get("type") == "url":
                url = source.get("url")
                if url and url.startswith(('http://', 'https://')):
                    messages.append(ImageUrl(url=url))
        
        if len(messages) <= 1:
            return "Error: No valid image sources provided"
        
        vision_agent = _create_vision_agent()
        result = vision_agent.run_sync(messages)
        return f"Multi-image analysis result:\n{result.output}"
    
    except Exception as e:
        return f"Image analysis failed: {str(e)}"
