import os
import time
import logging
import requests
import json
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logger = logging.getLogger('NVIDIAImageGenerator')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class NVIDIAImageGenerator:
    """
    Client for NVIDIA Image Generation API with built-in retry logic and error handling.
    
    Features:
    - Automatic retries with exponential backoff
    - API key authentication
    - Configurable timeout
    - Comprehensive error handling
    - Minimal logging (can be disabled for stealth mode)
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30, debug: bool = False):
        """
        Initialize the NVIDIA Image Generator client.
        
        Args:
            api_key: NVIDIA NIM API key. If not provided, will attempt to read from NVIDIA_API_KEY env var
            timeout: Request timeout in seconds
            debug: Enable debug logging
        """
        self.api_key = api_key or os.getenv('NVIDIA_API_KEY')
        if not self.api_key:
            raise ValueError("NVIDIA API key must be provided or set in NVIDIA_API_KEY environment variable")
        
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.timeout = timeout
        self.debug = debug
        
        if debug:
            logger.setLevel(logging.DEBUG)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, 
                                     requests.exceptions.Timeout,
                                     requests.exceptions.HTTPError)),
        before_sleep=lambda retry_state: logger.warning(f"Retrying in {retry_state.next_action.sleep} seconds (attempt {retry_state.attempt_number})")
    )
    def generate_image(self, 
                      prompt: str, 
                      size: str = "1024x1024", 
                      quality: str = "standard", 
                      style: str = "vivid") -> Dict[str, Any]:
        """
        Generate an image using the NVIDIA API.
        
        Args:
            prompt: Text description of the desired image
            size: Image dimensions (width x height). Supported sizes: 256x256, 512x512, 1024x1024, 2048x2048
            quality: Image quality (standard or hd)
            style: Image style (vivid or natural)
        
        Returns:
            Dict containing image URL and metadata
        
        Raises:
            ValueError: If input parameters are invalid
            requests.exceptions.HTTPError: If API returns an error
            requests.exceptions.RequestException: For network-related errors
        """
        # Validate inputs
        if not prompt or len(prompt) > 1000:
            raise ValueError("Prompt must be between 1 and 1000 characters")
        
        if size not in ["256x256", "512x512", "1024x1024", "2048x2048"]:
            raise ValueError(f"Invalid size: {size}. Supported sizes: 256x256, 512x512, 1024x1024, 2048x2048")
        
        if quality not in ["standard", "hd"]:
            raise ValueError(f"Invalid quality: {quality}. Supported values: standard, hd")
        
        if style not in ["vivid", "natural"]:
            raise ValueError(f"Invalid style: {style}. Supported values: vivid, natural")
        
        # Prepare request
        url = f"{self.base_url}/images/generations"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen/qwen3-235b-a22b",
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "style": style
        }
        
        logger.debug(f"Sending request to {url} with payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            # Raise for 4xx and 5xx status codes
            response.raise_for_status()
            
            logger.debug(f"Received response: {json.dumps(response.json(), indent=2)}")
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            if response.status_code == 402:
                logger.error("Payment required - check your NVIDIA NIM subscription status")
            elif response.status_code == 403:
                logger.error("Forbidden - check your API key permissions")
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded - retrying with exponential backoff")
            raise
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error occurred: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Initialize with API key from environment
        image_generator = NVIDIAImageGenerator(debug=True)
        
        # Generate an image
        result = image_generator.generate_image(
            prompt="A futuristic cityscape at sunset with flying cars",
            size="1024x1024",
            quality="hd",
            style="vivid"
        )
        
        print("Image generated successfully!")
        print(f"Image URL: {result['data'][0]['url']}")
        print(f"Model used: {result['model']}")
        print(f"Usage: {result['usage']}")
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        # In production, you would handle specific exceptions differently
        # For example:
        # if isinstance(e, requests.exceptions.HTTPError):
        #     handle_http_error(e)
        # elif isinstance(e, ValueError):
        #     handle_validation_error(e)
        # else:
        #     handle_generic_error(e)

# To use this code:
# 1. Install dependencies:
#    pip install requests tenacity
# 2. Set your NVIDIA API key:
#    export NVIDIA_API_KEY='your-api-key-here'
# 3. Run the script:
#    python nvidia_image_generator.py

# Note: The model name "qwen/qwen3-235b-a22b" is used as specified in the requirements.
# The actual model name in the NVIDIA API might differ slightly - check the official documentation
# for the exact model identifier."""