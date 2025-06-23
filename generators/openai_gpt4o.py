"""
OpenAI GPT-4o integration for AI art generation.
Based on the OpenAI API client with gpt-image-1 model.
"""

import os
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image, PngImagePlugin
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .base_generator import BaseGenerator

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Add the parent directory to the path for imports
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from prompts.prompt_parser import ArtPrompt
from config import get_config


def get_max_workers(service: str, num_prompts: int) -> int:
    """Determine optimal thread count based on API limits."""
    limits = {
        'gpt4o': 5,      # Tier 1: 5 images per minute
        'genai': 8,    # Within 50 RPM default quota
    }
    return min(num_prompts, limits.get(service, 5))


class OpenAIGPT4oGenerator(BaseGenerator):
    """OpenAI GPT-4o image generator using gpt-image-1 model."""
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the generator with API credentials."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize the client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Load configuration and initialize base class
        config = get_config(config_path)
        super().__init__(config)
        
        # Model configuration
        self.model_name = self.config.get('openai.model', 'gpt-image-1')
    
    def generate_image(self, prompt: ArtPrompt, output_dir: str, num_images: int = 1, 
                      size: Optional[str] = None, quality: Optional[str] = None,
                      style: Optional[str] = None, response_format: str = "url") -> List[str]:
        """
        Generate images using OpenAI GPT-4o.
        
        Args:
            prompt: ArtPrompt object containing prompt text and metadata
            output_dir: Directory where images should be saved
            num_images: Number of images to generate
            size: Override image size (e.g., "1024x1024", "1024x1536", "1536x1024")
            quality: Override quality setting ("low", "medium", "high") - gpt-image-1 model
            style: Override style ("vivid", "natural")
            response_format: Response format ("url" or "b64_json")
            
        Returns:
            List of paths to generated images
        """
        print(f"Generating {num_images} image(s) for: {prompt.filename}")
        print(f"Prompt: {prompt.prompt[:100]}...")
        
        try:
            # Get configuration for this prompt's category
            openai_config = self.config.get_openai_config(prompt.category)
            
            # Build API parameters with overrides
            api_params = {
                'model': self.model_name,
                'prompt': prompt.prompt,
                'n': num_images
            }
            
            # Add size
            final_size = size or openai_config.get('size', '1024x1024')
            api_params['size'] = final_size
            
            # Add quality (gpt-image-1 supports: low, medium, high)
            final_quality = quality or openai_config.get('quality', 'medium')
            if final_quality in ['low', 'medium', 'high']:
                api_params['quality'] = final_quality
            
            # Add style (only if supported)
            final_style = style or openai_config.get('style', 'natural')
            # Note: Style parameter may not be supported by gpt-image-1 model
            # if final_style != 'natural':
            #     api_params['style'] = final_style
            
            print(f"  Using size: {final_size}")
            if final_quality in ['low', 'medium', 'high']:
                print(f"  Using quality: {final_quality}")
            # Note: Style parameter may not be supported yet by gpt-image-1
            # if final_style != 'natural':
            #     print(f"  Using style: {final_style}")
            
            # Make the API call
            response = self.client.images.generate(**api_params)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save generated images
            saved_paths = []
            base_filename = Path(prompt.filename).stem  # Remove extension
            
            for idx, image_data in enumerate(response.data):
                # Handle OpenAI response format (b64_json by default for gpt-image-1)
                if image_data.b64_json:
                    # Decode base64 image data
                    image_bytes = base64.b64decode(image_data.b64_json)
                    image = Image.open(BytesIO(image_bytes))
                    print(f"  Loaded image from base64 data")
                elif image_data.url:
                    # Download image from URL
                    import requests
                    img_response = requests.get(image_data.url)
                    img_response.raise_for_status()
                    image = Image.open(BytesIO(img_response.content))
                    print(f"  Downloaded image from URL")
                else:
                    raise ValueError("No image data (URL or base64) returned from API")
                
                # Create filename
                if num_images > 1:
                    filename = f"{base_filename}_{idx+1:03d}.png"
                else:
                    filename = f"{base_filename}.png"
                
                # Save path
                save_path = os.path.join(output_dir, filename)
                
                # Save image with prompt metadata
                self._save_image_with_metadata(image, save_path, prompt.prompt, prompt.description, final_size, final_quality, final_style)
                saved_paths.append(save_path)
                
                print(f"  ✓ Saved: {save_path}")
            
            return saved_paths
            
        except Exception as e:
            print(f"  ✗ Error generating image: {e}")
            raise
    
    def _save_image_with_metadata(self, image: Image.Image, save_path: str, prompt: str, description: str, 
                                 size: str, quality: str, style: str):
        """Save image with prompt metadata embedded in PNG."""
        # Create metadata using base class helper
        metadata_fields = self._create_standard_metadata(
            prompt=prompt,
            description=description,
            generator_name="OpenAI GPT-4o",
            model_name=self.model_name,
            size=size,
            quality=quality,
            style=style
        )
        
        # Use base class save with backup functionality
        self._save_image_with_backup(image, save_path, metadata_fields)
    
    def generate_batch(self, prompts: List[ArtPrompt], base_output_dir: str, 
                      images_per_prompt: int = 1, size: Optional[str] = None,
                      quality: Optional[str] = None, style: Optional[str] = None) -> dict:
        """
        Generate images for multiple prompts using threading.
        
        Args:
            prompts: List of ArtPrompt objects
            base_output_dir: Base directory for output
            images_per_prompt: Number of images to generate per prompt
            size: Override size for all prompts
            quality: Override quality for all prompts
            style: Override style for all prompts
            
        Returns:
            Dictionary mapping prompt filenames to generated image paths
        """
        results = {}
        
        def generate_single_prompt(prompt: ArtPrompt) -> Tuple[str, List[str]]:
            """Generate images for a single prompt."""
            try:
                # Determine output directory
                if prompt.category == "background":
                    output_dir = os.path.join(base_output_dir, "backgrounds")
                else:
                    output_dir = os.path.join(base_output_dir, "characters")
                
                # Generate images
                generated_paths = self.generate_image(
                    prompt, output_dir, images_per_prompt,
                    size=size, quality=quality, style=style
                )
                return prompt.filename, generated_paths
                
            except Exception as e:
                print(f"✗ Failed: {prompt.filename}: {e}")
                return prompt.filename, []
        
        # Execute with threading
        max_workers = get_max_workers('gpt4o', len(prompts))
        print(f"Using {max_workers} concurrent threads for OpenAI generation (Tier 1: 5 IPM limit)")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_prompt = {
                executor.submit(generate_single_prompt, prompt): prompt 
                for prompt in prompts
            }
            
            # Collect results as they complete
            completed = 0
            total = len(prompts)
            
            for future in as_completed(future_to_prompt):
                filename, paths = future.result()
                results[filename] = paths
                completed += 1
                
                if paths:
                    print(f"  ✓ Completed ({completed}/{total}): {filename}")
                else:
                    print(f"  ✗ Failed ({completed}/{total}): {filename}")
        
        return results
    
    @staticmethod
    def test_connection(api_key: Optional[str] = None) -> bool:
        """Test if the API connection works."""
        try:
            generator = OpenAIGPT4oGenerator(api_key)
            # Test with a simple prompt - use minimal parameters for compatibility
            response = generator.client.images.generate(
                model="gpt-image-1",
                prompt="A simple test image of a red circle",
                size="1024x1024",
                n=1
            )
            if response and response.data:
                print("✓ OpenAI GPT-4o connection successful")
                return True
            else:
                print("✗ OpenAI GPT-4o connection failed: No response data")
                return False
        except Exception as e:
            print(f"✗ OpenAI GPT-4o connection failed: {e}")
            return False


def main():
    """Test the OpenAI GPT-4o generator."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python openai_gpt4o.py <test_prompt>")
        sys.exit(1)
    
    test_prompt_text = sys.argv[1]
    
    # Create a test prompt
    from prompts.prompt_parser import ArtPrompt
    test_prompt = ArtPrompt(
        filename="test_image.png",
        description="Test image generation",
        prompt=test_prompt_text,
        category="background",
        output_path="test_output/test_image.png"
    )
    
    try:
        generator = OpenAIGPT4oGenerator()
        output_dir = "test_output"
        
        paths = generator.generate_image(test_prompt, output_dir)
        print(f"Generated {len(paths)} image(s):")
        for path in paths:
            print(f"  {path}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()