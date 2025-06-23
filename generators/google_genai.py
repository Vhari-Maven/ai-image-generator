"""
Google GenAI integration for AI art generation.
Based on the Google GenAI client API.
"""

import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image, PngImagePlugin
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .base_generator import BaseGenerator

try:
    from google import genai
    from google.genai import types
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

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


class GoogleGenAIGenerator(BaseGenerator):
    """Google GenAI image generator."""
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the generator with API credentials."""
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("Google GenAI libraries not installed. Run: pip install google-genai")
        
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            raise ValueError("Google AI API key not provided. Set GOOGLE_AI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize the client
        self.client = genai.Client(api_key=self.api_key)
        
        # Load configuration and initialize base class
        config = get_config(config_path)
        super().__init__(config)
        
        # Model configuration
        self.model_name = self.config.get('genai.model', 'imagen-3.0-generate-002')
    
    def generate_image(self, prompt: ArtPrompt, output_dir: str, num_images: int = 1, 
                      aspect_ratio: Optional[str] = None, safety_filter_level: Optional[str] = None,
                      person_generation: Optional[str] = None) -> List[str]:
        """
        Generate images using Google GenAI.
        
        Args:
            prompt: ArtPrompt object containing prompt text and metadata
            output_dir: Directory where images should be saved
            num_images: Number of images to generate
            aspect_ratio: Override aspect ratio (e.g., "16:9", "3:4")
            safety_filter_level: Override safety filter level
            person_generation: Override person generation setting
            
        Returns:
            List of paths to generated images
        """
        print(f"Generating {num_images} image(s) for: {prompt.filename}")
        print(f"Prompt: {prompt.prompt[:100]}...")
        
        try:
            # Get configuration for this prompt's category
            genai_config = self.config.get_genai_config(prompt.category)
            
            # Build API config with overrides
            config_params = {'number_of_images': num_images}
            
            # Add aspect ratio
            final_aspect_ratio = aspect_ratio or genai_config.get('aspect_ratio', '1:1')
            config_params['aspect_ratio'] = final_aspect_ratio
            
            # Add safety filter level
            final_safety_level = safety_filter_level or genai_config.get('safety_filter_level')
            if final_safety_level:
                config_params['safety_filter_level'] = final_safety_level
            
            # Add person generation setting
            final_person_gen = person_generation or genai_config.get('person_generation')
            if final_person_gen:
                config_params['person_generation'] = final_person_gen
            
            print(f"  Using aspect ratio: {final_aspect_ratio}")
            if final_safety_level:
                print(f"  Using safety filter: {final_safety_level}")
            if final_person_gen:
                print(f"  Person generation: {final_person_gen}")
            
            # Create the request config
            config = types.GenerateImagesConfig(**config_params)
            
            # Make the API call
            image_response = self.client.models.generate_images(
                model=self.model_name,
                prompt=prompt.prompt,
                config=config
            )
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save generated images
            saved_paths = []
            base_filename = Path(prompt.filename).stem  # Remove extension
            
            for idx, generated_image in enumerate(image_response.generated_images):
                # Convert bytes to PIL Image
                image = Image.open(BytesIO(generated_image.image.image_bytes))
                
                # Create filename
                if num_images > 1:
                    filename = f"{base_filename}_{idx+1:03d}.png"
                else:
                    filename = f"{base_filename}.png"
                
                # Save path
                save_path = os.path.join(output_dir, filename)
                
                # Save image with prompt metadata
                self._save_image_with_metadata(image, save_path, prompt.prompt, prompt.description)
                saved_paths.append(save_path)
                
                print(f"  ✓ Saved: {save_path}")
            
            return saved_paths
            
        except Exception as e:
            print(f"  ✗ Error generating image: {e}")
            raise
    
    def _save_image_with_metadata(self, image: Image.Image, save_path: str, prompt: str, description: str):
        """Save image with prompt metadata embedded in PNG."""
        # Create metadata using base class helper
        metadata_fields = self._create_standard_metadata(
            prompt=prompt,
            description=description,
            generator_name="Google GenAI",
            model_name=self.model_name
        )
        
        # Use base class save with backup functionality
        self._save_image_with_backup(image, save_path, metadata_fields)
    
    def generate_batch(self, prompts: List[ArtPrompt], base_output_dir: str, 
                      images_per_prompt: int = 1, aspect_ratio: Optional[str] = None,
                      safety_filter_level: Optional[str] = None, 
                      person_generation: Optional[str] = None) -> dict:
        """
        Generate images for multiple prompts using threading.
        
        Args:
            prompts: List of ArtPrompt objects
            base_output_dir: Base directory for output
            images_per_prompt: Number of images to generate per prompt
            aspect_ratio: Override aspect ratio for all prompts
            safety_filter_level: Override safety filter level for all prompts
            person_generation: Override person generation setting for all prompts
            
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
                    aspect_ratio=aspect_ratio,
                    safety_filter_level=safety_filter_level,
                    person_generation=person_generation
                )
                return prompt.filename, generated_paths
                
            except Exception as e:
                print(f"✗ Failed: {prompt.filename}: {e}")
                return prompt.filename, []
        
        # Execute with threading
        max_workers = get_max_workers('genai', len(prompts))
        print(f"Using {max_workers} concurrent threads for GenAI generation (within 50 RPM default quota)")
        
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
            generator = GoogleGenAIGenerator(api_key)
            # You might want to add a simple test prompt here
            print("✓ Google GenAI connection successful")
            return True
        except Exception as e:
            print(f"✗ Google GenAI connection failed: {e}")
            return False


def main():
    """Test the Google GenAI generator."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python google_genai.py <test_prompt>")
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
        generator = GoogleGenAIGenerator()
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