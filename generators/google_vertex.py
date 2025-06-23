"""
Google Vertex AI integration for AI art generation.
Supports multiple Imagen models (Imagen 3 and Imagen 4) via Vertex AI.
Uses Application Default Credentials for authentication.
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
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

# Add the parent directory to the path for imports
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from prompts.prompt_parser import ArtPrompt
from config import get_config


def get_max_workers(service: str, num_prompts: int) -> int:
    """Determine optimal thread count based on API limits."""
    limits = {
        'gpt4o': 5,          # Tier 1: 5 images per minute
        'genai': 8,        # Within 50 RPM default quota
        'vertex': 6,         # Conservative for Vertex AI (varies by model)
        'vertex-imagen4': 4, # Imagen 4: 20 RPM limit, be conservative
    }
    return min(num_prompts, limits.get(service, 5))


class GoogleVertexGenerator(BaseGenerator):
    """Google Vertex AI image generator supporting multiple Imagen models."""
    
    def __init__(self, project_id: Optional[str] = None, location: Optional[str] = None, 
                 config_path: Optional[str] = None):
        """Initialize the generator with project credentials."""
        if not VERTEX_AI_AVAILABLE:
            raise ImportError("Vertex AI libraries not installed. Run: pip install google-cloud-aiplatform")
        
        # Get project configuration
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("Google Cloud Project ID not provided. Set GOOGLE_CLOUD_PROJECT environment variable or pass project_id parameter.")
        
        self.location = location or os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        
        # Load configuration and initialize base class
        config = get_config(config_path)
        super().__init__(config)
        
        # Supported models with their configurations
        self.models = {
            'imagen-3.0-generate-002': {
                'class': ImageGenerationModel,
                'name': 'imagen-3.0-generate-002',
                'max_images': 8,
                'rpm_limit': 50,
                'supports': ['aspect_ratio', 'safety_filter_level', 'person_generation', 'seed', 'add_watermark']
            },
            'imagen-4.0-generate-preview-06-06': {
                'class': ImageGenerationModel,
                'name': 'imagen-4.0-generate-preview-06-06', 
                'max_images': 4,
                'rpm_limit': 20,
                'supports': ['aspect_ratio', 'safety_filter_level', 'person_generation']
            },
            'imagen-4.0-fast-generate-preview-06-06': {
                'class': ImageGenerationModel,
                'name': 'imagen-4.0-fast-generate-preview-06-06',
                'max_images': 4,
                'rpm_limit': 20,
                'supports': ['aspect_ratio', 'safety_filter_level', 'person_generation']
            },
            'imagen-4.0-ultra-generate-preview-06-06': {
                'class': ImageGenerationModel,
                'name': 'imagen-4.0-ultra-generate-preview-06-06',
                'max_images': 4,
                'rpm_limit': 20,
                'supports': ['aspect_ratio', 'safety_filter_level', 'person_generation']
            }
        }
        
        # Quality to model mapping for Imagen 4
        self.quality_models = {
            'fast': 'imagen-4.0-fast-generate-preview-06-06',
            'standard': 'imagen-4.0-generate-preview-06-06',
            'ultra': 'imagen-4.0-ultra-generate-preview-06-06'
        }
        
        # Default model
        self.default_model = self.config.get('vertex.model', 'imagen-3.0-generate-002')
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model."""
        return self.models.get(model_name, self.models[self.default_model])
    
    def generate_image(self, prompt: ArtPrompt, output_dir: str, num_images: int = 1, 
                      model_name: Optional[str] = None, quality: Optional[str] = None,
                      aspect_ratio: Optional[str] = None, safety_filter_level: Optional[str] = None, 
                      person_generation: Optional[str] = None, add_watermark: Optional[bool] = None, 
                      seed: Optional[int] = None) -> List[str]:
        """
        Generate images using Google Vertex AI Imagen models.
        
        Args:
            prompt: ArtPrompt object containing prompt text and metadata
            output_dir: Directory where images should be saved
            num_images: Number of images to generate
            model_name: Imagen model to use (e.g., 'imagen-3.0-generate-002', 'imagen-4.0-generate-preview-06-06')
            quality: Quality level for Imagen 4 models ("fast", "standard", "ultra"). Overrides model_name if both provided.
            aspect_ratio: Override aspect ratio (e.g., "16:9", "3:4", "1:1")
            safety_filter_level: Override safety filter level
            person_generation: Override person generation setting
            add_watermark: Whether to add watermark (Imagen 3 only)
            seed: Seed for reproducible generation (Imagen 3 only)
            
        Returns:
            List of paths to generated images
        """
        # Get configuration for this prompt's category
        vertex_config = self.config.get_vertex_config(prompt.category)
        
        # Determine model - quality parameter overrides config and model_name
        if quality and quality in self.quality_models:
            final_model = self.quality_models[quality]
            final_quality = quality
        elif vertex_config.get('quality') and vertex_config.get('quality') in self.quality_models:
            # Use quality from config
            final_model = self.quality_models[vertex_config.get('quality')]
            final_quality = vertex_config.get('quality')
        else:
            final_model = model_name or self.default_model
            # Detect quality from model name if it's an Imagen 4 variant
            final_quality = None
            for qual, model in self.quality_models.items():
                if final_model == model:
                    final_quality = qual
                    break
        
        model_info = self.get_model_info(final_model)
        
        # Validate number of images
        max_images = model_info['max_images']
        if num_images > max_images:
            print(f"Warning: {final_model} supports max {max_images} images per request. Limiting to {max_images}.")
            num_images = max_images
        
        print(f"Generating {num_images} image(s) for: {prompt.filename}")
        print(f"Using model: {final_model}")
        if final_quality:
            print(f"Quality: {final_quality}")
        print(f"Prompt: {prompt.prompt[:100]}...")
        
        try:
            
            # Load the model
            model = model_info['class'].from_pretrained(final_model)
            
            # Build generation parameters
            generation_params = {
                'prompt': prompt.prompt,
                'number_of_images': num_images,
                'language': 'en'
            }
            
            # Add aspect ratio
            final_aspect_ratio = aspect_ratio or vertex_config.get('aspect_ratio', '1:1')
            generation_params['aspect_ratio'] = final_aspect_ratio
            
            # Add safety filter level  
            final_safety_level = safety_filter_level or vertex_config.get('safety_filter_level')
            if final_safety_level and 'safety_filter_level' in model_info['supports']:
                generation_params['safety_filter_level'] = final_safety_level
            
            # Add person generation setting
            final_person_gen = person_generation or vertex_config.get('person_generation')
            if final_person_gen and 'person_generation' in model_info['supports']:
                generation_params['person_generation'] = final_person_gen
            
            # Add watermark setting (Imagen 3 only)
            if add_watermark is not None and 'add_watermark' in model_info['supports']:
                generation_params['add_watermark'] = add_watermark
            elif 'add_watermark' in model_info['supports']:
                generation_params['add_watermark'] = vertex_config.get('add_watermark', False)
            
            # Add seed (Imagen 3 only)
            if seed is not None and 'seed' in model_info['supports']:
                generation_params['seed'] = seed
            
            print(f"  Using aspect ratio: {final_aspect_ratio}")
            if final_safety_level:
                print(f"  Using safety filter: {final_safety_level}")
            if final_person_gen:
                print(f"  Person generation: {final_person_gen}")
            if 'add_watermark' in generation_params:
                print(f"  Add watermark: {generation_params['add_watermark']}")
            if 'seed' in generation_params:
                print(f"  Using seed: {generation_params['seed']}")
            
            # Generate images
            images = model.generate_images(**generation_params)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save generated images
            saved_paths = []
            base_filename = Path(prompt.filename).stem  # Remove extension
            
            for idx, image in enumerate(images):
                # Create filename
                if num_images > 1:
                    filename = f"{base_filename}_{idx+1:03d}.png"
                else:
                    filename = f"{base_filename}.png"
                
                # Save path
                save_path = os.path.join(output_dir, filename)
                
                # Save image with prompt metadata
                self._save_image_with_metadata(image, save_path, prompt.prompt, prompt.description, final_model, final_quality)
                saved_paths.append(save_path)
                
                print(f"  ✓ Saved: {save_path}")
            
            return saved_paths
            
        except Exception as e:
            print(f"  ✗ Error generating image: {e}")
            raise
    
    def _save_image_with_metadata(self, vertex_image, save_path: str, prompt: str, description: str, model_name: str, quality: Optional[str] = None):
        """Save Vertex AI image with prompt metadata embedded in PNG."""
        # Create metadata using base class helper
        metadata_fields = self._create_standard_metadata(
            prompt=prompt,
            description=description,
            generator_name=f"Google Vertex AI - {model_name}",
            model_name=model_name,
            quality=quality
        )
        
        # Use base class save with backup functionality
        # This will automatically convert vertex_image to PIL and handle backups
        self._save_image_with_backup(vertex_image, save_path, metadata_fields)
    
    def generate_batch(self, prompts: List[ArtPrompt], base_output_dir: str, 
                      images_per_prompt: int = 1, model_name: Optional[str] = None,
                      quality: Optional[str] = None, aspect_ratio: Optional[str] = None, 
                      safety_filter_level: Optional[str] = None, person_generation: Optional[str] = None, 
                      add_watermark: Optional[bool] = None, seed: Optional[int] = None) -> dict:
        """
        Generate images for multiple prompts using threading.
        
        Args:
            prompts: List of ArtPrompt objects
            base_output_dir: Base directory for output
            images_per_prompt: Number of images to generate per prompt
            model_name: Imagen model to use for all prompts
            quality: Quality level for Imagen 4 models ("fast", "standard", "ultra")
            aspect_ratio: Override aspect ratio for all prompts
            safety_filter_level: Override safety filter level for all prompts
            person_generation: Override person generation setting for all prompts
            add_watermark: Override watermark setting for all prompts
            seed: Override seed for all prompts
            
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
                    model_name=model_name,
                    quality=quality,
                    aspect_ratio=aspect_ratio,
                    safety_filter_level=safety_filter_level,
                    person_generation=person_generation,
                    add_watermark=add_watermark,
                    seed=seed
                )
                return prompt.filename, generated_paths
                
            except Exception as e:
                print(f"✗ Failed: {prompt.filename}: {e}")
                return prompt.filename, []
        
        # Determine threading based on model
        # Use quality parameter if provided, otherwise use model_name
        if quality and quality in self.quality_models:
            final_model = self.quality_models[quality]
        else:
            final_model = model_name or self.default_model
        
        service_key = 'vertex-imagen4' if 'imagen-4' in final_model else 'vertex'
        max_workers = get_max_workers(service_key, len(prompts))
        
        model_info = self.get_model_info(final_model)
        print(f"Using {max_workers} concurrent threads for {final_model} (within {model_info['rpm_limit']} RPM limit)")
        
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
    def test_connection(project_id: Optional[str] = None, location: Optional[str] = None) -> bool:
        """Test if the Vertex AI connection works."""
        try:
            generator = GoogleVertexGenerator(project_id, location)
            print(f"✓ Google Vertex AI connection successful")
            print(f"  Project: {generator.project_id}")
            print(f"  Location: {generator.location}")
            print(f"  Available models: {list(generator.models.keys())}")
            return True
        except Exception as e:
            print(f"✗ Google Vertex AI connection failed: {e}")
            return False


def main():
    """Test the Google Vertex AI generator."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python google_vertex.py <test_prompt> [model_name]")
        print("Available models:")
        generator = GoogleVertexGenerator()
        for model in generator.models.keys():
            print(f"  - {model}")
        sys.exit(1)
    
    test_prompt_text = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None
    
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
        generator = GoogleVertexGenerator()
        output_dir = "test_output"
        
        paths = generator.generate_image(test_prompt, output_dir, model_name=model_name)
        print(f"Generated {len(paths)} image(s):") 
        for path in paths:
            print(f"  {path}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()