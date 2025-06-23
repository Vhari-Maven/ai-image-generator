"""
Base generator class providing common functionality for all AI image generators.
"""

import os
import io
import shutil
from datetime import datetime
from typing import Dict, Any, Union
from PIL import Image, PngImagePlugin


class BaseGenerator:
    """Base class for all AI image generators with shared backup and save functionality."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
    
    def _save_image_with_backup(self, image_data: Any, save_path: str, metadata_fields: Dict[str, Any]) -> None:
        """
        Unified save with backup logic for all generators.
        
        Args:
            image_data: Image data (PIL Image, Vertex AI image, or other format)
            save_path: Full path where image should be saved
            metadata_fields: Dictionary of metadata to embed in PNG
        """
        # 1. Handle backup if needed
        if os.path.exists(save_path) and self.config.get('output.create_backups', False):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.basename(save_path)
            base_name = os.path.splitext(filename)[0]
            extension = os.path.splitext(filename)[1]
            
            # Create backup in root/assets/drafts folder
            repo_root = self._find_repo_root(save_path)
            backup_dir = os.path.join(repo_root, 'assets', 'drafts')
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_filename = f"{base_name}_{timestamp}_backup{extension}"
            backup_path = os.path.join(backup_dir, backup_filename)
            shutil.copy2(save_path, backup_path)
            print(f"  📁 Backed up existing file to: {backup_path}")
        
        # 2. Convert any image type to PIL Image
        pil_image = self._convert_to_pil(image_data)
        
        # 3. Create standardized metadata
        metadata = PngImagePlugin.PngInfo()
        for key, value in metadata_fields.items():
            if value is not None:  # Skip None values
                metadata.add_text(key, str(value))
        
        # 4. Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 5. Single save operation
        pil_image.save(save_path, pnginfo=metadata, optimize=True)
    
    def _convert_to_pil(self, image_data: Any) -> Image.Image:
        """
        Convert any image type to PIL Image.
        
        Args:
            image_data: Image in various formats (PIL, Vertex AI, bytes, etc.)
            
        Returns:
            PIL Image object
        """
        # Already a PIL Image
        if isinstance(image_data, Image.Image):
            return image_data
        
        # Vertex AI GeneratedImage - access raw image data
        if hasattr(image_data, '_image_bytes'):
            return Image.open(io.BytesIO(image_data._image_bytes))
        
        # Try accessing .data attribute (common in cloud APIs)
        if hasattr(image_data, 'data'):
            return Image.open(io.BytesIO(image_data.data))
        
        # Raw bytes
        if isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data))
        
        # If all else fails, assume it's already a PIL Image
        # This shouldn't happen but provides a fallback
        return image_data
    
    def _create_standard_metadata(self, prompt: str, description: str, generator_name: str, 
                                 model_name: str = None, **extra_fields) -> Dict[str, Any]:
        """
        Create standardized metadata fields that all generators should include.
        
        Args:
            prompt: The AI generation prompt
            description: Human-readable description
            generator_name: Name of the AI service (e.g., "OpenAI GPT-4o")
            model_name: Specific model used
            **extra_fields: Additional service-specific metadata
            
        Returns:
            Dictionary of metadata fields
        """
        metadata = {
            "prompt": prompt,
            "description": description,
            "generator": generator_name,
            "generated_at": datetime.now().isoformat()
        }
        
        if model_name:
            metadata["model"] = model_name
            
        # Add any extra fields
        metadata.update(extra_fields)
        
        return metadata
    
    def _find_repo_root(self, file_path: str) -> str:
        """
        Find the repository root by looking for .git directory.
        
        Args:
            file_path: Path to start searching from
            
        Returns:
            Path to repository root
        """
        current_path = os.path.dirname(os.path.abspath(file_path))
        
        while current_path != os.path.dirname(current_path):  # Not at filesystem root
            if os.path.exists(os.path.join(current_path, '.git')):
                return current_path
            current_path = os.path.dirname(current_path)
        
        # Fallback: if no .git found, assume current working directory
        return os.getcwd()