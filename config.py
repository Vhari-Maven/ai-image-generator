"""
Configuration management for AI Art Generator.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config:
    """Configuration manager for the AI art generator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration."""
        self.config_dir = Path(__file__).parent
        
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Look for config.yaml in the same directory as this script
            self.config_path = self.config_dir / "config.yaml"
        
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            print(f"Config file not found: {self.config_path}")
            print(f"Using default configuration. Copy {self.config_dir}/config.yaml.example to {self.config_path} to customize.")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Merge with defaults to ensure all required keys exist
            default_config = self._get_default_config()
            return self._merge_configs(default_config, config)
            
        except Exception as e:
            print(f"Error loading config file {self.config_path}: {e}")
            print("Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'api': {
                'google_ai_key': None,
                'openai_key': None
            },
            'generation': {
                'images_per_prompt': 1,
                'default_service': 'genai',
                'image_format': 'png',
                'optimize_images': True,
                'embed_metadata': True
            },
            'output': {
                'create_backups': False,
                'organize_by_category': True
            },
            'genai': {
                'model': 'imagen-3.0-generate-002',
                'defaults': {
                    'aspect_ratio': '1:1',
                    'safety_filter_level': 'BLOCK_LOW_AND_ABOVE',
                    'person_generation': 'ALLOW_ADULT'
                },
                'categories': {
                    'background': {
                        'aspect_ratio': '16:9',
                        'safety_filter_level': 'BLOCK_LOW_AND_ABOVE',
                        'person_generation': 'ALLOW_ADULT'
                    },
                    'character': {
                        'aspect_ratio': '3:4',
                        'safety_filter_level': 'BLOCK_LOW_AND_ABOVE',
                        'person_generation': 'ALLOW_ADULT'
                    }
                }
            },
            'openai': {
                'model': 'gpt-image-1',
                'defaults': {
                    'size': '1024x1024',
                    'quality': 'medium',
                    'style': 'natural'
                },
                'categories': {
                    'background': {
                        'size': '1536x1024',
                        'quality': 'medium',
                        'style': 'natural'
                    },
                    'character': {
                        'size': '1024x1536',
                        'quality': 'high',
                        'style': 'vivid'
                    }
                }
            },
            'vertex': {
                'model': 'imagen-3.0-generate-002',
                'project_id': None,
                'location': 'us-central1',
                'defaults': {
                    'aspect_ratio': '1:1',
                    'quality': 'standard',
                    'safety_filter_level': 'block_some',
                    'person_generation': 'allow_adult',
                    'add_watermark': False
                },
                'categories': {
                    'background': {
                        'aspect_ratio': '16:9',
                        'quality': 'standard',
                        'safety_filter_level': 'block_some',
                        'person_generation': 'allow_adult',
                        'add_watermark': False
                    },
                    'character': {
                        'aspect_ratio': '3:4',
                        'quality': 'standard',
                        'safety_filter_level': 'block_some',
                        'person_generation': 'allow_adult',
                        'add_watermark': False
                    }
                },
                'models': {
                    'imagen-3.0-generate-002': {
                        'max_images': 8,
                        'rpm_limit': 50,
                        'supports_watermark': True,
                        'supports_seed': True
                    },
                    'imagen-4.0-generate-preview-06-06': {
                        'max_images': 4,
                        'rpm_limit': 20,
                        'supports_watermark': False,
                        'supports_seed': False
                    }
                }
            }
        }
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with defaults."""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'api.google_ai_key')."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a service, checking environment variables first."""
        if service == 'genai':
            # Check environment variable first
            env_key = os.getenv('GOOGLE_AI_API_KEY')
            if env_key:
                return env_key
            
            # Fall back to config file
            return self.get('api.google_ai_key')
        
        elif service == 'openai' or service == 'gpt4o':
            # Check environment variable first
            env_key = os.getenv('OPENAI_API_KEY')
            if env_key:
                return env_key
            
            # Fall back to config file
            return self.get('api.openai_key')
        
        elif service == 'vertex':
            # Vertex AI uses Application Default Credentials
            # Return project ID instead of API key
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            if project_id:
                return project_id
            
            # Fall back to config file
            return self.get('vertex.project_id')
        
        return None
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration."""
        return self.get('generation', {})
    
    def get_service_config(self, service: str) -> Dict[str, Any]:
        """Get service-specific configuration."""
        return self.get(service, {})
    
    def get_genai_config(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get GenAI configuration with category-specific overrides."""
        # Start with defaults
        config = self.get('genai.defaults', {}).copy()
        
        # Apply category-specific overrides if provided
        if category:
            category_config = self.get(f'genai.categories.{category}', {})
            config.update(category_config)
        
        return config
    
    def get_openai_config(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get OpenAI configuration with category-specific overrides."""
        # Start with defaults
        config = self.get('openai.defaults', {}).copy()
        
        # Apply category-specific overrides if provided
        if category:
            category_config = self.get(f'openai.categories.{category}', {})
            config.update(category_config)
        
        return config
    
    def get_vertex_config(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get Vertex AI configuration with category-specific overrides."""
        # Start with defaults
        config = self.get('vertex.defaults', {}).copy()
        
        # Apply category-specific overrides if provided
        if category:
            category_config = self.get(f'vertex.categories.{category}', {})
            config.update(category_config)
        
        return config


# Global config instance
_config = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None or config_path:
        _config = Config(config_path)
    return _config


def main():
    """Test the configuration system."""
    config = get_config()
    
    print("Configuration loaded:")
    print(f"  Default service: {config.get('generation.default_service')}")
    print(f"  Images per prompt: {config.get('generation.images_per_prompt')}")
    print(f"  Google AI key configured: {'Yes' if config.get_api_key('genai') else 'No'}")
    print(f"  OpenAI key configured: {'Yes' if config.get_api_key('openai') else 'No'}")
    print(f"  Vertex AI project configured: {'Yes' if config.get_api_key('vertex') else 'No'}")
    print(f"  GenAI model: {config.get('genai.model')}")
    print(f"  OpenAI model: {config.get('openai.model')}")
    print(f"  Vertex AI model: {config.get('vertex.model')}")
    print(f"  Vertex AI location: {config.get('vertex.location')}")


if __name__ == "__main__":
    main()