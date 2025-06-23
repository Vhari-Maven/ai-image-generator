"""
Prompt parser for extracting AI art generation prompts from JSON documentation.
Parses the image-gen.json file to extract structured prompt data.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ArtPrompt:
    """Represents a single art generation prompt with metadata."""
    filename: str
    description: str
    prompt: str
    category: str  # 'background' or 'character'
    output_path: str
    id: str = ""
    title: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class PromptParser:
    """Parser for extracting prompts from JSON documentation."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.config_path = self.project_root / "docs" / "ai-art" / "prompts" / "config.json"
        self.stories_dir = self.project_root / "docs" / "ai-art" / "prompts"
    
    def parse_story_prompts(self, story_name: str) -> List[ArtPrompt]:
        """Parse all prompts for a specific story from modular JSON files."""
        story_file = self.stories_dir / f"{story_name}.json"
        
        if not story_file.exists():
            raise FileNotFoundError(f"Story file not found: {story_file}")
        
        try:
            with open(story_file, 'r', encoding='utf-8') as f:
                story_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {story_file}: {e}")
        
        prompts = []
        
        # Parse background images
        for bg_data in story_data.get('backgrounds', []):
            prompt = self._create_art_prompt(bg_data, 'background', story_name)
            if prompt:
                prompts.append(prompt)
        
        # Parse character portraits
        for char_data in story_data.get('characters', []):
            prompt = self._create_art_prompt(char_data, 'character', story_name)
            if prompt:
                prompts.append(prompt)
        
        return prompts
    
    def _create_art_prompt(self, data: Dict, category: str, story_name: str) -> Optional[ArtPrompt]:
        """Create an ArtPrompt object from JSON data."""
        # Validate required fields
        required_fields = ['filename', 'prompt', 'description']
        for field in required_fields:
            if not data.get(field):
                print(f"Warning: Missing required field '{field}' in {category} prompt")
                return None
        
        # Determine output path
        if category == "background":
            output_dir = "backgrounds"
        else:
            output_dir = "characters"
        
        output_path = f"src/stories/{story_name}/assets/{output_dir}/{data['filename']}"
        
        return ArtPrompt(
            id=data.get('id', ''),
            filename=data['filename'],
            title=data.get('title', ''),
            description=data['description'],
            prompt=data['prompt'],
            category=category,
            output_path=output_path,
            tags=data.get('tags', [])
        )
    
    def get_available_stories(self) -> List[str]:
        """Get list of available stories that have asset directories."""
        stories_dir = self.project_root / "src" / "stories"
        if not stories_dir.exists():
            return []
        
        available_stories = []
        for story_dir in stories_dir.iterdir():
            if story_dir.is_dir() and story_dir.name != "__pycache__":
                # Check if it has an assets directory or could have one
                assets_dir = story_dir / "assets"
                if assets_dir.exists() or (story_dir / "story.json").exists():
                    available_stories.append(story_dir.name)
        
        return available_stories
    
    def get_story_metadata(self, story_name: str) -> Optional[Dict]:
        """Get metadata for a specific story."""
        story_file = self.stories_dir / f"{story_name}.json"
        
        if not story_file.exists():
            return None
        
        try:
            with open(story_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('metadata', {})
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def get_prompts_by_category(self, story_name: str, category: str) -> List[ArtPrompt]:
        """Get prompts filtered by category (background or character)."""
        all_prompts = self.parse_story_prompts(story_name)
        return [p for p in all_prompts if p.category == category]
    
    def get_prompts_by_tags(self, story_name: str, tags: List[str]) -> List[ArtPrompt]:
        """Get prompts that contain any of the specified tags."""
        all_prompts = self.parse_story_prompts(story_name)
        return [p for p in all_prompts if any(tag in p.tags for tag in tags)]
    
    
    def parse_prompts_file(self, prompt_file_path: str) -> List[ArtPrompt]:
        """Parse prompts from a standalone JSON file."""
        file_path = Path(prompt_file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        
        # Get prompts array from the data
        prompts_data = data.get('prompts', [])
        if not prompts_data:
            raise ValueError(f"No 'prompts' array found in {file_path}")
        
        prompts = []
        for i, prompt_data in enumerate(prompts_data):
            try:
                prompt = ArtPrompt(
                    id=prompt_data.get('id', f'prompt-{i}'),
                    title=prompt_data.get('title', ''),
                    filename=prompt_data.get('filename', f'image-{i}.png'),
                    description=prompt_data.get('description', ''),
                    prompt=prompt_data['prompt'],  # Required field
                    category=prompt_data.get('category', 'background'),
                    output_path='',  # Will be set by generator
                    tags=prompt_data.get('tags', [])
                )
                prompts.append(prompt)
            except KeyError as e:
                raise ValueError(f"Missing required field {e} in prompt {i}")
        
        return prompts
    
    def get_global_config(self) -> Dict:
        """Get global configuration settings."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Return minimal defaults if no config found
        return {
            "version": "2.0",
            "technical_specs": {
                "aspect_ratios": {
                    "characters": "3:4",
                    "backgrounds": "16:9"
                },
                "recommended_model": "genai"
            }
        }


def main():
    """Test the prompt parser."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prompt_parser.py <story_name>")
        sys.exit(1)
    
    story_name = sys.argv[1]
    project_root = Path(__file__).parent.parent.parent.parent
    
    parser = PromptParser(str(project_root))
    
    try:
        prompts = parser.parse_story_prompts(story_name)
        print(f"Found {len(prompts)} prompts for story '{story_name}':")
        
        # Group by category
        backgrounds = [p for p in prompts if p.category == 'background']
        characters = [p for p in prompts if p.category == 'character']
        
        if backgrounds:
            print(f"\nBackgrounds ({len(backgrounds)}):")
            for prompt in backgrounds:
                print(f"  • {prompt.title or prompt.filename}")
                print(f"    ID: {prompt.id}")
                print(f"    Tags: {', '.join(prompt.tags)}")
                print(f"    Prompt: {prompt.prompt[:80]}...")
                print()
        
        if characters:
            print(f"\nCharacters ({len(characters)}):")
            for prompt in characters:
                print(f"  • {prompt.title or prompt.filename}")
                print(f"    ID: {prompt.id}")
                print(f"    Tags: {', '.join(prompt.tags)}")
                print(f"    Prompt: {prompt.prompt[:80]}...")
                print()
        
        # Show metadata
        metadata = parser.get_story_metadata(story_name)
        if metadata:
            print(f"\nStory Metadata:")
            print(f"  Title: {metadata.get('title', 'N/A')}")
            print(f"  Style: {metadata.get('style', {}).get('name', 'N/A')}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()