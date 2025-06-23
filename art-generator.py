#!/usr/bin/env python3
"""
AI Art Generator CLI
Generates visual assets using AI image generation services.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading .env file
    pass

# Add the current directory and parent to the path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from prompts.prompt_parser import PromptParser, ArtPrompt
from generators.google_genai import GoogleGenAIGenerator
from generators.google_vertex import GoogleVertexGenerator
from generators.openai_gpt4o import OpenAIGPT4oGenerator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate AI art using multiple image generation services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from a prompt file
  python art-generator.py --prompts example-prompts.json
  
  # Generate only backgrounds
  python art-generator.py --prompts example-prompts.json --type backgrounds
  
  # Generate with multiple images per prompt
  python art-generator.py --prompts example-prompts.json --images-per-prompt 3
  
  # Generate only one specific image by ID
  python art-generator.py --prompts example-prompts.json --image-id bg-mystical-forest
  
  # Generate multiple specific images by ID (multiple flags)
  python art-generator.py --prompts example-prompts.json --image-id bg-mystical-forest --image-id char-wise-sage
  
  # Generate multiple specific images by ID (comma-separated)
  python art-generator.py --prompts example-prompts.json --image-id bg-mystical-forest,char-wise-sage
  
  # Generate with specific aspect ratio override
  python art-generator.py --prompts example-prompts.json --aspect-ratio 16:9
  
  # Generate characters with portrait aspect ratio
  python art-generator.py --prompts example-prompts.json --type characters --aspect-ratio 3:4
  
  # Test connection to Google GenAI
  python art-generator.py --test-connection
  
  # Generate using Google Vertex AI (requires project setup)
  python art-generator.py --prompts example-prompts.json --service vertex --project-id my-project-123
  
  # Generate with Vertex AI using specific location and model
  python art-generator.py --prompts example-prompts.json --service vertex --project-id my-project-123 --location us-west1 --model imagen-4.0-generate-preview-06-06
  
  # Test Vertex AI connection
  python art-generator.py --test-connection --service vertex --project-id my-project-123
  
  # Generate backgrounds with Vertex AI using environment variables
  python art-generator.py --prompts example-prompts.json --service vertex --type backgrounds
  
  # Custom output directory
  python art-generator.py --prompts example-prompts.json --output-dir ./my-images
        """
    )
    
    # Prompt file selection (primary method)
    parser.add_argument(
        '--prompts', '-p',
        help='JSON file containing prompts to generate'
    )
    
    # Asset type filtering
    parser.add_argument(
        '--type', '-t',
        choices=['backgrounds', 'characters', 'all'],
        default='all',
        help='Type of assets to generate (default: all)'
    )
    
    # Specific image selection
    parser.add_argument(
        '--image-id',
        action='append',
        help='Generate only images with these specific IDs (can be used multiple times or comma-separated, e.g., --image-id id1,id2,id3 or --image-id id1 --image-id id2)'
    )
    
    # Image generation options
    parser.add_argument(
        '--images-per-prompt', '-n',
        type=int,
        default=1,
        help='Number of images to generate per prompt (default: 1)'
    )
    
    parser.add_argument(
        '--aspect-ratio',
        choices=['1:1', '3:4', '4:3', '9:16', '16:9'],
        help='Override aspect ratio for all images (GenAI only, default: uses config/category defaults)'
    )
    
    parser.add_argument(
        '--size',
        choices=['256x256', '512x512', '1024x1024', '1024x1536', '1536x1024'],
        help='Override image size for all images (GPT-4o only, default: uses config/category defaults)'
    )
    
    parser.add_argument(
        '--quality',
        choices=['low', 'medium', 'high', 'fast', 'standard', 'ultra'],
        help='Override image quality (GPT-4o: low/medium/high, Vertex: fast/standard/ultra, default: varies by service)'
    )
    
    parser.add_argument(
        '--style',
        choices=['natural', 'vivid'],
        help='Override image style (GPT-4o only, default: natural)'
    )
    
    parser.add_argument(
        '--safety-filter',
        choices=['BLOCK_LOW_AND_ABOVE'],
        help='Override safety filter level (GenAI only, currently only BLOCK_LOW_AND_ABOVE supported)'
    )
    
    parser.add_argument(
        '--allow-people',
        choices=['DONT_ALLOW', 'ALLOW_ADULT'],
        help='Override person generation setting (GenAI only, default: uses config defaults)'
    )
    
    # Service selection
    parser.add_argument(
        '--service',
        choices=['genai', 'gpt4o', 'vertex'],
        default='genai',
        help='AI service to use (default: genai)'
    )
    
    # API key override
    parser.add_argument(
        '--api-key',
        help='API key for the service (overrides environment variable)'
    )
    
    # Google Vertex AI specific arguments
    parser.add_argument(
        '--project-id',
        help='Google Cloud Project ID for Vertex AI (if not provided, will use GOOGLE_CLOUD_PROJECT environment variable)'
    )
    
    parser.add_argument(
        '--location',
        default='us-central1',
        help='Google Cloud location for Vertex AI (default: us-central1, can also use GOOGLE_CLOUD_LOCATION environment variable)'
    )
    
    parser.add_argument(
        '--model',
        help='Override model name (vertex: imagen-3.0-generate-002, imagen-4.0-generate-preview-06-06)'
    )
    
    # Utility commands
    parser.add_argument(
        '--list-prompts',
        action='store_true',
        help='List available prompts in the file and exit'
    )
    
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test API connection and exit'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        help='Custom output directory (default: ./generated-images)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without actually generating'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize prompt parser  
    prompt_parser = PromptParser(str(current_dir))
    
    # Handle utility commands
    if args.test_connection:
        test_service_connection(args.service, args.api_key, args.project_id, args.location)
        return
    
    if args.list_prompts:
        if not args.prompts:
            print("Error: --prompts is required when using --list-prompts")
            sys.exit(1)
        list_available_prompts(prompt_parser, args.prompts)
        return
    
    # Validate required arguments
    if not args.prompts:
        print("Error: --prompts is required for generation")
        parser.print_help()
        sys.exit(1)
    
    # Validate prompt file exists
    prompt_file = Path(args.prompts)
    if not prompt_file.exists():
        print(f"Error: Prompt file '{args.prompts}' not found.")
        sys.exit(1)
    
    # Parse prompts from the file
    try:
        all_prompts = prompt_parser.parse_prompts_file(args.prompts)
    except Exception as e:
        print(f"Error parsing prompts: {e}")
        sys.exit(1)
    
    if not all_prompts:
        print(f"No prompts found in '{args.prompts}'")
        sys.exit(1)
    
    # Process comma-separated image IDs
    if args.image_id:
        # Expand comma-separated values: ['id1,id2', 'id3'] -> ['id1', 'id2', 'id3']
        expanded_ids = []
        for id_group in args.image_id:
            expanded_ids.extend([id.strip() for id in id_group.split(',')])
        args.image_id = expanded_ids
    
    # Filter prompts by image ID first (most specific)
    if args.image_id:
        prompts = [p for p in all_prompts if p.id in args.image_id]
        if not prompts:
            print(f"Error: No images found with IDs '{', '.join(args.image_id)}'")
            print("Available image IDs:")
            for p in all_prompts:
                print(f"  {p.id} ({p.category}): {p.title}")
            sys.exit(1)
        
        # Check for invalid IDs
        found_ids = {p.id for p in prompts}
        invalid_ids = set(args.image_id) - found_ids
        if invalid_ids:
            print(f"Warning: Image IDs not found: {', '.join(invalid_ids)}")
            print("Available image IDs:")
            for p in all_prompts:
                print(f"  {p.id} ({p.category}): {p.title}")
            print(f"Continuing with valid IDs: {', '.join(found_ids)}")
            print()
    # Filter prompts by type
    elif args.type != 'all':
        # Convert 'backgrounds' -> 'background', 'characters' -> 'character'
        filter_type = args.type.rstrip('s')
        prompts = [p for p in all_prompts if p.category == filter_type]
    else:
        prompts = all_prompts
    
    if not prompts:
        print(f"No {args.type} prompts found for '{args.prompts}'")
        sys.exit(1)
    
    # Show what will be generated
    print(f"Prompt file: {args.prompts}")
    print(f"Service: {args.service}")
    if args.image_id:
        print(f"Specific images: {', '.join(args.image_id)}")
    else:
        print(f"Type: {args.type}")
    print(f"Images per prompt: {args.images_per_prompt}")
    if args.aspect_ratio:
        print(f"Aspect ratio override (GenAI): {args.aspect_ratio}")
    if args.size:
        print(f"Size override (GPT-4o): {args.size}")
    if args.quality:
        if args.service == 'gpt4o':
            print(f"Quality override (GPT-4o): {args.quality}")
        elif args.service == 'vertex':
            print(f"Quality override (Vertex): {args.quality}")
        else:
            print(f"Quality override: {args.quality}")
    if args.style:
        print(f"Style override (GPT-4o): {args.style}")
    if args.safety_filter:
        print(f"Safety filter override (GenAI): {args.safety_filter}")
    if args.allow_people:
        print(f"Person generation override (GenAI): {args.allow_people}")
    print(f"Total prompts: {len(prompts)}")
    
    if args.verbose:
        print("\nPrompts to generate:")
        for prompt in prompts:
            print(f"  {prompt.category}: {prompt.filename} - {prompt.description}")
    
    if args.dry_run:
        print("\nDry run - no images will be generated")
        return
    
    # Initialize generator
    try:
        if args.service == 'genai':
            generator = GoogleGenAIGenerator(args.api_key)
        elif args.service == 'gpt4o':
            generator = OpenAIGPT4oGenerator(args.api_key)
        elif args.service == 'vertex':
            generator = GoogleVertexGenerator(args.project_id, args.location)
        else:
            print(f"Error: Service '{args.service}' not supported")
            sys.exit(1)
    except Exception as e:
        print(f"Error initializing generator: {e}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        # Default output directory
        base_output_dir = "./generated-images"
    
    # Generate images
    print(f"\nGenerating images to: {base_output_dir}")
    print("=" * 50)
    
    try:
        if args.service == 'genai':
            results = generator.generate_batch(
                prompts, base_output_dir, args.images_per_prompt,
                aspect_ratio=args.aspect_ratio,
                safety_filter_level=args.safety_filter,
                person_generation=args.allow_people
            )
        elif args.service == 'gpt4o':
            results = generator.generate_batch(
                prompts, base_output_dir, args.images_per_prompt,
                size=args.size,
                quality=args.quality,
                style=args.style
            )
        elif args.service == 'vertex':
            results = generator.generate_batch(
                prompts, base_output_dir, args.images_per_prompt,
                model_name=args.model,
                quality=args.quality,
                aspect_ratio=args.aspect_ratio,
                safety_filter_level=args.safety_filter,
                person_generation=args.allow_people
            )
        else:
            print(f"Error: Service '{args.service}' not supported for generation")
            sys.exit(1)
        
        # Summary
        print("\n" + "=" * 50)
        print("Generation Summary:")
        
        total_generated = 0
        failed_count = 0
        
        for filename, paths in results.items():
            if paths:
                print(f"  ✓ {filename}: {len(paths)} image(s)")
                total_generated += len(paths)
            else:
                print(f"  ✗ {filename}: Failed")
                failed_count += 1
        
        print(f"\nTotal generated: {total_generated}")
        if failed_count > 0:
            print(f"Failed: {failed_count}")
        
        print("Done!")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during generation: {e}")
        sys.exit(1)


def list_available_prompts(prompt_parser: PromptParser, prompt_file: str):
    """List all available prompts in the specified file."""
    try:
        prompts = prompt_parser.parse_prompts_file(prompt_file)
    except Exception as e:
        print(f"Error parsing prompts file: {e}")
        return
    
    if not prompts:
        print("No prompts found in file")
        return
    
    print(f"Available prompts in {prompt_file}:")
    
    # Group by category
    categories = {}
    for prompt in prompts:
        if prompt.category not in categories:
            categories[prompt.category] = []
        categories[prompt.category].append(prompt)
    
    for category, category_prompts in categories.items():
        print(f"  {category.title()}s ({len(category_prompts)}):")
        for prompt in category_prompts:
            print(f"    {prompt.id}: {prompt.title}")
            if prompt.description:
                print(f"      {prompt.description}")
        print()


def test_service_connection(service: str, api_key: Optional[str] = None, project_id: Optional[str] = None, location: Optional[str] = None):
    """Test connection to the specified service."""
    print(f"Testing {service} connection...")
    
    if service == 'genai':
        success = GoogleGenAIGenerator.test_connection(api_key)
    elif service == 'gpt4o':
        success = OpenAIGPT4oGenerator.test_connection(api_key)
    elif service == 'vertex':
        success = GoogleVertexGenerator.test_connection(project_id, location)
    else:
        print(f"Service '{service}' not supported")
        return
    
    if success:
        print("Connection test passed!")
    else:
        print("Connection test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()