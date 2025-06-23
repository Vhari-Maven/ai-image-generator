# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Art Generator - a Python CLI tool for generating visual assets using multiple AI image generation services including Google GenAI (Imagen), Google Vertex AI (Imagen 3/4), and OpenAI GPT-4o. The tool supports batch generation, flexible configuration, and structured prompt management.

## Development Commands

### Package Management
```bash
# Install dependencies (recommended)
uv sync

# Install with development dependencies 
uv sync --dev
```

### Code Quality
```bash
# Format code
uv run black .

# Lint code
uv run ruff check

# Run tests (when available)
uv run pytest
```

### Running the Application
```bash
# Test connection to services
uv run python art-generator.py --test-connection

# Generate from prompt file
uv run python art-generator.py --prompts example-prompts.json

# Generate with specific service
uv run python art-generator.py --prompts prompts.json --service vertex

# Dry run to test configuration
uv run python art-generator.py --prompts prompts.json --dry-run
```

## Architecture

### Core Components

**Main CLI (`art-generator.py`)**: Entry point handling argument parsing, service initialization, and orchestrating the generation process.

**Configuration System (`config.py`)**: Centralized configuration management using YAML files with environment variable overrides. Supports service-specific defaults and category-based parameter overrides.

**Generator Architecture**: Plugin-based system with a base class pattern:
- `BaseGenerator`: Provides shared functionality for image saving, backup handling, and metadata embedding
- Service-specific generators inherit from BaseGenerator and implement their API integrations
- Located in `generators/` directory

**Prompt Management (`prompts/prompt_parser.py`)**: Handles parsing of JSON prompt files and creates structured `ArtPrompt` objects with metadata.

### Key Abstractions

**ArtPrompt**: Standardized data structure containing prompt text, metadata, category, and output configuration.

**Service Generators**: Each AI service (GenAI, Vertex AI, OpenAI) has its own generator class implementing service-specific parameters while sharing common functionality through BaseGenerator.

**Configuration Hierarchy**: Settings cascade from defaults → config file → environment variables → CLI arguments, with category-specific overrides supported for each service.

## Configuration

### Environment Setup
Create `.env` file with API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Create config file:
```bash
cp config.yaml.example config.yaml
# Customize settings as needed
```

### Service Configuration
- **Google GenAI**: Requires `GOOGLE_AI_API_KEY`
- **Google Vertex AI**: Requires `GOOGLE_CLOUD_PROJECT` + `gcloud auth application-default login`
- **OpenAI GPT-4o**: Requires `OPENAI_API_KEY`

## Prompt File Format

The tool uses JSON files with this structure:
```json
{
  "prompts": [
    {
      "id": "unique-id",
      "category": "background|character",
      "title": "Display Title",
      "filename": "output-filename.png",
      "description": "Human description",
      "prompt": "AI generation prompt text"
    }
  ]
}
```

## Adding New Generators

1. Create new generator class in `generators/`
2. Inherit from `BaseGenerator`
3. Implement required methods for API integration
4. Add service option to CLI argument parser
5. Update service initialization logic in main CLI

## Service-Specific Notes

- **Rate Limiting**: Each service has different rate limits and concurrent thread counts configured in the system
- **Model Support**: Vertex AI supports multiple Imagen models with different capabilities (3.0, 4.0 variants)
- **Parameter Mapping**: Each service uses different parameter names - the generators handle translation from common interface