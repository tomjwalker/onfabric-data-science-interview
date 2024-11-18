import os
from pathlib import Path
from typing import List, Dict, Set
import argparse

# Files/directories to ignore
IGNORE_PATTERNS = {
    '.git', '__pycache__', '.pytest_cache', '.env', '.venv', 'venv', '.idea', '.vscode',
    'egg-info', '*.egg-info', 'build', 'dist', '*.pyc', '*.pyo', '*.pyd', '.Python', 'develop-eggs', 'downloads', 'eggs', 'parts', 'sdist', 'var', '.installed.cfg', '*.egg'
}

# Key file descriptions
FILE_DESCRIPTIONS = {
    'catalog_queries.py': 'Handles personalised catalog search and recommendations',
    'process_catalog.py': 'Main script for processing and embedding catalog data',
    'catalog_processor.py': 'Core processing logic for fashion catalog items',
    'embeddings.py': 'Manages vector embeddings for catalog items',
    'schema.py': 'Data models and validation schemas',
    'model_config.py': 'Configuration for ML models and vector DB',
    'prompts.yaml': 'Template prompts for analysis tasks',
    'fashion_analysis.py': 'Analyzes fashion-related entities and trends',
    'user_comp_analysis.py': 'Comprehensive user behaviour analysis',
    'analysis_summary.txt': 'Generated user profile and recommendations',
    'fashion_analysis.json': 'Extracted fashion entities and topics',
    'fashion_catalog.json': 'Raw fashion product catalog',
    'search_history.json': 'User search query history',
    'chroma_db': 'Vector database storing processed and embedded fashion catalog items'
}

def should_ignore(path: str) -> bool:
    """Check if path should be ignored"""
    return any(ignore in path.lower() for ignore in IGNORE_PATTERNS)

def should_skip_contents(path: str) -> bool:
    """Check if directory contents should be skipped"""
    return os.path.basename(path) == 'chroma_db'

def generate_tree(
    start_path: str, 
    include_descriptions: bool = False,
    prefix: str = ''
) -> List[str]:
    """Generate directory tree structure"""
    tree = []
    start_path = os.path.abspath(start_path)
    
    try:
        items = sorted(os.listdir(start_path))
    except (PermissionError, FileNotFoundError) as e:
        print(f"Error accessing {start_path}: {str(e)}")
        return []

    # Filter items that shouldn't be ignored
    visible_items = [item for item in items if not should_ignore(item)]
    
    for i, item in enumerate(visible_items):
        path = os.path.join(start_path, item)
        
        if not (os.path.isfile(path) or os.path.isdir(path)):
            continue
            
        is_last = i == len(visible_items) - 1
        
        # Create branch prefix
        curr_prefix = '└── ' if is_last else '├── '
        next_prefix = '    ' if is_last else '│   '
        
        # Add item to tree
        tree_item = f"{prefix}{curr_prefix}{item}"
        if include_descriptions and item in FILE_DESCRIPTIONS:
            tree_item += f" - {FILE_DESCRIPTIONS[item]}"
        tree.append(tree_item)
        
        # Recurse into directories unless it's chroma_db
        if os.path.isdir(path) and not should_skip_contents(path):
            tree.extend(
                generate_tree(
                    path,
                    include_descriptions,
                    prefix + next_prefix
                )
            )
    
    return tree

def main():
    parser = argparse.ArgumentParser(description='Generate project directory tree')
    parser.add_argument('--path', default='.', help='Starting path')
    parser.add_argument('--desc', action='store_true', help='Include descriptions')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    tree = generate_tree(args.path, args.desc)
    
    if not tree:
        print("No files found or error occurred while generating tree.")
        return
        
    output = '\n'.join(['Project Structure:', ''] + tree)
    
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Tree structure saved to {args.output}")
        except Exception as e:
            print(f"Error saving to file: {str(e)}")
    else:
        print(output)

if __name__ == '__main__':
    main() 