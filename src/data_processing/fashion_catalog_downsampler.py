"""
Fashion Catalog Downsampling Module
=================================

This module provides functionality for downsampling large fashion catalog JSON files
while preserving the output directory structure.

Features:
---------
- JSON file reading and writing
- Random downsampling of data entries
- Automatic output directory creation
"""
import json
from pathlib import Path
import random
from typing import Dict

def process_fashion_catalog(input_path: str, output_path: str, sample_fraction: float = 0.01) -> dict:
    """
    Process fashion catalog JSON file by counting entries and downsampling.
    
    Args:
        input_path: Path to raw JSON file
        output_path: Path to save processed JSON
        sample_fraction: Fraction of entries to keep (default 0.01 for 1/100th)
        
    Returns:
        dict with metadata about the processing
    """
    # Read input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get total count
    total_entries = len(data)
    
    # Calculate sample size
    sample_size = int(total_entries * sample_fraction)
    
    # Randomly sample entries
    sampled_data = random.sample(data, sample_size)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2)
    
    return {
        "total_entries": total_entries,
        "sample_size": sample_size,
        "sample_fraction": sample_fraction
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process and downsample fashion catalog JSON file')
    parser.add_argument('--input', '-i', default="data/raw/fashion_catalog.json",
                      help='Input JSON file path (default: data/raw/fashion_catalog.json)')
    parser.add_argument('--output', '-o', default="data/processed/fashion_catalog_sampled.json",
                      help='Output JSON file path (default: data/processed/fashion_catalog_sampled.json)')
    parser.add_argument('--fraction', '-f', type=float, default=0.01,
                      help='Fraction of entries to keep (default: 0.01)')
    
    args = parser.parse_args()
    
    results = process_fashion_catalog(args.input, args.output, args.fraction)
    print(f"Processed fashion catalog:")
    print(f"Total entries: {results['total_entries']}")
    print(f"Sampled entries: {results['sample_size']}")
    print(f"Sample fraction: {results['sample_fraction']}") 