"""
Search History Processing Module
===============================

This module provides functionality for processing and downsampling JSON search history data.
It includes utilities for random sampling of large datasets while preserving the output
directory structure.

Features:
---------
- JSON file reading and writing
- Random downsampling of data entries
- Automatic output directory creation
- Command-line interface for easy usage

Usage:
------
As a module:
    >>> from search_history_processor import process_search_history
    >>> results = process_search_history(
    ...     input_path='data/raw/search_history.json',
    ...     output_path='data/processed/sampled.json',
    ...     sample_fraction=0.01
    ... )

From command line:
    $ python search_history_processor.py -i input.json -o output.json -f 0.01

Command Line Arguments:
---------------------
--input, -i : str
    Path to input JSON file (default: data/raw/search_history.json)
--output, -o : str
    Path to output JSON file (default: data/processed/search_history_sampled.json)
--fraction, -f : float
    Fraction of entries to keep (default: 0.01)

Dependencies:
------------
- json: For JSON file handling
- random: For random sampling
- pathlib: For path manipulation
- argparse: For command-line argument parsing
"""

import json
import random
from pathlib import Path
import argparse

def process_search_history(input_path: str, output_path: str, sample_fraction: float = 0.01) -> dict:
    """
    Process search history JSON file by counting entries and downsampling.
    
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
    parser = argparse.ArgumentParser(description='Process and downsample search history JSON file')
    parser.add_argument('--input', '-i', default="data/raw/search_history.json",
                      help='Input JSON file path (default: data/raw/search_history.json)')
    parser.add_argument('--output', '-o', default="data/processed/search_history_sampled.json",
                      help='Output JSON file path (default: data/processed/search_history_sampled.json)')
    parser.add_argument('--fraction', '-f', type=float, default=0.01,
                      help='Fraction of entries to keep (default: 0.01)')
    
    args = parser.parse_args()
    
    results = process_search_history(args.input, args.output, args.fraction)
    print(f"Processed search history:")
    print(f"Total entries: {results['total_entries']}")
    print(f"Sampled entries: {results['sample_size']}")
    print(f"Sample fraction: {results['sample_fraction']}") 