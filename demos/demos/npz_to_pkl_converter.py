#!/usr/bin/env python3
"""
NPZ to PKL Converter for SERL Demonstrations

This script converts demonstration data from NPZ format (as recorded by the 
scripted controller) to PKL format required by SERL training scripts.

Usage:
    python npz_to_pkl_converter.py input.npz output.pkl
    python npz_to_pkl_converter.py input.npz output.pkl --verbose
    python npz_to_pkl_converter.py input.npz output.pkl --filter-successful
    python npz_to_pkl_converter.py --batch input_dir/ output_dir/
"""

import os
import sys
import glob
import argparse
import numpy as np
import pickle as pkl
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


def convert_npz_to_pkl(npz_path: str, pkl_path: str, verbose: bool = False) -> Tuple[int, int]:
    """
    Convert NPZ demonstration data to PKL format for SERL.
    
    Args:
        npz_path: Path to input NPZ file
        pkl_path: Path to output PKL file
        verbose: Whether to print detailed information
    
    Returns:
        Tuple of (num_episodes, num_transitions)
    """
    # Load NPZ file
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract data arrays - these are arrays of episodes
    actions_episodes = data['acs']
    observations_episodes = data['obs']
    rewards_episodes = data['rewards']
    dones_episodes = data['dones']
    
    # Optional fields
    terminateds_episodes = data.get('terminateds', dones_episodes)
    truncateds_episodes = data.get('truncateds', dones_episodes)
    
    # Store all transitions across all episodes
    all_transitions = []
    
    # Process each episode
    num_episodes = len(actions_episodes)
    
    for ep_idx in range(num_episodes):
        # Get data for this episode
        ep_actions = actions_episodes[ep_idx]
        ep_observations = observations_episodes[ep_idx]
        ep_rewards = rewards_episodes[ep_idx]
        ep_dones = dones_episodes[ep_idx]
        
        # Convert episode to transitions
        # Note: observations has one extra element at the beginning (initial observation)
        episode_length = len(ep_actions)
        
        if verbose and ep_idx == 0:
            print(f"Episode {ep_idx} structure:")
            print(f"  - Observations shape: {len(ep_observations)} x {ep_observations[0].shape if len(ep_observations) > 0 else 'N/A'}")
            print(f"  - Actions shape: {len(ep_actions)} x {ep_actions[0].shape if len(ep_actions) > 0 else 'N/A'}")
            print(f"  - Episode length: {episode_length}")
        
        for t in range(episode_length):
            transition = {
                'observations': np.array(ep_observations[t]),
                'actions': np.array(ep_actions[t]),
                'next_observations': np.array(ep_observations[t + 1]),
                'rewards': float(ep_rewards[t]),
                'masks': 1.0 - float(ep_dones[t]),
                'dones': bool(ep_dones[t])
            }
            all_transitions.append(transition)
    
    # Save as pickle file
    with open(pkl_path, 'wb') as f:
        pkl.dump(all_transitions, f)
    
    # Close NPZ file
    data.close()
    
    return num_episodes, len(all_transitions)


def convert_npz_to_pkl_filtered(
    npz_path: str, 
    pkl_path: str, 
    filter_successful_only: bool = False,
    verbose: bool = False
) -> Tuple[int, int, int]:
    """
    Convert NPZ demonstration data to PKL format with optional filtering.
    
    Args:
        npz_path: Path to input NPZ file
        pkl_path: Path to output PKL file
        filter_successful_only: If True, only include episodes that end with positive reward
        verbose: Whether to print detailed information
    
    Returns:
        Tuple of (total_episodes, successful_episodes, num_transitions)
    """
    if verbose:
        print(f"Loading NPZ file: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Print available keys
    if verbose:
        print(f"Available keys in NPZ: {list(data.keys())}")
    
    # Extract data
    actions_episodes = data['acs']
    observations_episodes = data['obs']
    rewards_episodes = data['rewards']
    dones_episodes = data['dones']
    
    # Get metadata if available
    transition_ctr = data.get('transition_ctr', None)
    num_demos = data.get('num_demos', len(actions_episodes))
    
    if verbose:
        print(f"Number of episodes in file: {len(actions_episodes)}")
        if transition_ctr is not None:
            print(f"Total transitions recorded: {transition_ctr}")
    
    all_transitions = []
    successful_episodes = 0
    total_episodes = len(actions_episodes)
    
    for ep_idx in range(total_episodes):
        ep_actions = actions_episodes[ep_idx]
        ep_observations = observations_episodes[ep_idx]
        ep_rewards = rewards_episodes[ep_idx]
        ep_dones = dones_episodes[ep_idx]
        
        # Check if episode was successful (positive final reward)
        episode_success = ep_rewards[-1] > 0 if len(ep_rewards) > 0 else False
        
        if filter_successful_only and not episode_success:
            if verbose:
                print(f"Skipping episode {ep_idx} (unsuccessful, final reward: {ep_rewards[-1] if len(ep_rewards) > 0 else 'N/A'})")
            continue
            
        if episode_success:
            successful_episodes += 1
        
        # Validate data consistency
        try:
            assert len(ep_observations) == len(ep_actions) + 1, \
                f"Episode {ep_idx}: observations ({len(ep_observations)}) should be actions ({len(ep_actions)}) + 1"
            assert len(ep_actions) == len(ep_rewards), \
                f"Episode {ep_idx}: actions ({len(ep_actions)}) and rewards ({len(ep_rewards)}) length mismatch"
        except AssertionError as e:
            print(f"Warning: {e}")
            print(f"Skipping episode {ep_idx} due to data inconsistency")
            continue
        
        # Convert to transitions
        for t in range(len(ep_actions)):
            transition = {
                'observations': np.array(ep_observations[t]),
                'actions': np.array(ep_actions[t]),
                'next_observations': np.array(ep_observations[t + 1]),
                'rewards': float(ep_rewards[t]),
                'masks': 1.0 - float(ep_dones[t]),
                'dones': bool(ep_dones[t])
            }
            all_transitions.append(transition)
    
    # Save transitions
    with open(pkl_path, 'wb') as f:
        pkl.dump(all_transitions, f)
    
    data.close()
    
    return total_episodes, successful_episodes, len(all_transitions)


def batch_convert_npz_to_pkl(
    npz_dir: str, 
    pkl_dir: str, 
    pattern: str = "*.npz",
    filter_successful_only: bool = False,
    verbose: bool = False
) -> None:
    """
    Convert all NPZ files in a directory to PKL format.
    
    Args:
        npz_dir: Directory containing NPZ files
        pkl_dir: Directory to save PKL files
        pattern: File pattern to match (default: "*.npz")
        filter_successful_only: If True, only include successful episodes
        verbose: Whether to print detailed information
    """
    # Create output directory if it doesn't exist
    os.makedirs(pkl_dir, exist_ok=True)
    
    # Find all NPZ files
    npz_files = glob.glob(os.path.join(npz_dir, pattern))
    print(f"Found {len(npz_files)} NPZ files to convert")
    
    if len(npz_files) == 0:
        print(f"No files matching pattern '{pattern}' found in {npz_dir}")
        return
    
    total_stats = {
        'files': 0,
        'episodes': 0,
        'successful_episodes': 0,
        'transitions': 0,
        'errors': 0
    }
    
    for npz_path in npz_files:
        # Generate output filename
        base_name = os.path.basename(npz_path)
        pkl_name = base_name.replace('.npz', '.pkl')
        pkl_path = os.path.join(pkl_dir, pkl_name)
        
        print(f"\nConverting: {base_name}")
        try:
            if filter_successful_only or verbose:
                total_eps, success_eps, num_trans = convert_npz_to_pkl_filtered(
                    npz_path, pkl_path, filter_successful_only, verbose
                )
                total_stats['episodes'] += total_eps
                total_stats['successful_episodes'] += success_eps
                total_stats['transitions'] += num_trans
            else:
                num_eps, num_trans = convert_npz_to_pkl(npz_path, pkl_path, verbose)
                total_stats['episodes'] += num_eps
                total_stats['transitions'] += num_trans
            
            total_stats['files'] += 1
            print(f"✓ Successfully converted to: {pkl_name}")
            
        except Exception as e:
            print(f"✗ Error converting {npz_path}: {e}")
            total_stats['errors'] += 1
            continue
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Batch Conversion Summary:")
    print(f"{'='*50}")
    print(f"Files processed: {total_stats['files']}/{len(npz_files)}")
    print(f"Total episodes: {total_stats['episodes']}")
    if filter_successful_only or verbose:
        print(f"Successful episodes: {total_stats['successful_episodes']}")
    print(f"Total transitions: {total_stats['transitions']}")
    if total_stats['errors'] > 0:
        print(f"Errors: {total_stats['errors']}")
    print(f"{'='*50}")


def validate_pkl_file(pkl_path: str, num_samples: int = 5) -> None:
    """
    Validate and display information about a PKL file.
    
    Args:
        pkl_path: Path to PKL file to validate
        num_samples: Number of sample transitions to display
    """
    print(f"\nValidating PKL file: {pkl_path}")
    print(f"{'='*50}")
    
    try:
        with open(pkl_path, 'rb') as f:
            transitions = pkl.load(f)
        
        print(f"Total transitions: {len(transitions)}")
        
        if len(transitions) > 0:
            # Check first transition
            first_trans = transitions[0]
            print(f"\nTransition keys: {list(first_trans.keys())}")
            print(f"\nData shapes:")
            for key, value in first_trans.items():
                if isinstance(value, np.ndarray):
                    print(f"  - {key}: {value.shape}")
                else:
                    print(f"  - {key}: {type(value).__name__}")
            
            # Show sample transitions
            print(f"\nFirst {min(num_samples, len(transitions))} transitions:")
            for i in range(min(num_samples, len(transitions))):
                trans = transitions[i]
                print(f"\nTransition {i}:")
                print(f"  - observations: {trans['observations'][:3]}... (showing first 3)")
                print(f"  - actions: {trans['actions']}")
                print(f"  - rewards: {trans['rewards']:.4f}")
                print(f"  - masks: {trans['masks']}")
                print(f"  - dones: {trans['dones']}")
        
        print(f"\n✓ PKL file is valid and ready for use!")
        
    except Exception as e:
        print(f"✗ Error validating PKL file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert NPZ demonstration files to PKL format for SERL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python npz_to_pkl_converter.py demo.npz demo.pkl
  
  # Convert with verbose output
  python npz_to_pkl_converter.py demo.npz demo.pkl --verbose
  
  # Convert only successful episodes
  python npz_to_pkl_converter.py demo.npz demo.pkl --filter-successful
  
  # Batch convert directory
  python npz_to_pkl_converter.py --batch demos_npz/ demos_pkl/
  
  # Validate converted file
  python npz_to_pkl_converter.py --validate demo.pkl
        """
    )
    
    # Positional arguments for single file conversion
    parser.add_argument('input', nargs='?', help='Input NPZ file path')
    parser.add_argument('output', nargs='?', help='Output PKL file path')
    
    # Optional arguments
    parser.add_argument('--batch', action='store_true', 
                       help='Batch convert mode (input and output become directories)')
    parser.add_argument('--pattern', default='*.npz', 
                       help='File pattern for batch mode (default: *.npz)')
    parser.add_argument('--filter-successful', action='store_true',
                       help='Only include episodes that end with positive reward')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information during conversion')
    parser.add_argument('--validate', type=str, metavar='PKL_FILE',
                       help='Validate a PKL file and show its contents')
    
    args = parser.parse_args()
    
    # Handle validation mode
    if args.validate:
        validate_pkl_file(args.validate)
        return
    
    # Check if we have required arguments
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    # Handle batch mode
    if args.batch:
        if not args.output:
            print("Error: Output directory required for batch mode")
            sys.exit(1)
        
        batch_convert_npz_to_pkl(
            args.input, 
            args.output, 
            args.pattern,
            args.filter_successful,
            args.verbose
        )
    
    # Handle single file mode
    else:
        if not args.output:
            # Auto-generate output filename
            args.output = args.input.replace('.npz', '.pkl')
            print(f"No output file specified, using: {args.output}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Converting: {args.input} -> {args.output}")
        
        try:
            if args.filter_successful or args.verbose:
                total_eps, success_eps, num_trans = convert_npz_to_pkl_filtered(
                    args.input, 
                    args.output, 
                    args.filter_successful,
                    args.verbose
                )
                print(f"\nConversion complete!")
                print(f"Total episodes: {total_eps}")
                print(f"Successful episodes: {success_eps}")
                print(f"Total transitions: {num_trans}")
            else:
                num_eps, num_trans = convert_npz_to_pkl(
                    args.input, 
                    args.output,
                    args.verbose
                )
                print(f"\nConversion complete!")
                print(f"Episodes: {num_eps}")
                print(f"Transitions: {num_trans}")
            
            print(f"Output saved to: {args.output}")
            
        except Exception as e:
            print(f"Error during conversion: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
