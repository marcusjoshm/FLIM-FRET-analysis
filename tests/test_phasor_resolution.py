#!/usr/bin/env python3
"""
Test script for consistent phasor plot pixel resolution.

This script demonstrates how the new pixels_per_unit parameter
ensures consistent pixel sizes across different datasets.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from modules.phasor_plot_utils import (
    create_phasor_plot, 
    get_phasor_plot_resolution_info, 
    get_preset_resolutions
)


def generate_test_data(n_points=10000, noise_level=0.1):
    """
    Generate synthetic phasor data for testing.
    
    Args:
        n_points (int): Number of data points
        noise_level (float): Noise level for the data
        
    Returns:
        tuple: (g_data, s_data, intensity)
    """
    # Generate data with some structure
    np.random.seed(42)
    
    # Create multiple clusters
    centers = [(0.3, 0.2), (0.7, 0.4), (0.5, 0.1)]
    weights = [0.4, 0.3, 0.3]
    
    g_data = []
    s_data = []
    intensity = []
    
    for center, weight in zip(centers, weights):
        n_cluster = int(n_points * weight)
        
        # Generate cluster data
        g_cluster = np.random.normal(center[0], noise_level, n_cluster)
        s_cluster = np.random.normal(center[1], noise_level, n_cluster)
        
        # Ensure data is within phasor plot bounds
        g_cluster = np.clip(g_cluster, -0.005, 1.005)
        s_cluster = np.clip(s_cluster, 0, 0.7)
        
        # Generate intensity (higher for center of clusters)
        intensity_cluster = np.random.exponential(1.0, n_cluster)
        
        g_data.extend(g_cluster)
        s_data.extend(s_cluster)
        intensity.extend(intensity_cluster)
    
    return np.array(g_data), np.array(s_data), np.array(intensity)


def test_resolution_consistency():
    """
    Test that different datasets produce consistent pixel sizes.
    """
    print("=== Testing Phasor Plot Resolution Consistency ===\n")
    
    # Generate test data with different characteristics
    datasets = {
        'small_dataset': generate_test_data(5000, 0.05),
        'medium_dataset': generate_test_data(10000, 0.1),
        'large_dataset': generate_test_data(20000, 0.15),
        'noisy_dataset': generate_test_data(15000, 0.2)
    }
    
    # Test different resolution settings - much higher for better quality
    resolutions = [200, 400, 600]
    
    for resolution in resolutions:
        print(f"\n--- Testing Resolution: {resolution} pixels per unit ---")
        
        # Get resolution info
        info = get_phasor_plot_resolution_info(resolution)
        print(f"Target G bins: {info['target_g_bins']}, Target S bins: {info['target_s_bins']}")
        print(f"Target total bins: {info['target_total_bins']}")
        print(f"Target bin size G: {info['target_bin_size_g']:.6f}, Target bin size S: {info['target_bin_size_s']:.6f}")
        print(f"Target pixels per unit - G: {info['target_pixels_per_unit_g']:.1f}, S: {info['target_pixels_per_unit_s']:.1f}")
        print(f"Approach: {info['description']}")
        
        # Create plots for each dataset
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, (g_data, s_data, intensity)) in enumerate(datasets.items()):
            ax = axes[i]
            
            # Create phasor plot with consistent resolution
            title = f"{name.replace('_', ' ').title()}\n({len(g_data)} points)"
            create_phasor_plot(g_data, s_data, intensity, title, 
                             ax=ax, pixels_per_unit=resolution, show_colorbar=False)
            
            # Add dataset info
            ax.text(0.02, 0.98, f"Points: {len(g_data)}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'test_resolution_{resolution}.png', dpi=150, bbox_inches='tight')
        print(f"Saved test plot: test_resolution_{resolution}.png")
        plt.close()


def demonstrate_preset_resolutions():
    """
    Demonstrate the preset resolution configurations.
    """
    print("\n=== Preset Resolution Configurations ===\n")
    
    presets = get_preset_resolutions()
    
    for name, config in presets.items():
        print(f"{name.upper()} RESOLUTION:")
        print(f"  Pixels per unit: {config['pixels_per_unit']}")
        print(f"  Description: {config['description']}")
        print(f"  Total bins: {config['total_bins']:,}")
        
        # Get detailed info
        info = get_phasor_plot_resolution_info(config['target_pixels_per_unit'])
        print(f"  Target G bins: {info['target_g_bins']}, Target S bins: {info['target_s_bins']}")
        print(f"  Target bin size G: {info['target_bin_size_g']:.6f}, Target bin size S: {info['target_bin_size_s']:.6f}")
        print(f"  Target pixels per unit - G: {info['target_pixels_per_unit_g']:.1f}, S: {info['target_pixels_per_unit_s']:.1f}")
        print(f"  Approach: {info['description']}")
        print()


def test_performance_comparison():
    """
    Test performance with different resolutions.
    """
    print("\n=== Performance Comparison ===\n")
    
    # Generate a standard dataset
    g_data, s_data, intensity = generate_test_data(15000, 0.1)
    
    presets = get_preset_resolutions()
    
    import time
    
    for name, config in presets.items():
        print(f"Testing {name} resolution...")
        
        start_time = time.time()
        
        # Create plot
        fig, ax = create_phasor_plot(g_data, s_data, intensity, 
                                    f"Performance Test - {name.title()}", 
                                    target_pixels_per_unit=config['target_pixels_per_unit'])
        
        end_time = time.time()
        
        print(f"  Render time: {end_time - start_time:.3f} seconds")
        print(f"  Total bins: {config['total_bins']:,}")
        
        plt.close(fig)


if __name__ == "__main__":
    print("Phasor Plot Resolution Consistency Test")
    print("=" * 50)
    
    # Demonstrate preset resolutions
    demonstrate_preset_resolutions()
    
    # Test resolution consistency
    test_resolution_consistency()
    
    # Test performance
    test_performance_comparison()
    
    print("\nTest completed! Check the generated PNG files to see the consistency.") 