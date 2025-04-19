#!/usr/bin/env python3
"""
Test script for configuration switching
"""

import os
import time
import subprocess

def test_switch():
    """Test switching between different configurations."""
    print("Testing configuration switching...")
    
    # Test each level
    for level in ['low', 'medium', 'high']:
        print(f"\nSwitching to {level} interaction level...")
        result = subprocess.run(
            ['/opt/cowrie/adaptive-honeypot/switch_config.py', level],
            capture_output=True,
            text=True
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        
        # Check if the symlink points to the correct file
        config_link = '/opt/cowrie/cowrie.cfg'  # Updated path
        target = os.path.realpath(config_link)
        expected = f'/opt/cowrie/profiles/cowrie-{level}.cfg'  # Updated path
        
        print(f"Current config: {target}")
        print(f"Expected: {expected}")
        print(f"Match: {target == expected}")
        
        # Wait before next switch
        time.sleep(5)
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_switch()
