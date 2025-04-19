#!/usr/bin/env python3
"""
Simplified Adaptive Honeypot Integration Script
This script demonstrates configuration switching without model loading.
"""

import os
import time
import subprocess
import logging
import random
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='/opt/cowrie/adaptive-honeypot/simple_integration.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define paths
COWRIE_DIR = "/opt/cowrie"
PROFILES_DIR = os.path.join(COWRIE_DIR, "profiles")
CONFIG_LINK = os.path.join(COWRIE_DIR, "cowrie.cfg")
PROCESSED_DIR = "/opt/cowrie/adaptive-honeypot/processed_data"

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

def switch_config(level):
    """Switch to the specified interaction level configuration."""
    print(f"Switching to {level} interaction level...")
    logging.info(f"Switching to {level} interaction level")
    
    # Check if profile exists
    profile_path = os.path.join(PROFILES_DIR, f"cowrie-{level}.cfg")
    if not os.path.exists(profile_path):
        print(f"Profile not found: {profile_path}")
        logging.error(f"Profile not found: {profile_path}")
        return False
    
    # Remove existing symlink if it exists
    if os.path.exists(CONFIG_LINK):
        if os.path.islink(CONFIG_LINK):
            os.unlink(CONFIG_LINK)
        else:
            # Backup the original config if it's not a symlink
            backup_path = f"{CONFIG_LINK}.backup.{int(time.time())}"
            os.rename(CONFIG_LINK, backup_path)
            print(f"Backed up original config to {backup_path}")
            logging.info(f"Backed up original config to {backup_path}")
    
    # Create new symlink
    os.symlink(profile_path, CONFIG_LINK)
    print(f"Switched to {level} interaction profile")
    logging.info(f"Switched to {level} interaction profile")
    
    # Prompt user to restart Cowrie manually
    print("\nPlease restart Cowrie manually with:")
    print(f"cd {COWRIE_DIR} && ./bin/cowrie restart")
    
    return True

def simulate_threat_detection():
    """Simulate threat detection and return a threat level."""
    # In a real implementation, this would use ML models
    # Here we just randomly select a threat level for demonstration
    threat_level = random.choice([0, 1, 2])  # 0=low, 1=medium, 2=high
    
    response_strategies = {
        0: "low",
        1: "medium",
        2: "high"
    }
    
    strategy = response_strategies.get(threat_level, "low")
    print(f"Detected threat level: {threat_level} ({strategy})")
    logging.info(f"Detected threat level: {threat_level} ({strategy})")
    
    return threat_level

def main():
    """Main function."""
    print("Starting Simplified Adaptive Honeypot Integration")
    logging.info("Starting Simplified Adaptive Honeypot Integration")
    
    try:
        while True:
            # Simulate threat detection
            print("\n--- New detection cycle ---")
            threat_level = simulate_threat_detection()
            
            # Map threat level to configuration
            config_level = "low"
            if threat_level == 1:
                config_level = "medium"
            elif threat_level == 2:
                config_level = "high"
            
            # Switch configuration
            switch_config(config_level)
            
            # Wait before next cycle
            wait_time = 30  # seconds
            print(f"\nWaiting {wait_time} seconds before next detection cycle...")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("\nExiting...")
        logging.info("Script terminated by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        logging.error(f"Error in main loop: {str(e)}")

if __name__ == "__main__":
    main()
