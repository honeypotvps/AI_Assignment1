#!/usr/bin/env python3
"""
Configuration Switcher for Adaptive Honeypot
This script switches between different Cowrie configuration profiles.
"""

import os
import sys
import subprocess
import logging
import argparse
import signal
import time

# Set up logging
logging.basicConfig(
    filename='/opt/cowrie/adaptive-honeypot/config_switcher.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define paths
COWRIE_DIR = "/opt/cowrie"
PROFILES_DIR = os.path.join(COWRIE_DIR, "profiles")  # Updated path
CONFIG_LINK = os.path.join(COWRIE_DIR, "cowrie.cfg")  # Updated path
COWRIE_PID_FILE = os.path.join(COWRIE_DIR, "var/run/cowrie.pid")

def get_cowrie_pid():
    """Get the PID of the running Cowrie process."""
    try:
        with open(COWRIE_PID_FILE, 'r') as f:
            return int(f.read().strip())
    except:
        return None

def reload_cowrie():
    """Reload Cowrie configuration by sending SIGHUP."""
    pid = get_cowrie_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGHUP)
            logging.info(f"Sent SIGHUP to Cowrie process (PID: {pid})")
            return True
        except Exception as e:
            logging.error(f"Failed to send SIGHUP: {str(e)}")
            return False
    else:
        logging.error("Cowrie PID not found")
        return False

def restart_cowrie():
    """Restart Cowrie service."""
    try:
        logging.info("Stopping Cowrie...")
        # Use the Cowrie script directly with shell=True to ensure proper environment
        stop_cmd = f"cd {COWRIE_DIR} && ./bin/cowrie stop"
        subprocess.run(stop_cmd, shell=True, check=True)
        time.sleep(2)  # Give it time to stop
        
        logging.info("Starting Cowrie...")
        start_cmd = f"cd {COWRIE_DIR} && ./bin/cowrie start"
        subprocess.run(start_cmd, shell=True, check=True)
        
        logging.info("Cowrie restarted successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to restart Cowrie: {str(e)}")
        return False


def switch_config(level):
    """Switch to the specified interaction level configuration."""
    # Validate level
    if level not in ['low', 'medium', 'high']:
        logging.error(f"Invalid interaction level: {level}")
        return False
    
    # Check if profile exists
    profile_path = os.path.join(PROFILES_DIR, f"cowrie-{level}.cfg")
    if not os.path.exists(profile_path):
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
            logging.info(f"Backed up original config to {backup_path}")
    
    # Create new symlink
    os.symlink(profile_path, CONFIG_LINK)
    logging.info(f"Switched to {level} interaction profile")
    
    # Reload or restart Cowrie
    if reload_cowrie():
        logging.info("Cowrie configuration reloaded")
    else:
        logging.warning("Failed to reload, attempting restart")
        if restart_cowrie():
            logging.info("Cowrie restarted with new configuration")
        else:
            logging.error("Failed to apply new configuration")
            return False
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Switch Cowrie interaction level')
    parser.add_argument('level', choices=['low', 'medium', 'high'],
                        help='Interaction level to switch to')
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    if switch_config(args.level):
        print(f"Successfully switched to {args.level} interaction level")
        sys.exit(0)
    else:
        print(f"Failed to switch to {args.level} interaction level")
        sys.exit(1)

if __name__ == "__main__":
    main()
