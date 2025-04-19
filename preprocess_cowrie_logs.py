#!/usr/bin/env python3
"""
Preprocessing Script for Cowrie Honeypot Logs (Part 1)
This script processes raw Cowrie honeypot logs and extracts relevant features.
"""

import os
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
sns.set(style="darkgrid")

# Define directories
# LOG_DIR = "/opt/cowrie/var/log/cowrie"

LOG_DIRS = [
    "/opt/cowrie/var/log/cowrie",
    "/opt/cowrie/adaptive-honeypot/training_dataset"
]

OUTPUT_DIR = "/opt/cowrie/adaptive-honeypot/processed_data"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json_logs(file_path):
    """Load JSON logs from a file."""
    logs = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                logs.append(log)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line}")
    return logs

def extract_session_data(logs):
    """Extract session data from logs."""
    sessions = {}
    
    for log in logs:
        session_id = log.get('session', '')
        if not session_id:
            continue
        
        # Initialize session if not exists
        if session_id not in sessions:
            sessions[session_id] = {
                'src_ip': log.get('src_ip', ''),
                'start_time': log.get('timestamp', ''),
                'end_time': None,
                'duration': 0,
                'events': [],
                'commands': [],
                'login_attempts': [],
                'login_success': False,
                'username': None,
                'password': None,
                'client_version': log.get('version', ''),
                'hassh': log.get('hassh', ''),
                'hassh_algorithms': log.get('hasshAlgorithms', '')
            }
        
        # Update session based on event type
        event_id = log.get('eventid', '')
        
        # Store all events
        sessions[session_id]['events'].append({
            'timestamp': log.get('timestamp', ''),
            'eventid': event_id,
            'data': log
        })
        
        # Process specific event types
        if event_id == 'cowrie.login.success':
            sessions[session_id]['login_success'] = True
            sessions[session_id]['username'] = log.get('username', '')
            sessions[session_id]['password'] = log.get('password', '')
            sessions[session_id]['login_attempts'].append({
                'timestamp': log.get('timestamp', ''),
                'username': log.get('username', ''),
                'password': log.get('password', ''),
                'success': True
            })
            
        elif event_id == 'cowrie.login.failed':
            sessions[session_id]['login_attempts'].append({
                'timestamp': log.get('timestamp', ''),
                'username': log.get('username', ''),
                'password': log.get('password', ''),
                'success': False
            })
            
        elif event_id == 'cowrie.command.input':
            sessions[session_id]['commands'].append({
                'timestamp': log.get('timestamp', ''),
                'input': log.get('input', '')
            })
            
        elif event_id == 'cowrie.session.closed':
            sessions[session_id]['end_time'] = log.get('timestamp', '')
            sessions[session_id]['duration'] = log.get('duration', 0)
    
    return sessions

def extract_event_data(logs):
    """Extract event data from logs."""
    events = []
    
    for log in logs:
        if 'session' in log and 'eventid' in log:
            event = {
                'session_id': log.get('session', ''),
                'timestamp': log.get('timestamp', ''),
                'event_type': log.get('eventid', ''),
                'src_ip': log.get('src_ip', ''),
                'input': log.get('input', '') if 'input' in log else '',
                'success': 1 if log.get('eventid', '') == 'cowrie.login.success' else 0
            }
            events.append(event)
    
    return events


def extract_features(sessions):
    """Extract features from session data."""
    session_features = []
    
    for session_id, session in sessions.items():
        # Basic session info
        features = {
            'session_id': session_id,
            'src_ip': session['src_ip'],
            'start_time': session['start_time'],
            'end_time': session['end_time'],
            'duration': session['duration'],
            'login_success': session['login_success'],
            'username': session['username'],
            'password': session['password'],
            'client_version': session['client_version'],
            'hassh': session['hassh'],
            'login_attempt_count': len(session['login_attempts']),
            'command_count': len(session['commands']),
            'event_count': len(session['events'])
        }
        
        # Extract commands as a single string for analysis
        commands = ' | '.join([cmd['input'] for cmd in session['commands']])
        features['commands'] = commands
        
        # Command type features
        features['has_download_command'] = int(bool(re.search(r'wget|curl|tftp|scp|fetch', commands)))
        features['has_privilege_escalation'] = int(bool(re.search(r'sudo|su\s|chmod|chown', commands)))
        features['has_file_access'] = int(bool(re.search(r'cat|less|more|head|tail|nano|vi|vim', commands)))
        features['has_reconnaissance'] = int(bool(re.search(r'ls|dir|pwd|whoami|id|uname|ifconfig|ip\s', commands)))
        features['has_persistence'] = int(bool(re.search(r'crontab|systemctl|service|rc\.local|init\.d', commands)))
        features['has_malware_indicators'] = int(bool(re.search(r'\.sh|\.pl|\.py|\.bin|\.elf|chmod\s+[0-7]*x', commands)))
        
        # Calculate threat score based on features
        threat_score = 0
        if features['login_success']:
            threat_score += 5
        if features['has_download_command']:
            threat_score += 3
        if features['has_privilege_escalation']:
            threat_score += 4
        if features['has_file_access']:
            threat_score += 1
        if features['has_reconnaissance']:
            threat_score += 2
        if features['has_persistence']:
            threat_score += 4
        if features['has_malware_indicators']:
            threat_score += 3
        
        features['threat_score'] = threat_score
        
        # Determine threat level
        if threat_score >= 10:
            features['threat_level'] = 'high'
        elif threat_score >= 5:
            features['threat_level'] = 'medium'
        else:
            features['threat_level'] = 'low'
        
        session_features.append(features)
    
    return session_features
def extract_events(logs):
    """Extract individual events from logs."""
    events = []
    
    for log in logs:
        event_id = log.get('eventid', '')
        session_id = log.get('session', '')
        
        if not event_id or not session_id:
            continue
        
        event = {
            'timestamp': log.get('timestamp', ''),
            'session_id': session_id,
            'event_id': event_id,
            'src_ip': log.get('src_ip', ''),
            'username': log.get('username', ''),
            'password': log.get('password', ''),
            'input': log.get('input', ''),
            'duration': log.get('duration', 0),
            'success': log.get('success', False)
        }
        
        events.append(event)
    
    return events

def process_logs(log_files=None):
    """Process Cowrie logs and extract sessions and events."""
    # If log_files is not provided, find them in the default LOG_DIR
    if log_files is None:
        log_files = []
        for root, dirs, files in os.walk(LOG_DIR):
            for file in files:
                if file.startswith('cowrie.json'):
                    log_files.append(os.path.join(root, file))
    
    # Initialize data structures
    all_sessions = {}
    all_events = []
    
    # Process each log file
    for log_file in tqdm(log_files, desc="Processing log files"):
        try:
            # Load logs from file
            logs = load_json_logs(log_file)
            
            # Extract sessions
            sessions = extract_session_data(logs)
            all_sessions.update(sessions)
            
            # Extract events
            events = extract_event_data(logs)
            all_events.extend(events)
        except Exception as e:
            print(f"Error processing {log_file}: {str(e)}")
    
    # Convert to dataframes
    if all_sessions:
        sessions_df = pd.DataFrame(list(all_sessions.values()))
    else:
        # Create empty dataframe with required columns
        sessions_df = pd.DataFrame(columns=[
            'session_id', 'src_ip', 'start_time', 'end_time', 'duration',
            'commands', 'command_count', 'unique_commands', 'login_attempts',
            'login_success', 'download_attempts', 'threat_level'
        ])
    
    if all_events:
        events_df = pd.DataFrame(all_events)
    else:
        # Create empty dataframe with required columns
        events_df = pd.DataFrame(columns=[
            'session_id', 'timestamp', 'event_type', 'src_ip', 'input', 'success'
        ])
    
    # Ensure all required columns exist
    required_columns = [
        'command_count', 'login_success', 'download_attempts', 'threat_level'
    ]
    for col in required_columns:
        if col not in sessions_df.columns:
            sessions_df[col] = 0
    
    # Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sessions_df.to_csv(os.path.join(OUTPUT_DIR, 'cowrie_sessions.csv'), index=False)
    events_df.to_csv(os.path.join(OUTPUT_DIR, 'cowrie_events.csv'), index=False)
    
    return sessions_df, events_df


def generate_basic_statistics(sessions_df, events_df):
    """Generate basic statistics from processed data."""
    stats = {}
    
    # Session statistics
    stats['total_sessions'] = len(sessions_df)
    stats['total_events'] = len(events_df)
    
    # Check if columns exist before using them
    if 'login_success' in sessions_df.columns:
        stats['successful_logins'] = sessions_df['login_success'].sum()
    else:
        stats['successful_logins'] = 0
    
    if 'command_count' in sessions_df.columns:
        stats['sessions_with_commands'] = (sessions_df['command_count'] > 0).sum()
    else:
        stats['sessions_with_commands'] = 0
    
    if 'threat_level' in sessions_df.columns:
        stats['threat_levels'] = {
            'low': (sessions_df['threat_level'] == 0).sum(),
            'medium': (sessions_df['threat_level'] == 1).sum(),
            'high': (sessions_df['threat_level'] == 2).sum()
        }
    else:
        stats['threat_levels'] = {'low': 0, 'medium': 0, 'high': 0}
    
    return stats


def main():
    """Main function to process Cowrie logs."""
    print("Starting Cowrie log preprocessing...")
    
    # Define multiple log directories
    LOG_DIRS = [
        "/opt/cowrie/var/log/cowrie",
        "/opt/cowrie/adaptive-honeypot/training_dataset"
    ]
    
    # Get all log files from all directories
    log_files = []
    for log_dir in LOG_DIRS:
        print(f"Searching for log files in {log_dir}...")
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith('cowrie.json'):
                    log_files.append(os.path.join(root, file))
    
    if not log_files:
        print(f"No Cowrie log files found in the specified directories")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # Process logs with the combined log files
    sessions_df, events_df = process_logs(log_files)
    
    # Generate basic statistics
    stats = generate_basic_statistics(sessions_df, events_df)
    
    print("Preprocessing complete!")
    print(f"Processed {stats['total_sessions']} sessions with {stats['total_events']} events")
    print(f"Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

