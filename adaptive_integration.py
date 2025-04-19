#!/usr/bin/env python3
"""
Adaptive Honeypot Integration Script
This script monitors Cowrie logs and applies adaptive responses.
"""
import sys
import os
import time
import json
import pickle
import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    filename='/opt/cowrie/adaptive-honeypot/adaptive_honeypot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define paths
COWRIE_LOG_DIR = "/opt/cowrie/var/log/cowrie"
MODEL_DIR = "/opt/cowrie/adaptive-honeypot/ml_models/"
PROCESSED_DIR = "/opt/cowrie/adaptive-honeypot/processed_data"
CONFIG_DIR = "/opt/cowrie"

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_models():
    """Load the trained models."""
    logging.info("Loading trained models...")
    
    models = {}
    
    try:
        # Load threat classifier
        with open(os.path.join(MODEL_DIR, 'best_threat_classifier.pkl'), 'rb') as f:
            models['threat_classifier'] = pickle.load(f)
        logging.info("Loaded threat classifier")
        
        # Load clustering model
        with open(os.path.join(MODEL_DIR, 'kmeans_clustering.pkl'), 'rb') as f:
            kmeans, preprocessor = pickle.load(f)
            models['kmeans'] = kmeans
            models['preprocessor'] = preprocessor
        logging.info("Loaded clustering model")
        
        # Load adaptive response mapping
        with open(os.path.join(MODEL_DIR, 'adaptive_response_mapping.json'), 'r') as f:
            models['response_mapping'] = json.load(f)
        logging.info("Loaded adaptive response mapping")
        
        return models
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        return None

def extract_features(session_data):
    """Extract features from session data."""
    # Basic session info
    features = {
        'duration': session_data.get('duration', 0),
        'login_success': int(session_data.get('login_success', False)),
        'command_count': len(session_data.get('commands', [])),
    }
    
    # Extract commands as a single string for analysis
    commands = ' | '.join([cmd.get('input', '') for cmd in session_data.get('commands', [])])
    
    # Command type features
    features['has_download_command'] = int(bool(re.search(r'wget|curl|tftp|scp|fetch', commands)))
    features['has_privilege_escalation'] = int(bool(re.search(r'sudo|su\s|chmod|chown', commands)))
    features['has_file_access'] = int(bool(re.search(r'cat|less|more|head|tail|nano|vi|vim', commands)))
    features['has_reconnaissance'] = int(bool(re.search(r'ls|dir|pwd|whoami|id|uname|ifconfig|ip\s', commands)))
    features['has_persistence'] = int(bool(re.search(r'crontab|systemctl|service|rc\.local|init\.d', commands)))
    features['has_malware_indicators'] = int(bool(re.search(r'\.sh|\.pl|\.py|\.bin|\.elf|chmod\s+[0-7]*x', commands)))
    
    return features

def predict_threat_level(features, models):
    """Predict threat level using the trained classifier."""
    try:
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        threat_level = models['threat_classifier'].predict(feature_df)[0]
        
        return int(threat_level)
    except Exception as e:
        logging.error(f"Error predicting threat level: {str(e)}")
        return 0  # Default to low threat

def predict_attack_pattern(features, models):
    """Predict attack pattern cluster using the trained clustering model."""
    try:
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Preprocess features
        X_processed = models['preprocessor'].transform(feature_df)
        
        # Predict cluster
        cluster = models['kmeans'].predict(X_processed)[0]
        
        return int(cluster)
    except Exception as e:
        logging.error(f"Error predicting attack pattern: {str(e)}")
        return 0  # Default to cluster 0

def get_adaptive_response(threat_level, cluster, models):
    """Determine the appropriate response level."""
    try:
        key = f"({threat_level}, {cluster})"
        return models['response_mapping'].get(key, 0)  # Default to low interaction
    except Exception as e:
        logging.error(f"Error getting adaptive response: {str(e)}")
        return 0  # Default to low interaction





def apply_response_config(session_id, response_level):
    """Apply the response configuration to Cowrie for this session."""
    # Response strategies
    response_strategies = {
        0: "low",
        1: "medium",
        2: "high"
    }
    
    strategy = response_strategies.get(response_level, "low")
    
    logging.info(f"Session {session_id}: Applying {strategy} interaction strategy")
    
    # Log the decision
    with open(os.path.join(PROCESSED_DIR, "adaptive_decisions.log"), "a") as f:
        f.write(f"{datetime.now()} - Session {session_id}: Applied {strategy} interaction strategy\n")
    
    # Store the current session's response level
    with open(os.path.join(PROCESSED_DIR, "active_sessions.json"), "a+") as f:
        try:
            f.seek(0)
            content = f.read().strip()
            if content:
                sessions = json.loads(content)
            else:
                sessions = {}
            
            sessions[session_id] = {
                "response_level": response_level,
                "strategy": strategy,
                "timestamp": datetime.now().isoformat()
            }
            
            f.seek(0)
            f.truncate()
            json.dump(sessions, f, indent=4)
        except Exception as e:
            logging.error(f"Error updating active sessions: {str(e)}")
    
    # Determine the highest required interaction level among active sessions
    highest_level = 0
    try:
        for session_info in sessions.values():
            highest_level = max(highest_level, session_info.get("response_level", 0))
    except:
        highest_level = response_level
    
    # Convert to strategy name
    highest_strategy = response_strategies.get(highest_level, "low")
    
    # Switch Cowrie configuration if needed
    try:
        # Check current configuration
        current_config = os.path.realpath("/opt/cowrie/cowrie.cfg")  # Updated path
        expected_config = os.path.join("/opt/cowrie/profiles", f"cowrie-{highest_strategy}.cfg")  # Updated path
        
        if current_config != expected_config:
            logging.info(f"Switching to {highest_strategy} interaction configuration")
            
            # Call the configuration switcher script
            result = subprocess.run(
                [os.path.join(os.path.dirname(__file__), "switch_config.py"), highest_strategy],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logging.info(f"Successfully switched to {highest_strategy} interaction level")
            else:
                logging.error(f"Failed to switch configuration: {result.stderr}")
    except Exception as e:
        logging.error(f"Error switching configuration: {str(e)}")
    
    return response_level




def process_session(session_id, session_data, models):
    """Process a session and determine the appropriate response."""
    logging.info(f"Processing session {session_id}")
    
    # Extract features
    features = extract_features(session_data)
    
    # Predict threat level
    threat_level = predict_threat_level(features, models)
    logging.info(f"Session {session_id}: Threat level {threat_level}")
    
    # Predict attack pattern
    cluster = predict_attack_pattern(features, models)
    logging.info(f"Session {session_id}: Attack pattern cluster {cluster}")
    
    # Get adaptive response
    response_level = get_adaptive_response(threat_level, cluster, models)
    logging.info(f"Session {session_id}: Response level {response_level}")
    
    # Apply response configuration
    apply_response_config(session_id, response_level)
    
    # Log the decision
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "src_ip": session_data.get('src_ip', ''),
        "threat_level": threat_level,
        "cluster": cluster,
        "response_level": response_level,
        "features": features
    }
    
    with open(os.path.join(PROCESSED_DIR, "adaptive_decisions.jsonl"), "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    return response_level

def load_json_logs(file_path):
    """Load JSON logs from a file."""
    logs = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                logs.append(log)
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from line: {line}")
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
                'hassh': log.get('hassh', '')
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

def monitor_logs(models):
    """Monitor Cowrie logs for new sessions."""
    logging.info("Starting log monitoring...")
    
    # Keep track of processed sessions
    processed_sessions = set()
    
    # Load already processed sessions if available
    try:
        with open(os.path.join(PROCESSED_DIR, "processed_sessions.json"), "r") as f:
            processed_sessions = set(json.load(f))
        logging.info(f"Loaded {len(processed_sessions)} previously processed sessions")
    except:
        logging.info("No previously processed sessions found")
    
    while True:
        try:
            # Find all current log files
            log_files = [f for f in os.listdir(COWRIE_LOG_DIR) if f.startswith('cowrie.json')]
            
            for log_file in log_files:
                file_path = os.path.join(COWRIE_LOG_DIR, log_file)
                
                # Skip if file is too old (optional)
                # file_time = os.path.getmtime(file_path)
                # if time.time() - file_time > 86400:  # Older than 1 day
                #     continue
                
                # Process the file
                logs = load_json_logs(file_path)
                sessions = extract_session_data(logs)
                
                # Process completed sessions that haven't been processed yet
                for session_id, session_data in sessions.items():
                    if session_id not in processed_sessions and session_data['end_time'] is not None:
                        logging.info(f"Found new completed session: {session_id}")
                        process_session(session_id, session_data, models)
                        processed_sessions.add(session_id)
            
            # Save processed sessions
            with open(os.path.join(PROCESSED_DIR, "processed_sessions.json"), "w") as f:
                json.dump(list(processed_sessions), f)
            
            # Sleep before checking again
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            logging.error(f"Error in monitoring loop: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retrying after an error

def main():
    """Main function."""
    print("Starting Adaptive Honeypot Integration")
    sys.stdout.flush()  # Force output to be displayed immediately
    
    logging.info("Starting Adaptive Honeypot Integration")
    
    # Load models
    print("Attempting to load models...")
    sys.stdout.flush()
    models = load_models()
    if not models:
        print("Failed to load models. Exiting.")
        sys.stdout.flush()
        logging.error("Failed to load models. Exiting.")
        return
    
    print("Models loaded successfully. Starting log monitoring...")
    sys.stdout.flush()
    logging.info("Models loaded successfully. Starting log monitoring...")
    
    # Start monitoring logs
    monitor_logs(models)

if __name__ == "__main__":
    main()

