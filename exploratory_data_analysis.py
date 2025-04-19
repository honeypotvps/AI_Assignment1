#!/usr/bin/env python3
"""
Exploratory Data Analysis for Cowrie Honeypot Logs (Part 1)
This script performs exploratory data analysis on preprocessed Cowrie logs.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter

# Set plot style
plt.style.use('ggplot')
sns.set(style="darkgrid")

# Define directories
DATA_DIR = "/opt/cowrie/adaptive-honeypot/processed_data"
OUTPUT_DIR = "/opt/cowrie/adaptive-honeypot/analysis_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    """Load the preprocessed data."""
    sessions_df = pd.read_csv(os.path.join(DATA_DIR, 'cowrie_sessions.csv'))
    events_df = pd.read_csv(os.path.join(DATA_DIR, 'cowrie_events.csv'))
    
    # Convert timestamp to datetime
    if 'start_time' in sessions_df.columns:
        sessions_df['start_datetime'] = pd.to_datetime(sessions_df['start_time'], errors='coerce')
    
    if 'end_time' in sessions_df.columns:
        sessions_df['end_datetime'] = pd.to_datetime(sessions_df['end_time'], errors='coerce')
    
    if 'timestamp' in events_df.columns:
        events_df['datetime'] = pd.to_datetime(events_df['timestamp'], errors='coerce')
    
    return sessions_df, events_df
def analyze_event_distribution(events_df):
    """Analyze the distribution of event types."""
    # Count event types
    event_counts = events_df['event_id'].value_counts()
    total_events = len(events_df)
    
    # Calculate percentages
    event_percentages = event_counts / total_events * 100
    
    # Create DataFrame for plotting
    event_df = pd.DataFrame({
        'Count': event_counts,
        'Percentage': event_percentages
    })
    
    # Plot event distribution
    plt.figure(figsize=(12, 8))
    ax = event_df['Percentage'].sort_values(ascending=False).head(15).plot(kind='bar')
    plt.title('Top 15 Event Types')
    plt.xlabel('Event Type')
    plt.ylabel('Percentage of Events')
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.2f}%", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'event_distribution.png'), dpi=300)
    
    # Save results
    event_df.sort_values('Count', ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, 'event_distribution.csv')
    )
    
    return event_df
def analyze_login_attempts(sessions_df, events_df):
    """Analyze login attempts."""
    # Filter login events
    login_success = events_df[events_df['event_id'] == 'cowrie.login.success']
    login_failed = events_df[events_df['event_id'] == 'cowrie.login.failed']
    
    # Count successful and failed logins
    success_count = len(login_success)
    failed_count = len(login_failed)
    total_attempts = success_count + failed_count
    
    # Calculate success rate
    success_rate = success_count / total_attempts * 100 if total_attempts > 0 else 0
    
    # Top usernames and passwords
    top_usernames_success = login_success['username'].value_counts().head(10)
    top_passwords_success = login_success['password'].value_counts().head(10)
    top_usernames_failed = login_failed['username'].value_counts().head(10)
    top_passwords_failed = login_failed['password'].value_counts().head(10)
    
    # Plot login success vs failure
    plt.figure(figsize=(10, 6))
    plt.bar(['Failed', 'Successful'], [failed_count, success_count])
    plt.title('Login Attempts')
    plt.ylabel('Count')
    plt.annotate(f"{failed_count} ({100-success_rate:.1f}%)", 
                xy=(0, failed_count), 
                xytext=(0, 5),
                textcoords='offset points',
                ha='center')
    plt.annotate(f"{success_count} ({success_rate:.1f}%)", 
                xy=(1, success_count), 
                xytext=(0, 5),
                textcoords='offset points',
                ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'login_attempts.png'), dpi=300)
    
    # Plot top usernames (successful logins)
    plt.figure(figsize=(12, 8))
    top_usernames_success.plot(kind='bar')
    plt.title('Top Usernames in Successful Logins')
    plt.xlabel('Username')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_usernames_success.png'), dpi=300)
    
    # Plot top passwords (successful logins)
    plt.figure(figsize=(12, 8))
    top_passwords_success.plot(kind='bar')
    plt.title('Top Passwords in Successful Logins')
    plt.xlabel('Password')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_passwords_success.png'), dpi=300)
    
    # Save results
    login_stats = {
        'total_attempts': total_attempts,
        'successful_logins': success_count,
        'failed_logins': failed_count,
        'success_rate': success_rate,
        'top_usernames_success': top_usernames_success.to_dict(),
        'top_passwords_success': top_passwords_success.to_dict(),
        'top_usernames_failed': top_usernames_failed.to_dict(),
        'top_passwords_failed': top_passwords_failed.to_dict()
    }
    
    with open(os.path.join(OUTPUT_DIR, 'login_analysis.json'), 'w') as f:
        json.dump(login_stats, f, indent=4)
    
    return login_stats
def analyze_command_execution(sessions_df):
    """Analyze command execution patterns."""
    # Filter sessions with commands
    command_sessions = sessions_df[sessions_df['command_count'] > 0]
    
    # Extract all commands
    all_commands = []
    command_types = {
        'download': 0,
        'privilege_escalation': 0,
        'file_access': 0,
        'reconnaissance': 0,
        'persistence': 0,
        'malware': 0,
        'other': 0
    }
    
    # Process commands
    for commands in command_sessions['commands']:
        if isinstance(commands, str):
            cmd_list = commands.split(' | ')
            all_commands.extend(cmd_list)
            
            # Count command types
            if re.search(r'wget|curl|tftp|scp|fetch', commands):
                command_types['download'] += 1
            if re.search(r'sudo|su\s|chmod|chown', commands):
                command_types['privilege_escalation'] += 1
            if re.search(r'cat|less|more|head|tail|nano|vi|vim', commands):
                command_types['file_access'] += 1
            if re.search(r'ls|dir|pwd|whoami|id|uname|ifconfig|ip\s', commands):
                command_types['reconnaissance'] += 1
            if re.search(r'crontab|systemctl|service|rc\.local|init\.d', commands):
                command_types['persistence'] += 1
            if re.search(r'\.sh|\.pl|\.py|\.bin|\.elf|chmod\s+[0-7]*x', commands):
                command_types['malware'] += 1
            if not any(re.search(pattern, commands) for pattern in [
                r'wget|curl|tftp|scp|fetch',
                r'sudo|su\s|chmod|chown',
                r'cat|less|more|head|tail|nano|vi|vim',
                r'ls|dir|pwd|whoami|id|uname|ifconfig|ip\s',
                r'crontab|systemctl|service|rc\.local|init\.d',
                r'\.sh|\.pl|\.py|\.bin|\.elf|chmod\s+[0-7]*x'
            ]):
                command_types['other'] += 1
    
    # Count command frequency
    command_counter = Counter(all_commands)
    top_commands = command_counter.most_common(20)
    
    # Plot command type distribution
    plt.figure(figsize=(12, 8))
    command_types_df = pd.DataFrame({
        'Count': command_types.values(),
        'Percentage': [v / sum(command_types.values()) * 100 for v in command_types.values()]
    }, index=command_types.keys())
    
    ax = command_types_df.sort_values('Count', ascending=False)['Percentage'].plot(kind='bar')
    plt.title('Command Type Distribution')
    plt.xlabel('Command Type')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.2f}%", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'command_type_distribution.png'), dpi=300)
    
    # Plot top commands
    plt.figure(figsize=(14, 10))
    top_commands_df = pd.DataFrame(top_commands, columns=['Command', 'Count'])
    top_commands_df.set_index('Command')['Count'].plot(kind='bar')
    plt.title('Top 20 Commands')
    plt.xlabel('Command')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_commands.png'), dpi=300)
    
    # Save results
    command_analysis = {
        'total_command_sessions': len(command_sessions),
        'total_commands': len(all_commands),
        'unique_commands': len(command_counter),
        'command_types': command_types,
        'top_commands': dict(top_commands)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'command_analysis.json'), 'w') as f:
        json.dump(command_analysis, f, indent=4)
    
    return command_analysis
def analyze_threat_levels(sessions_df):
    """Analyze threat level distribution and characteristics."""
    # Count threat levels
    threat_counts = sessions_df['threat_level'].value_counts()
    total_sessions = len(sessions_df)
    
    # Calculate percentages
    threat_percentages = threat_counts / total_sessions * 100
    
    # Create DataFrame for plotting
    threat_df = pd.DataFrame({
        'Count': threat_counts,
        'Percentage': threat_percentages
    })
    
    # Plot threat level distribution
    plt.figure(figsize=(10, 6))
    ax = threat_df['Percentage'].plot(kind='bar')
    plt.title('Threat Level Distribution')
    plt.xlabel('Threat Level')
    plt.ylabel('Percentage of Sessions')
    plt.xticks(rotation=0)
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.2f}%", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'threat_level_distribution.png'), dpi=300)
    
    # Analyze characteristics by threat level

    threat_characteristics = {}
    
    for level in ['high', 'medium', 'low']:
        level_df = sessions_df[sessions_df['threat_level'] == level]
        
        characteristics = {
            'count': len(level_df),
            'percentage': len(level_df) / total_sessions * 100,
            'avg_duration': level_df['duration'].mean(),
            'avg_command_count': level_df['command_count'].mean(),
            'login_success_rate': level_df['login_success'].mean() * 100,
            'has_download_command': level_df['has_download_command'].mean() * 100 if 'has_download_command' in level_df.columns else 0,
            'has_privilege_escalation': level_df['has_privilege_escalation'].mean() * 100 if 'has_privilege_escalation' in level_df.columns else 0,
            'has_reconnaissance': level_df['has_reconnaissance'].mean() * 100 if 'has_reconnaissance' in level_df.columns else 0,
            'has_persistence': level_df['has_persistence'].mean() * 100 if 'has_persistence' in level_df.columns else 0,
            'has_malware_indicators': level_df['has_malware_indicators'].mean() * 100 if 'has_malware_indicators' in level_df.columns else 0
        }
        
        threat_characteristics[level] = characteristics
    
    # Plot characteristics comparison
    characteristics_to_plot = [
        'avg_command_count', 'login_success_rate', 'has_download_command',
        'has_privilege_escalation', 'has_reconnaissance', 'has_persistence',
        'has_malware_indicators'
    ]
    
    for char in characteristics_to_plot:
        plt.figure(figsize=(10, 6))
        values = [threat_characteristics[level][char] for level in ['high', 'medium', 'low']]
        
        ax = plt.bar(['High', 'Medium', 'Low'], values)
        plt.title(f'{char.replace("_", " ").title()} by Threat Level')
        plt.ylabel('Value')
        
        # Add value labels
        for i, v in enumerate(values):
            if char == 'avg_command_count':
                plt.annotate(f"{v:.2f}", 
                           (i, v), 
                           ha = 'center', va = 'bottom', 
                           xytext = (0, 5), textcoords = 'offset points')
            else:
                plt.annotate(f"{v:.2f}%", 
                           (i, v), 
                           ha = 'center', va = 'bottom', 
                           xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'threat_level_{char}.png'), dpi=300)
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'threat_level_analysis.json'), 'w') as f:
        json.dump({
            'threat_counts': threat_counts.to_dict(),
            'threat_percentages': threat_percentages.to_dict(),
            'threat_characteristics': threat_characteristics
        }, f, indent=4)
    
    return threat_characteristics
def analyze_temporal_patterns(sessions_df, events_df):
    """Analyze temporal patterns in the data."""
    # Ensure datetime columns exist
    if 'start_datetime' not in sessions_df.columns or 'datetime' not in events_df.columns:
        print("Datetime columns not available for temporal analysis")
        return None
    
    # Sessions by day
    sessions_df['day'] = sessions_df['start_datetime'].dt.date
    sessions_by_day = sessions_df.groupby('day').size()
    
    # Sessions by hour
    sessions_df['hour'] = sessions_df['start_datetime'].dt.hour
    sessions_by_hour = sessions_df.groupby('hour').size()
    
    # Events by day
    events_df['day'] = events_df['datetime'].dt.date
    events_by_day = events_df.groupby('day').size()
    
    # Events by hour
    events_df['hour'] = events_df['datetime'].dt.hour
    events_by_hour = events_df.groupby('hour').size()
    
    # Plot sessions by day
    plt.figure(figsize=(14, 8))
    sessions_by_day.plot(kind='line', marker='o')
    plt.title('Sessions by Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Sessions')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sessions_by_day.png'), dpi=300)
    
    # Plot sessions by hour
    plt.figure(figsize=(14, 8))
    sessions_by_hour.plot(kind='bar')
    plt.title('Sessions by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Sessions')
    plt.xticks(range(24))
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sessions_by_hour.png'), dpi=300)
    
    # Plot events by day
    plt.figure(figsize=(14, 8))
    events_by_day.plot(kind='line', marker='o')
    plt.title('Events by Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Events')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'events_by_day.png'), dpi=300)
    
    # Plot events by hour
    plt.figure(figsize=(14, 8))
    events_by_hour.plot(kind='bar')
    plt.title('Events by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Events')
    plt.xticks(range(24))
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'events_by_hour.png'), dpi=300)
    
    # Save results
    temporal_analysis = {
        'sessions_by_day': sessions_by_day.to_dict(),
        'sessions_by_hour': sessions_by_hour.to_dict(),
        'events_by_day': events_by_day.to_dict(),
        'events_by_hour': events_by_hour.to_dict()
    }
    
    with open(os.path.join(OUTPUT_DIR, 'temporal_analysis.json'), 'w') as f:
        json.dump(temporal_analysis, f, indent=4)
    
    return temporal_analysis
def analyze_source_ips(sessions_df):
    """Analyze source IP addresses."""
    # Count sessions by source IP
    ip_counts = sessions_df['src_ip'].value_counts()
    
    # Top source IPs
    top_ips = ip_counts.head(20)
    
    # Plot top source IPs
    plt.figure(figsize=(14, 10))
    top_ips.plot(kind='bar')
    plt.title('Top 20 Source IP Addresses')
    plt.xlabel('IP Address')
    plt.ylabel('Number of Sessions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_source_ips.png'), dpi=300)
    
    # Analyze threat levels by top IPs
    top_ip_threats = {}
    
    for ip in top_ips.index:
        ip_sessions = sessions_df[sessions_df['src_ip'] == ip]
        threat_counts = ip_sessions['threat_level'].value_counts()
        
        top_ip_threats[ip] = {
            'total_sessions': len(ip_sessions),
            'high_threat': threat_counts.get('high', 0),
            'medium_threat': threat_counts.get('medium', 0),
            'low_threat': threat_counts.get('low', 0),
            'login_success_rate': ip_sessions['login_success'].mean() * 100,
            'avg_command_count': ip_sessions['command_count'].mean()
        }
    
    # Plot threat distribution for top IPs
    plt.figure(figsize=(16, 12))
    
    # Create data for stacked bar chart
    ip_labels = list(top_ip_threats.keys())[:10]  # Top 10 IPs
    high_threats = [top_ip_threats[ip]['high_threat'] for ip in ip_labels]
    medium_threats = [top_ip_threats[ip]['medium_threat'] for ip in ip_labels]
    low_threats = [top_ip_threats[ip]['low_threat'] for ip in ip_labels]
    
    # Create stacked bar chart
    plt.bar(ip_labels, low_threats, label='Low Threat')
    plt.bar(ip_labels, medium_threats, bottom=low_threats, label='Medium Threat')
    plt.bar(ip_labels, high_threats, bottom=[i+j for i,j in zip(low_threats, medium_threats)], label='High Threat')
    
    plt.title('Threat Distribution for Top 10 Source IPs')
    plt.xlabel('IP Address')
    plt.ylabel('Number of Sessions')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'ip_threat_distribution.png'), dpi=300)
    
    # Save results
    ip_analysis = {
        'unique_ips': len(ip_counts),
        'top_ips': top_ips.to_dict(),
        'top_ip_threats': top_ip_threats
    }
    
    with open(os.path.join(OUTPUT_DIR, 'source_ip_analysis.json'), 'w') as f:
        json.dump(ip_analysis, f, indent=4)
    
    return ip_analysis
def generate_analysis_summary():
    """Generate a summary of the exploratory data analysis."""
    summary = []
    
    summary.append("# Exploratory Data Analysis Summary")
    summary.append("\n## Overview")
    summary.append("This document summarizes the findings from the exploratory data analysis of Cowrie honeypot logs.")
    
    # Load analysis results
    try:
        with open(os.path.join(OUTPUT_DIR, 'login_analysis.json'), 'r') as f:
            login_analysis = json.load(f)
        
        with open(os.path.join(OUTPUT_DIR, 'command_analysis.json'), 'r') as f:
            command_analysis = json.load(f)
        
        with open(os.path.join(OUTPUT_DIR, 'threat_level_analysis.json'), 'r') as f:
            threat_analysis = json.load(f)
        
        with open(os.path.join(OUTPUT_DIR, 'source_ip_analysis.json'), 'r') as f:
            ip_analysis = json.load(f)
    except:
        summary.append("\nAnalysis result files not found.")
        return '\n'.join(summary)
    
    # Login Attempts
    summary.append("\n## Login Attempts")
    summary.append(f"\nTotal login attempts: {login_analysis['total_attempts']}")
    summary.append(f"Successful logins: {login_analysis['successful_logins']} ({login_analysis['success_rate']:.2f}%)")
    summary.append(f"Failed logins: {login_analysis['failed_logins']} ({100-login_analysis['success_rate']:.2f}%)")
    
    summary.append("\n### Top Usernames in Successful Logins")
    for username, count in list(login_analysis['top_usernames_success'].items())[:5]:
        summary.append(f"- {username}: {count}")
    
    summary.append("\n### Top Passwords in Successful Logins")
    for password, count in list(login_analysis['top_passwords_success'].items())[:5]:
        summary.append(f"- {password}: {count}")
    
    # Command Execution
    summary.append("\n## Command Execution")
    summary.append(f"\nSessions with commands: {command_analysis['total_command_sessions']}")
    summary.append(f"Total commands executed: {command_analysis['total_commands']}")
    summary.append(f"Unique commands: {command_analysis['unique_commands']}")
    
    summary.append("\n### Command Type Distribution")
    total_commands = sum(command_analysis['command_types'].values())
    for cmd_type, count in sorted(command_analysis['command_types'].items(), key=lambda x: x[1], reverse=True):
        summary.append(f"- {cmd_type.replace('_', ' ').title()}: {count} ({count/total_commands*100:.2f}%)")
    
    summary.append("\n### Top Commands")
    for cmd, count in list(command_analysis['top_commands'].items())[:5]:
        summary.append(f"- {cmd}: {count}")
    
    # Threat Levels
    summary.append("\n## Threat Levels")
    summary.append(f"\nHigh threat sessions: {threat_analysis['threat_counts']['high']} ({threat_analysis['threat_percentages']['high']:.2f}%)")
    summary.append(f"Medium threat sessions: {threat_analysis['threat_counts']['medium']} ({threat_analysis['threat_percentages']['medium']:.2f}%)")
    summary.append(f"Low threat sessions: {threat_analysis['threat_counts']['low']} ({threat_analysis['threat_percentages']['low']:.2f}%)")
    
    summary.append("\n### Characteristics by Threat Level")
    summary.append("\n**High Threat Sessions:**")
    summary.append(f"- Average command count: {threat_analysis['threat_characteristics']['high']['avg_command_count']:.2f}")
    summary.append(f"- Login success rate: {threat_analysis['threat_characteristics']['high']['login_success_rate']:.2f}%")
    summary.append(f"- Download command presence: {threat_analysis['threat_characteristics']['high']['has_download_command']:.2f}%")
    summary.append(f"- Privilege escalation attempts: {threat_analysis['threat_characteristics']['high']['has_privilege_escalation']:.2f}%")
    
    summary.append("\n**Medium Threat Sessions:**")
    summary.append(f"- Average command count: {threat_analysis['threat_characteristics']['medium']['avg_command_count']:.2f}")
    summary.append(f"- Login success rate: {threat_analysis['threat_characteristics']['medium']['login_success_rate']:.2f}%")
    
    summary.append("\n**Low Threat Sessions:**")
    summary.append(f"- Average command count: {threat_analysis['threat_characteristics']['low']['avg_command_count']:.2f}")
    summary.append(f"- Login success rate: {threat_analysis['threat_characteristics']['low']['login_success_rate']:.2f}%")
    
    # Source IPs
    summary.append("\n## Source IP Analysis")
    summary.append(f"\nUnique source IPs: {ip_analysis['unique_ips']}")
    
    summary.append("\n### Top Source IPs")
    for ip, count in list(ip_analysis['top_ips'].items())[:5]:
        summary.append(f"- {ip}: {count} sessions")
    
    # Write summary to file
    with open(os.path.join(OUTPUT_DIR, 'analysis_summary.md'), 'w') as f:
        f.write('\n'.join(summary))
    
    return '\n'.join(summary)
def main():
    """Main function to perform exploratory data analysis."""
    print("Starting exploratory data analysis...")
    
    # Load data
    sessions_df, events_df = load_data()
    
    print(f"Loaded {len(sessions_df)} sessions and {len(events_df)} events")
    
    # Perform analyses
    print("Analyzing event distribution...")
    event_df = analyze_event_distribution(events_df)
    
    print("Analyzing login attempts...")
    login_stats = analyze_login_attempts(sessions_df, events_df)
    
    print("Analyzing command execution...")
    command_analysis = analyze_command_execution(sessions_df)
    
    print("Analyzing threat levels...")
    threat_characteristics = analyze_threat_levels(sessions_df)
    
    print("Analyzing temporal patterns...")
    temporal_analysis = analyze_temporal_patterns(sessions_df, events_df)
    
    print("Analyzing source IPs...")
    ip_analysis = analyze_source_ips(sessions_df)
    
    print("Generating analysis summary...")
    summary = generate_analysis_summary()
    
    print("Exploratory data analysis complete!")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
