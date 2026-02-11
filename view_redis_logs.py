#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Redis Training Logs Viewer

Quick utility to view training logs from Redis without continuous monitoring.

Usage:
    python view_redis_logs.py              # Show last 20 logs
    python view_redis_logs.py --all        # Show all logs
    python view_redis_logs.py --latest     # Show only latest log
    python view_redis_logs.py --json       # Output as JSON
"""

import redis
import json
import sys
from datetime import datetime
from argparse import ArgumentParser

class RedisLogsViewer:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.r = None
        self.connect()
    
    def connect(self):
        """Connect to Redis."""
        try:
            self.r = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            self.r.ping()
            print(f"Connected to Redis: {self.host}:{self.port}")
        except redis.ConnectionError as e:
            print(f"Error: Could not connect to Redis at {self.host}:{self.port}")
            print(f"Details: {e}")
            print("\nStart Redis with: redis-server")
            sys.exit(1)
    
    def get_logs(self, start=0, end=-1):
        """Get logs from Redis list."""
        try:
            # Try multiple possible log keys
            for key_pattern in ['training_logs', 'gptmed_training_logs', 'logs', 'training:logs']:
                logs = self.r.lrange(key_pattern, start, end)
                if logs:
                    return logs
            return []
        except Exception as e:
            print(f"Error fetching logs: {e}")
            return []
    
    def get_all_keys(self):
        """Get all keys available in Redis."""
        try:
            keys = self.r.keys('*')
            return keys
        except Exception as e:
            print(f"Error fetching keys: {e}")
            return []
    
    def view_logs(self, limit=20, show_all=False, json_format=False):
        """Display training logs."""
        if show_all:
            logs = self.get_logs()
        else:
            # Get last 'limit' logs
            total = self.r.llen('training_logs')
            if total == 0:
                lens = [self.r.llen(key) for key in ['gptmed_training_logs', 'logs', 'training:logs'] if self.r.exists(key)]
                total = max(lens) if lens else 0
            start = max(0, total - limit) if total > 0 else 0
            logs = self.get_logs(start, -1)
        
        if not logs:
            print("No training logs found in standard locations.")
            print("\nAvailable keys in Redis (these contain training data):")
            keys = self.get_all_keys()
            
            # Display all keys and their content
            if keys:
                for key in sorted(keys):
                    key_type = self.r.type(key)
                    
                    if key_type == 'string':
                        value = self.r.get(key)
                        print(f"\n  Key: {key} (string)")
                        try:
                            parsed = json.loads(value)
                            print(f"    {json.dumps(parsed, indent=6)}")
                        except:
                            print(f"    {value[:100]}")
                    
                    elif key_type == 'list':
                        count = self.r.llen(key)
                        print(f"\n  Key: {key} (list, {count} items)")
                        # Show last few items
                        items = self.r.lrange(key, -5, -1)
                        for item in items:
                            try:
                                parsed = json.loads(item)
                                print(f"    {json.dumps(parsed, indent=6)}")
                            except:
                                print(f"    {item[:100]}")
                    
                    elif key_type == 'hash':
                        data = self.r.hgetall(key)
                        print(f"\n  Key: {key} (hash)")
                        for k, v in data.items():
                            try:
                                parsed = json.loads(v)
                                print(f"    {k}: {parsed}")
                            except:
                                print(f"    {k}: {v}")
                    
                    elif key_type == 'zset':
                        count = self.r.zcard(key)
                        print(f"\n  Key: {key} (sorted set, {count} items)")
                        items = self.r.zrange(key, -5, -1)
                        for item in items:
                            print(f"    {item}")
            
            return
        
        print(f"\nFound {len(logs)} log entries:\n")
        
        if json_format:
            # Output as JSON array
            parsed_logs = []
            for log_entry in logs:
                try:
                    parsed_logs.append(json.loads(log_entry))
                except:
                    parsed_logs.append(log_entry)
            print(json.dumps(parsed_logs, indent=2))
        else:
            # Output as formatted text
            print("="*80)
            print(f"{'Epoch':<8} {'Step':<10} {'Loss':<15} {'Val Loss':<15} {'Time':<20}")
            print("="*80)
            
            for i, log_entry in enumerate(logs):
                try:
                    data = json.loads(log_entry)
                    epoch = data.get('epoch', 'N/A')
                    step = data.get('step', 'N/A')
                    loss = data.get('loss', 'N/A')
                    val_loss = data.get('val_loss', 'N/A')
                    timestamp = data.get('timestamp', '')
                    
                    if isinstance(loss, float):
                        loss = f"{loss:.6f}"
                    if isinstance(val_loss, float):
                        val_loss = f"{val_loss:.6f}"
                    
                    print(f"{str(epoch):<8} {str(step):<10} {str(loss):<15} {str(val_loss):<15} {str(timestamp):<20}")
                except json.JSONDecodeError:
                    print(f"[Entry {i}] {log_entry[:75]}")
            
            print("="*80)
    
    def get_summary(self):
        """Print summary of training progress."""
        logs = self.get_logs()
        
        if not logs:
            print("\nNo training logs found, but found these training keys:")
            # Check for training:validation and training:steps
            if self.r.exists('training:validation'):
                val_data = self.r.get('training:validation')
                print(f"\ntraining:validation:")
                try:
                    parsed = json.loads(val_data)
                    print(f"  {json.dumps(parsed, indent=2)}")
                except:
                    print(f"  {val_data}")
            
            if self.r.exists('training:steps'):
                steps_data = self.r.get('training:steps')
                print(f"\ntraining:steps:")
                try:
                    parsed = json.loads(steps_data)
                    print(f"  {json.dumps(parsed, indent=2)}")
                except:
                    print(f"  {steps_data}")
            return
        
        print("\nTraining Summary:")
        print("-" * 40)
        
        try:
            # Parse last log to get current status
            last_log = json.loads(logs[-1])
            
            print(f"Total entries: {len(logs)}")
            print(f"Current epoch: {last_log.get('epoch', 'N/A')}")
            print(f"Current step: {last_log.get('step', 'N/A')}")
            print(f"Latest loss: {last_log.get('loss', 'N/A')}")
            print(f"Latest val loss: {last_log.get('val_loss', 'N/A')}")
            print(f"Last update: {last_log.get('timestamp', 'N/A')}")
        
        except Exception as e:
            print(f"Error parsing summary: {e}")

def main():
    parser = ArgumentParser(description='View training logs from Redis')
    parser.add_argument('--host', default='localhost', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--db', type=int, default=0, help='Redis database')
    parser.add_argument('--limit', type=int, default=20, help='Number of logs to show')
    parser.add_argument('--all', action='store_true', help='Show all logs')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    parser.add_argument('--latest', action='store_true', help='Show only latest log')
    
    args = parser.parse_args()
    
    viewer = RedisLogsViewer(host=args.host, port=args.port, db=args.db)
    
    if args.latest:
        viewer.view_logs(limit=1)
    elif args.summary:
        viewer.get_summary()
    else:
        viewer.view_logs(limit=args.limit, show_all=args.all, json_format=args.json)
        print("\n" + "-"*40)
        viewer.get_summary()

if __name__ == '__main__':
    main()
