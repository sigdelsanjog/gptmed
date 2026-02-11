#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitor GptMed Training via Redis

Displays real-time training metrics stored in Redis database.
Requires Redis server to be running.

Usage:
    python monitor_training.py
"""

import redis
import json
from datetime import datetime
import time
import sys

def connect_to_redis(host='localhost', port=6379, db=0, password=None):
    """Connect to Redis server."""
    try:
        r = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        # Test connection
        r.ping()
        print("✓ Connected to Redis successfully")
        return r
    except redis.ConnectionError as e:
        print(f"✗ Failed to connect to Redis: {e}")
        print("\nMake sure Redis server is running:")
        print("  redis-server")
        sys.exit(1)

def get_training_metrics(r):
    """Fetch training metrics from Redis."""
    try:
        # Get all keys related to training
        keys = r.keys('*training*')
        metrics_keys = r.keys('*metrics*')
        
        all_keys = keys + metrics_keys
        
        if not all_keys:
            print("No training data found in Redis yet...")
            return None
        
        metrics = {}
        for key in all_keys:
            value = r.get(key)
            if value:
                try:
                    metrics[key] = json.loads(value)
                except:
                    metrics[key] = value
        
        return metrics
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None

def get_jsonl_logs(r):
    """Get training logs from Redis list."""
    try:
        # Get logs stored as list items
        logs = r.lrange('training_logs', 0, -1)
        return logs
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return None

def display_metrics(metrics):
    """Display metrics in readable format."""
    if not metrics:
        print("No metrics available")
        return
    
    print("\n" + "="*60)
    print("Training Metrics from Redis")
    print("="*60)
    
    for key, value in sorted(metrics.items()):
        print(f"\n{key}:")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {value}")

def monitor_training(host='localhost', port=6379, db=0, password=None, interval=5):
    """Monitor training in real-time."""
    r = connect_to_redis(host, port, db, password)
    
    print("\n" + "="*60)
    print("Training Monitoring")
    print("="*60)
    print(f"Redis: {host}:{port} (db={db})")
    print(f"Update interval: {interval}s")
    print("Press Ctrl+C to stop\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Get latest logs
            logs = get_jsonl_logs(r)
            
            # Get metrics
            metrics = get_training_metrics(r)
            
            # Clear screen (optional)
            if iteration > 1:
                print("\n" + "-"*60)
            
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking Redis...")
            
            if logs:
                print(f"\nFound {len(logs)} log entries:")
                # Display last few entries
                for log_entry in logs[-5:]:
                    try:
                        log_data = json.loads(log_entry)
                        epoch = log_data.get('epoch', 'N/A')
                        step = log_data.get('step', 'N/A')
                        loss = log_data.get('loss', 'N/A')
                        val_loss = log_data.get('val_loss', 'N/A')
                        
                        print(f"  Epoch {epoch}, Step {step}: loss={loss}, val_loss={val_loss}")
                    except:
                        print(f"  {log_entry[:100]}")
            else:
                print("No logs in Redis yet...")
            
            if metrics:
                display_metrics(metrics)
            
            print(f"\nNext update in {interval}s... (Ctrl+C to stop)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)

if __name__ == '__main__':
    # Redis configuration
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_PASSWORD = None
    UPDATE_INTERVAL = 5  # seconds
    
    # Start monitoring
    monitor_training(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        interval=UPDATE_INTERVAL
    )
