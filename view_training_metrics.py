#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Redis Training Metrics Viewer

Display training metrics stored in Redis lists:
- training:steps - Training step metrics
- training:validation - Validation metrics

Usage:
    python view_training_metrics.py              # Show both steps and validation
    python view_training_metrics.py --steps      # Show only training steps
    python view_training_metrics.py --validation # Show only validation metrics
    python view_training_metrics.py --latest     # Show latest metrics
"""

import redis
import json
import ast
import sys
from datetime import datetime
from argparse import ArgumentParser

class TrainingMetricsViewer:
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
            return True
        except redis.ConnectionError as e:
            print(f"Error: Could not connect to Redis at {self.host}:{self.port}")
            return False
    
    def get_steps(self):
        """Get training steps from Redis list."""
        try:
            steps = self.r.lrange('training:steps', 0, -1)
            parsed = []
            for step in steps:
                try:
                    # Try JSON first
                    try:
                        parsed.append(json.loads(step))
                    except:
                        # Fallback to Python literal eval for dict strings
                        parsed.append(ast.literal_eval(step))
                except:
                    pass
            return parsed
        except Exception as e:
            print(f"Error fetching steps: {e}")
            return []
    
    def get_validation(self):
        """Get validation metrics from Redis list."""
        try:
            val = self.r.lrange('training:validation', 0, -1)
            parsed = []
            for v in val:
                try:
                    # Try JSON first
                    try:
                        parsed.append(json.loads(v))
                    except:
                        # Fallback to Python literal eval for dict strings
                        parsed.append(ast.literal_eval(v))
                except:
                    pass
            return parsed
        except Exception as e:
            print(f"Error fetching validation: {e}")
            return []
    
    def display_steps(self, latest_only=False):
        """Display training steps."""
        steps = self.get_steps()
        
        if not steps:
            print("No training steps in Redis.")
            return
        
        if latest_only:
            steps = steps[-1:]
        
        print("\n" + "="*80)
        print("TRAINING STEPS")
        print("="*80)
        print(f"{'Step':<10} {'Loss (MA)':<15} {'Timestamp':<20} {'Type':<10}")
        print("-"*80)
        
        for step in steps:
            step_num = step.get('step', 'N/A')
            loss = step.get('moving_avg_loss', 'N/A')
            timestamp = step.get('timestamp', 'N/A')
            msg_type = step.get('type', 'N/A')
            
            if isinstance(loss, float):
                loss_str = f"{loss:.6f}"
            else:
                loss_str = str(loss)
            
            timestamp_str = f"{timestamp:.2f}" if isinstance(timestamp, float) else str(timestamp)
            
            print(f"{str(step_num):<10} {loss_str:<15} {timestamp_str:<20} {str(msg_type):<10}")
        
        print("="*80)
    
    def display_validation(self, latest_only=False):
        """Display validation metrics."""
        val = self.get_validation()
        
        if not val:
            print("No validation metrics in Redis.")
            return
        
        if latest_only:
            val = val[-1:]
        
        print("\n" + "="*80)
        print("VALIDATION METRICS")
        print("="*80)
        print(f"{'Step':<10} {'Val Loss':<15} {'Best':<8} {'Timestamp':<20}")
        print("-"*80)
        
        for v in val:
            step = v.get('step', 'N/A')
            val_loss = v.get('val_loss', 'N/A')
            is_best = v.get('is_best', False)
            timestamp = v.get('timestamp', 'N/A')
            
            val_loss_str = f"{val_loss:.6f}" if isinstance(val_loss, float) else str(val_loss)
            best_str = "✓ YES" if is_best else "NO"
            timestamp_str = f"{timestamp:.2f}" if isinstance(timestamp, float) else str(timestamp)
            
            print(f"{str(step):<10} {val_loss_str:<15} {best_str:<8} {timestamp_str:<20}")
        
        print("="*80)
    
    def display_summary(self):
        """Display training summary."""
        steps = self.get_steps()
        val = self.get_validation()
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        if steps:
            latest_step = steps[-1]
            print(f"Total training steps recorded: {len(steps)}")
            print(f"Latest step: {latest_step.get('step', 'N/A')}")
            print(f"Latest loss (moving avg): {latest_step.get('moving_avg_loss', 'N/A'):.6f}")
        
        if val:
            latest_val = val[-1]
            print(f"\nTotal validation checks: {len(val)}")
            print(f"Latest val step: {latest_val.get('step', 'N/A')}")
            print(f"Latest val loss: {latest_val.get('val_loss', 'N/A')}")
            print(f"Is best model: {latest_val.get('is_best', False)}")
            
            # Find best validation
            best_val = min(val, key=lambda x: x.get('val_loss', float('inf')))
            print(f"\nBest validation loss: {best_val.get('val_loss', 'N/A')} (step {best_val.get('step', 'N/A')})")
        
        print("="*60)

def main():
    parser = ArgumentParser(description='View training metrics from Redis')
    parser.add_argument('--host', default='localhost', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--db', type=int, default=0, help='Redis database')
    parser.add_argument('--steps', action='store_true', help='Show only training steps')
    parser.add_argument('--validation', action='store_true', help='Show only validation metrics')
    parser.add_argument('--latest', action='store_true', help='Show only latest entry')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    
    args = parser.parse_args()
    
    viewer = TrainingMetricsViewer(host=args.host, port=args.port, db=args.db)
    
    if not viewer.r:
        sys.exit(1)
    
    print(f"Connected to Redis: {args.host}:{args.port}")
    
    if args.summary:
        viewer.display_summary()
    elif args.steps:
        viewer.display_steps(latest_only=args.latest)
    elif args.validation:
        viewer.display_validation(latest_only=args.latest)
    else:
        # Show both
        viewer.display_steps(latest_only=args.latest)
        viewer.display_validation(latest_only=args.latest)
        viewer.display_summary()

if __name__ == '__main__':
    main()
