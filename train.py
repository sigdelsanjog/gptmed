#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple training script for GptMed casual Q&A model
"""

import gptmed
from pathlib import Path

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting GptMed Training")
    print("="*60)
    
    # Use absolute path for config
    config_path = Path(__file__).parent / 'gptmed' / 'configs' / 'training_config.yaml'
    
    # Train the model
    results = gptmed.train_from_config(
        config_path=str(config_path),
        verbose=True,
        device='auto'  # Auto-selects best device (cuda or cpu)
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model checkpoint: {results['best_checkpoint']}")
    print(f"Final validation loss: {results['final_val_loss']:.4f}")
    print(f"Total epochs trained: {results['total_epochs']}")
    print("\nYou can now use the model for inference!")
