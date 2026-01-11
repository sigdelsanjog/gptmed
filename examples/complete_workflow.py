"""
Complete Training Example

This example shows the complete workflow from data preparation to training.
"""

import gptmed

def main():
    print("="*60)
    print("GptMed - Complete Training Example")
    print("="*60)
    
    # Step 1: Create configuration
    print("\n📝 Step 1: Creating training configuration...")
    config_file = 'training_config.yaml'
    gptmed.create_config(config_file)
    print(f"✓ Created: {config_file}")
    
    print("\n" + "="*60)
    print("⏸️  PAUSE - Edit the configuration file!")
    print("="*60)
    print(f"\nOpen '{config_file}' and update these settings:")
    print("  • model.size: Choose tiny/small/medium")
    print("  • data.train_data: Path to your train.npy")
    print("  • data.val_data: Path to your val.npy")
    print("  • training.num_epochs: How many epochs to train")
    print("  • training.batch_size: Adjust based on your GPU")
    print("\nPress Enter after editing the file...")
    input()
    
    # Step 2: Train the model
    print("\n🚀 Step 2: Starting training...")
    results = gptmed.train_from_config(config_file)
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {results['best_checkpoint']}")
    print(f"   Val loss: {results['final_val_loss']:.4f}")
    
    # Step 3: Test generation
    print("\n🤖 Step 3: Testing the trained model...")
    
    # You need to provide your tokenizer path
    tokenizer_path = input("\nEnter tokenizer path (e.g., tokenizer/my_tokenizer.model): ")
    test_prompt = input("Enter a test question: ")
    
    answer = gptmed.generate(
        checkpoint=results['best_checkpoint'],
        tokenizer=tokenizer_path,
        prompt=test_prompt,
        max_length=150,
        temperature=0.7
    )
    
    print(f"\n{'='*60}")
    print("GENERATED ANSWER")
    print(f"{'='*60}")
    print(f"Q: {test_prompt}")
    print(f"A: {answer}")
    print(f"{'='*60}")
    
    print("\n✅ All done! Your model is ready to use.")
    print(f"\n📁 Files created:")
    print(f"   Config: {config_file}")
    print(f"   Checkpoint: {results['best_checkpoint']}")
    print(f"   Logs: {results['log_dir']}")

if __name__ == "__main__":
    main()
