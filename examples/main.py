"""
main.py - Your Project Training Script

This is an example of how a user would use gptmed in their own project.
Just install gptmed, create this file, and run it!

Usage:
    1. pip install gptmed
    2. Create this main.py file
    3. Edit the config file path below
    4. Run: python main.py
"""

import gptmed

def train_my_model():
    """
    Train a custom GPT model using my configuration.
    """
    # Path to your training configuration
    config_path = 'my_training_config.yaml'
    
    # Train the model
    print("Starting training with gptmed...")
    results = gptmed.train_from_config(config_path)
    
    print(f"\n✅ Training complete!")
    print(f"Best model saved at: {results['best_checkpoint']}")
    
    return results

def test_my_model(checkpoint_path, tokenizer_path):
    """
    Test the trained model with some example questions.
    """
    questions = [
        "What is your name?",
        "How does this work?",
        "Can you explain this concept?"
    ]
    
    print("\n" + "="*60)
    print("Testing the trained model...")
    print("="*60)
    
    for question in questions:
        answer = gptmed.generate(
            checkpoint=checkpoint_path,
            tokenizer=tokenizer_path,
            prompt=question,
            max_length=100,
            temperature=0.7
        )
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print("-" * 60)

if __name__ == "__main__":
    # First time: Create a config file
    print("Do you need to create a config file? (y/n): ", end="")
    create = input().lower()
    
    if create == 'y':
        gptmed.create_config('my_training_config.yaml')
        print("\n✓ Config file created!")
        print("\nEdit 'my_training_config.yaml' with your settings, then run this script again.")
    else:
        # Train the model
        results = train_my_model()
        
        # Optional: Test the model
        print("\n\nDo you want to test the model? (y/n): ", end="")
        test = input().lower()
        
        if test == 'y':
            tokenizer = input("Enter tokenizer path: ")
            test_my_model(results['best_checkpoint'], tokenizer)
