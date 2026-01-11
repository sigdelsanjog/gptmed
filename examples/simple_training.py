"""
Example: Simple Training with GptMed

This shows the simplest way to train a model using a config file.
Perfect for users who just want to train without dealing with code.
"""

import gptmed

# Step 1: Create a training configuration file
print("Creating training configuration file...")
gptmed.create_config('my_training_config.yaml')

print("\n✓ Configuration file created: my_training_config.yaml")
print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("1. Edit 'my_training_config.yaml' with your settings:")
print("   - Set your model size (tiny/small/medium)")
print("   - Point to your tokenized data files")
print("   - Adjust batch size, learning rate, epochs, etc.")
print("")
print("2. Run training:")
print("   results = gptmed.train_from_config('my_training_config.yaml')")
print("")
print("3. Use your trained model:")
print("   answer = gptmed.generate(")
print("       checkpoint=results['best_checkpoint'],")
print("       tokenizer='path/to/tokenizer.model',")
print("       prompt='Your question?'")
print("   )")
print("="*60)

# Uncomment below to train after editing the config file:
# 
# # Step 2: Train the model
# print("\nStarting training...")
# results = gptmed.train_from_config('my_training_config.yaml')
# 
# # Step 3: Use the trained model
# print("\nGenerating sample answer...")
# answer = gptmed.generate(
#     checkpoint=results['best_checkpoint'],
#     tokenizer='tokenizer/my_tokenizer.model',
#     prompt='What is machine learning?',
#     max_length=100,
#     temperature=0.7
# )
# 
# print(f"\nQ: What is machine learning?")
# print(f"A: {answer}")
