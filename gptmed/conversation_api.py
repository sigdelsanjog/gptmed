"""
Conversation API - Main entry point for conversation language model

Provides simple interface for training, inference, and conversation testing.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

# Add framework to path
FRAMEWORK_DIR = Path(__file__).parent / 'framework' / 'conversation'


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    data_file: str,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 8,
    batch_size: int = 32,
    num_epochs: int = 10,
    device: str = 'cuda',
):
    """
    Train conversation model
    
    Args:
        data_file: Path to merged_tokens.jsonl
        d_model: Model dimension
        n_layers: Number of decoder layers
        n_heads: Number of attention heads
        batch_size: Training batch size
        num_epochs: Number of epochs
        device: Device to train on (cuda/cpu)
    """
    import torch
    from framework.conversation.training.train import Trainer
    from framework.conversation.model.configs.model_config import ConversationModelConfig
    
    # Fallback to CPU if CUDA not available
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    logger.info(f"Starting model training on {device.upper()}...")
    
    config = ConversationModelConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
    )
    
    trainer = Trainer(config)
    trainer.train(data_file)
    
    logger.info("Training completed!")


def inference_model(
    checkpoint_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    device: str = 'cuda',
):
    """
    Run interactive inference mode
    
    Args:
        checkpoint_path: Direct path to checkpoint
        checkpoint_dir: Directory containing checkpoints
        device: Device to use (cuda/cpu)
    """
    import torch
    from framework.conversation.inference.inference import (
        ConversationInference,
        InferenceConfig
    )
    
    # Fallback to CPU if CUDA not available
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Determine checkpoint path
    if not checkpoint_path and not checkpoint_dir:
        # Default to best checkpoint in model directory
        checkpoint_dir = str(FRAMEWORK_DIR / 'model' / 'checkpoints')
    
    logger.info("Loading model for inference...")
    
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        device=device,
        max_length=100,
    )
    
    inference = ConversationInference(config)
    
    logger.info("Model loaded! Starting interactive mode...")
    logger.info("Type 'quit' to exit\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate response
            response = inference.chat(
                user_input,
                max_tokens=100,
                temperature=0.7,
            )
            
            print(f"Bot: {response}\n")
        
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            continue


def test_model(checkpoint_dir: Optional[str] = None):
    """
    Test model with sample prompts
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    import torch
    from framework.conversation.inference.inference import (
        ConversationInference,
        InferenceConfig
    )
    
    # Use CPU for testing if GPU not available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not checkpoint_dir:
        checkpoint_dir = str(FRAMEWORK_DIR / 'model' / 'checkpoints')
    
    logger.info("Loading model for testing...")
    
    config = InferenceConfig(
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    
    inference = ConversationInference(config)
    
    # Test prompts
    test_prompts = [
        "What is machine learning?",
        "Hello, how are you?",
        "Explain neural networks",
    ]
    
    logger.info("Running model tests...\n")
    
    for prompt in test_prompts:
        logger.info(f"Prompt: {prompt}")
        
        response = inference.chat(
            prompt,
            max_tokens=50,
            temperature=0.7,
        )
        
        logger.info(f"Response: {response}\n")


def check_setup():
    """Check if framework directory structure exists"""
    framework_path = FRAMEWORK_DIR
    
    required_dirs = [
        'model/architecture',
        'model/configs',
        'model/checkpoints',
        'training',
        'inference',
        'data',
    ]
    
    missing = []
    for dir_name in required_dirs:
        dir_path = framework_path / dir_name
        if not dir_path.exists():
            missing.append(str(dir_path))
    
    if missing:
        logger.warning("Missing framework directories:")
        for path in missing:
            logger.warning(f"  - {path}")
        return False
    
    logger.info(f"✓ Framework structure found at {framework_path}")
    return True


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Conversation Language Model API'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--data-file',
        required=True,
        help='Path to merged_tokens.jsonl file'
    )
    train_parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    train_parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    train_parser.add_argument('--n-heads', type=int, default=8, help='Number of heads')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument(
        '--device', 
        default='cuda', 
        choices=['cuda', 'cpu'],
        help='Device to train on'
    )
    
    # Inference command
    infer_parser = subparsers.add_parser('inference', help='Run interactive inference')
    infer_parser.add_argument(
        '--checkpoint-path',
        help='Direct path to checkpoint file'
    )
    infer_parser.add_argument(
        '--checkpoint-dir',
        help='Directory containing checkpoints'
    )
    infer_parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test model with sample prompts')
    test_parser.add_argument(
        '--checkpoint-dir',
        help='Directory containing checkpoints'
    )
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check framework setup')
    
    args = parser.parse_args()
    
    # Default to check if no command
    if not args.command:
        check_setup()
        print("\nUsage: python conversation_api.py <command> [options]")
        print("Commands: train, inference, test, check")
        print("\nExamples:")
        print("  python conversation_api.py train --data-file data/merged_tokens.jsonl")
        print("  python conversation_api.py inference")
        print("  python conversation_api.py test")
        print("  python conversation_api.py check")
        return
    
    # Run command
    try:
        if args.command == 'train':
            train_model(
                args.data_file,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                device=args.device,
            )
        
        elif args.command == 'inference':
            inference_model(
                checkpoint_path=args.checkpoint_path,
                checkpoint_dir=args.checkpoint_dir,
                device=args.device,
            )
        
        elif args.command == 'test':
            test_model(checkpoint_dir=args.checkpoint_dir)
        
        elif args.command == 'check':
            check_setup()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
