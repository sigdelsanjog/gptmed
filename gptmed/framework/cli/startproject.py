import os
import sys

def create_qna_templates(project_name):
    """Create boilerplate for QNA model training architecture"""
    
    # Create directory structure
    os.makedirs(os.path.join(project_name, "configs"))
    os.makedirs(os.path.join(project_name, "data", "raw"))
    os.makedirs(os.path.join(project_name, "data", "processed"))
    os.makedirs(os.path.join(project_name, "models", "checkpoints"))
    os.makedirs(os.path.join(project_name, "tokenizer"))
    os.makedirs(os.path.join(project_name, "logs"))
    os.makedirs(os.path.join(project_name, "inference"))
    
    # Create main.py
    with open(os.path.join(project_name, "main.py"), "w") as f:
        f.write("""\"\"\"
Main entry point for QNA Model Training
\"\"\"
import gptmed
from pathlib import Path

def main():
    # Step 1: Create configuration
    config_path = 'configs/training_config.yaml'
    if not Path(config_path).exists():
        gptmed.create_config(config_path)
        print(f"Configuration file created at {config_path}")
        print("Please edit the configuration file and run again.")
        return
    
    # Step 2: Train the model
    print("Starting QNA model training...")
    results = gptmed.train_from_config(config_path, device='auto')
    
    print(f"\\nTraining completed!")
    print(f"Best checkpoint: {results['best_checkpoint']}")
    print(f"Final validation loss: {results['final_val_loss']}")

if __name__ == "__main__":
    main()
""")
    
    # Create preprocess.py
    with open(os.path.join(project_name, "preprocess.py"), "w") as f:
        f.write("""\"\"\"
Data preprocessing for QNA dataset
\"\"\"
import json
from pathlib import Path

def preprocess_qna_data(input_file, output_file):
    \"\"\"
    Preprocess QNA data from raw format to training format.
    
    Expected input format (JSONL):
    {"question": "What is X?", "answer": "X is..."}
    
    Output format:
    {"text": "Q: What is X?\\nA: X is..."}
    \"\"\"
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            
            if question and answer:
                formatted_text = f"Q: {question}\\nA: {answer}"
                processed_data.append({"text": formatted_text})
    
    # Save processed data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\\n')
    
    print(f"Processed {len(processed_data)} QNA pairs")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    preprocess_qna_data(
        'data/raw/qna_data.jsonl',
        'data/processed/train.jsonl'
    )
""")
    
    # Create inference script
    with open(os.path.join(project_name, "inference", "generate_answer.py"), "w") as f:
        f.write("""\"\"\"
Generate answers using trained QNA model
\"\"\"
import gptmed
from pathlib import Path

def generate_answer(question, checkpoint_path, tokenizer_path, max_length=200):
    \"\"\"Generate answer for a given question\"\"\"
    
    # Format question
    prompt = f"Q: {question}\\nA:"
    
    # Generate answer
    answer = gptmed.generate(
        checkpoint=checkpoint_path,
        prompt=prompt,
        tokenizer=tokenizer_path,
        max_length=max_length,
        temperature=0.7,
        top_k=50
    )
    
    # Extract answer (remove the prompt)
    answer = answer.replace(prompt, '').strip()
    
    return answer

if __name__ == "__main__":
    # Example usage
    checkpoint = "models/checkpoints/best_model.pt"
    tokenizer = "tokenizer/qna_tokenizer.model"
    
    question = "What is machine learning?"
    answer = generate_answer(question, checkpoint, tokenizer)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
""")
    
    # Create README.md
    with open(os.path.join(project_name, "README.md"), "w") as f:
        f.write(f"""# {project_name} - QNA Model

Question and Answer generation model training architecture.

## Directory Structure

```
{project_name}/
├── configs/              # Training configurations
├── data/
│   ├── raw/             # Raw QNA data
│   └── processed/       # Preprocessed data
├── models/
│   └── checkpoints/     # Model checkpoints
├── tokenizer/           # Tokenizer files
├── logs/                # Training logs
├── inference/           # Inference scripts
├── main.py              # Main training script
├── preprocess.py        # Data preprocessing
└── README.md
```

## Getting Started

### 1. Prepare Data

Place your QNA data in JSONL format in `data/raw/qna_data.jsonl`:

```json
{{"question": "What is X?", "answer": "X is..."}}
{{"question": "How does Y work?", "answer": "Y works by..."}}
```

### 2. Preprocess Data

```bash
python preprocess.py
```

### 3. Configure Training

```bash
python main.py  # This will create a default config file
```

Edit `configs/training_config.yaml` with your settings.

### 4. Train Model

```bash
python main.py
```

### 5. Generate Answers

```bash
python inference/generate_answer.py
```

## Training Configuration

Edit `configs/training_config.yaml` to customize:
- Model size (tiny, small, medium)
- Training parameters (epochs, batch size, learning rate)
- Data paths
- Device selection (CPU/GPU)

## Inference

Use the trained model to generate answers:

```python
import gptmed

answer = gptmed.generate(
    checkpoint='models/checkpoints/best_model.pt',
    prompt='Q: Your question?\\nA:',
    tokenizer='tokenizer/qna_tokenizer.model'
)
```
""")

def create_conversational_templates(project_name):
    """Create boilerplate for conversational model training architecture"""
    
    # Create directory structure
    os.makedirs(os.path.join(project_name, "configs"))
    os.makedirs(os.path.join(project_name, "data", "raw"))
    os.makedirs(os.path.join(project_name, "data", "processed"))
    os.makedirs(os.path.join(project_name, "models", "checkpoints"))
    os.makedirs(os.path.join(project_name, "tokenizer"))
    os.makedirs(os.path.join(project_name, "logs"))
    os.makedirs(os.path.join(project_name, "inference"))
    os.makedirs(os.path.join(project_name, "utils"))
    
    # Create main.py
    with open(os.path.join(project_name, "main.py"), "w") as f:
        f.write("""\"\"\"
Main entry point for Conversational Model Training
\"\"\"
import gptmed
from pathlib import Path

def main():
    # Step 1: Create configuration
    config_path = 'configs/training_config.yaml'
    if not Path(config_path).exists():
        gptmed.create_config(config_path)
        print(f"Configuration file created at {config_path}")
        print("Please edit the configuration file and run again.")
        return
    
    # Step 2: Train the model
    print("Starting conversational model training...")
    results = gptmed.train_from_config(config_path, device='auto')
    
    print(f"\\nTraining completed!")
    print(f"Best checkpoint: {results['best_checkpoint']}")
    print(f"Final validation loss: {results['final_val_loss']}")

if __name__ == "__main__":
    main()
""")
    
    # Create preprocess.py for conversational data
    with open(os.path.join(project_name, "preprocess.py"), "w") as f:
        f.write("""\"\"\"
Data preprocessing for conversational dataset
\"\"\"
import json
from pathlib import Path
from typing import List, Dict

def format_conversation(messages: List[Dict[str, str]], 
                       user_token="<|user|>", 
                       assistant_token="<|assistant|>",
                       end_token="<|endoftext|>") -> str:
    \"\"\"
    Format a conversation into training text.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        user_token: Token to mark user messages
        assistant_token: Token to mark assistant messages
        end_token: Token to mark end of conversation
    
    Returns:
        Formatted conversation string
    \"\"\"
    conversation = []
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '').strip()
        
        if role == 'user':
            conversation.append(f"{user_token} {content}")
        elif role == 'assistant':
            conversation.append(f"{assistant_token} {content}")
    
    return "\\n".join(conversation) + f"\\n{end_token}"

def preprocess_conversational_data(input_file, output_file):
    \"\"\"
    Preprocess conversational data from raw format to training format.
    
    Expected input format (JSONL):
    {
        "conversation": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Tell me about AI"},
            {"role": "assistant", "content": "AI is..."}
        ]
    }
    
    Output format:
    {"text": "<|user|> Hello!\\n<|assistant|> Hi! How can I help?\\n..."}
    \"\"\"
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            conversation = item.get('conversation', [])
            
            if len(conversation) >= 2:  # At least one exchange
                formatted_text = format_conversation(conversation)
                processed_data.append({"text": formatted_text})
    
    # Save processed data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\\n')
    
    print(f"Processed {len(processed_data)} conversations")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    preprocess_conversational_data(
        'data/raw/conversations.jsonl',
        'data/processed/train.jsonl'
    )
""")
    
    # Create conversation handler utility
    with open(os.path.join(project_name, "utils", "conversation_handler.py"), "w") as f:
        f.write("""\"\"\"
Conversation management utilities
\"\"\"
from typing import List, Dict, Optional

class ConversationHistory:
    \"\"\"Manage conversation history for multi-turn dialogues\"\"\"
    
    def __init__(self, max_history: int = 10):
        \"\"\"
        Initialize conversation history.
        
        Args:
            max_history: Maximum number of turns to keep in history
        \"\"\"
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history
    
    def add_user_message(self, content: str):
        \"\"\"Add a user message to history\"\"\"
        self.messages.append({"role": "user", "content": content})
        self._trim_history()
    
    def add_assistant_message(self, content: str):
        \"\"\"Add an assistant message to history\"\"\"
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()
    
    def _trim_history(self):
        \"\"\"Keep only the most recent messages\"\"\"
        if len(self.messages) > self.max_history * 2:  # 2 messages per turn
            self.messages = self.messages[-(self.max_history * 2):]
    
    def get_prompt(self, 
                   user_token: str = "<|user|>",
                   assistant_token: str = "<|assistant|>") -> str:
        \"\"\"
        Generate prompt from conversation history.
        
        Returns:
            Formatted conversation prompt
        \"\"\"
        prompt_parts = []
        
        for msg in self.messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                prompt_parts.append(f"{user_token} {content}")
            elif role == 'assistant':
                prompt_parts.append(f"{assistant_token} {content}")
        
        # Add assistant token for next response
        prompt_parts.append(assistant_token)
        
        return "\\n".join(prompt_parts) + " "
    
    def clear(self):
        \"\"\"Clear conversation history\"\"\"
        self.messages = []
    
    def get_last_n_turns(self, n: int) -> List[Dict[str, str]]:
        \"\"\"Get last n conversation turns\"\"\"
        return self.messages[-(n * 2):]
""")
    
    # Create interactive chat script
    with open(os.path.join(project_name, "inference", "interactive_chat.py"), "w") as f:
        f.write("""\"\"\"
Interactive chat with conversational model
\"\"\"
import sys
sys.path.append('..')

import gptmed
from pathlib import Path
from utils.conversation_handler import ConversationHistory

class ChatBot:
    \"\"\"Interactive chatbot using trained conversational model\"\"\"
    
    def __init__(self, checkpoint_path: str, tokenizer_path: str):
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.history = ConversationHistory(max_history=5)
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.end_token = "<|endoftext|>"
    
    def generate_response(self, user_input: str, max_length: int = 150) -> str:
        \"\"\"Generate response to user input\"\"\"
        
        # Add user message to history
        self.history.add_user_message(user_input)
        
        # Get prompt from conversation history
        prompt = self.history.get_prompt(self.user_token, self.assistant_token)
        
        # Generate response
        response = gptmed.generate(
            checkpoint=self.checkpoint_path,
            prompt=prompt,
            tokenizer=self.tokenizer_path,
            max_length=max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        # Extract assistant response
        response = response.replace(prompt, '').strip()
        
        # Remove end token if present
        if self.end_token in response:
            response = response.split(self.end_token)[0].strip()
        
        # Remove user token if model generated it (shouldn't happen but just in case)
        if self.user_token in response:
            response = response.split(self.user_token)[0].strip()
        
        # Add assistant response to history
        self.history.add_assistant_message(response)
        
        return response
    
    def chat(self):
        \"\"\"Start interactive chat session\"\"\"
        print("Conversational AI Chatbot")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'clear' to reset conversation history")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.history.clear()
                    print("Conversation history cleared.")
                    continue
                
                response = self.generate_response(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    # Initialize chatbot
    checkpoint = "../models/checkpoints/best_model.pt"
    tokenizer = "../tokenizer/conv_tokenizer.model"
    
    if not Path(checkpoint).exists():
        print(f"Error: Checkpoint not found at {checkpoint}")
        print("Please train your model first using main.py")
        sys.exit(1)
    
    chatbot = ChatBot(checkpoint, tokenizer)
    chatbot.chat()
""")
    
    # Create batch inference script
    with open(os.path.join(project_name, "inference", "batch_inference.py"), "w") as f:
        f.write("""\"\"\"
Batch inference for conversational model
\"\"\"
import sys
sys.path.append('..')

import json
import gptmed
from pathlib import Path
from utils.conversation_handler import ConversationHistory
from typing import List, Dict

def generate_conversation_response(
    conversation_history: List[Dict[str, str]],
    checkpoint_path: str,
    tokenizer_path: str,
    max_length: int = 150
) -> str:
    \"\"\"
    Generate response for a conversation.
    
    Args:
        conversation_history: List of messages with role and content
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        max_length: Maximum response length
    
    Returns:
        Generated response
    \"\"\"
    history = ConversationHistory()
    
    # Rebuild conversation history
    for msg in conversation_history:
        if msg['role'] == 'user':
            history.add_user_message(msg['content'])
        elif msg['role'] == 'assistant':
            history.add_assistant_message(msg['content'])
    
    # Get prompt
    prompt = history.get_prompt()
    
    # Generate response
    response = gptmed.generate(
        checkpoint=checkpoint_path,
        prompt=prompt,
        tokenizer=tokenizer_path,
        max_length=max_length,
        temperature=0.8,
        top_k=50
    )
    
    # Clean up response
    response = response.replace(prompt, '').strip()
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0].strip()
    
    return response

def batch_process_conversations(
    input_file: str,
    output_file: str,
    checkpoint_path: str,
    tokenizer_path: str
):
    \"\"\"
    Process multiple conversations in batch.
    
    Input format (JSONL):
    {"conversation": [{"role": "user", "content": "Hi"}, ...]}
    
    Output format (JSONL):
    {"conversation": [...], "generated_response": "..."}
    \"\"\"
    results = []
    
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            conversation = item.get('conversation', [])
            
            if conversation:
                response = generate_conversation_response(
                    conversation,
                    checkpoint_path,
                    tokenizer_path
                )
                
                results.append({
                    "conversation": conversation,
                    "generated_response": response
                })
    
    # Save results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\\n')
    
    print(f"Processed {len(results)} conversations")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    batch_process_conversations(
        input_file='test_conversations.jsonl',
        output_file='results.jsonl',
        checkpoint_path='../models/checkpoints/best_model.pt',
        tokenizer_path='../tokenizer/conv_tokenizer.model'
    )
""")
    
    # Create README.md
    with open(os.path.join(project_name, "README.md"), "w") as f:
        f.write(f"""# {project_name} - Conversational Model

Multi-turn conversational language model training architecture.

## Directory Structure

```
{project_name}/
├── configs/              # Training configurations
├── data/
│   ├── raw/             # Raw conversation data
│   └── processed/       # Preprocessed data
├── models/
│   └── checkpoints/     # Model checkpoints
├── tokenizer/           # Tokenizer files
├── logs/                # Training logs
├── inference/           # Inference scripts
│   ├── interactive_chat.py   # Interactive chat interface
│   └── batch_inference.py    # Batch processing
├── utils/               # Utility modules
│   └── conversation_handler.py
├── main.py              # Main training script
├── preprocess.py        # Data preprocessing
└── README.md
```

## Getting Started

### 1. Prepare Data

Place your conversational data in JSONL format in `data/raw/conversations.jsonl`:

```json
{{
  "conversation": [
    {{"role": "user", "content": "Hello!"}},
    {{"role": "assistant", "content": "Hi! How can I help you today?"}},
    {{"role": "user", "content": "Tell me about AI"}},
    {{"role": "assistant", "content": "AI stands for Artificial Intelligence..."}}
  ]
}}
```

### 2. Preprocess Data

```bash
python preprocess.py
```

This will format conversations with special tokens:
- `<|user|>` - marks user messages
- `<|assistant|>` - marks assistant messages
- `<|endoftext|>` - marks end of conversation

### 3. Configure Training

```bash
python main.py  # Creates default config
```

Edit `configs/training_config.yaml` with your settings.

### 4. Train Model

```bash
python main.py
```

### 5. Interactive Chat

```bash
cd inference
python interactive_chat.py
```

Commands:
- Type your message and press Enter
- Type `clear` to reset conversation history
- Type `quit` or `exit` to end chat

### 6. Batch Inference

```bash
cd inference
python batch_inference.py
```

## Features

### Multi-turn Conversations
The model maintains conversation history and generates contextually relevant responses.

### Conversation Management
The `ConversationHistory` class manages:
- Message history tracking
- Automatic history trimming
- Prompt generation from history

### Interactive Chat
Real-time chat interface with:
- Multi-turn conversation support
- History management
- User-friendly commands

### Batch Processing
Process multiple conversations:
- Evaluate model on test sets
- Generate responses for datasets
- Performance benchmarking

## Conversation Format

The model uses special tokens to structure conversations:

```
<|user|> Hello, how are you?
<|assistant|> I'm doing well, thank you! How can I assist you today?
<|user|> I need help with Python
<|assistant|> I'd be happy to help with Python! What specific topic?
<|endoftext|>
```

## Training Tips

1. **Data Quality**: Ensure conversations are natural and coherent
2. **History Length**: Adjust `max_history` based on your use case
3. **Temperature**: Lower (0.6-0.8) for focused responses, higher (0.8-1.0) for creative
4. **Model Size**: Start with tiny/small, scale up as needed

## Inference Parameters

Adjust generation parameters in inference scripts:

```python
response = gptmed.generate(
    checkpoint=checkpoint_path,
    prompt=prompt,
    tokenizer=tokenizer_path,
    max_length=150,      # Maximum response length
    temperature=0.8,     # Randomness (0.0-1.0)
    top_k=50,            # Top-k sampling
    top_p=0.9            # Nucleus sampling
)
```

## Example Usage

```python
from utils.conversation_handler import ConversationHistory
import gptmed

# Initialize conversation
history = ConversationHistory(max_history=5)

# Add messages
history.add_user_message("What is machine learning?")

# Generate prompt
prompt = history.get_prompt()

# Generate response
response = gptmed.generate(
    checkpoint='models/checkpoints/best_model.pt',
    prompt=prompt,
    tokenizer='tokenizer/conv_tokenizer.model'
)

# Add response to history
history.add_assistant_message(response)
```
""")

def create_basic_project(project_name):
    """Create basic project structure (original behavior)"""
    os.makedirs(os.path.join(project_name, "configs"))
    os.makedirs(os.path.join(project_name, "tasks"))
    os.makedirs(os.path.join(project_name, "models"))
    os.makedirs(os.path.join(project_name, "data"))
    with open(os.path.join(project_name, "main.py"), "w") as f:
        f.write("import gptmed\n\n# Your project entrypoint\n")

def startproject(project_name, project_type=None):
    """
    Create a new gptmed project.
    
    Args:
        project_name: Name of the project
        project_type: Type of project ('qna', 'conversational', or None for basic)
    """
    if not project_name.isidentifier():
        print("Invalid project name. Your project name must be a valid Python identifier. "
              "Do not use hyphens or spaces. Use underscores instead.")
        sys.exit(1)
    
    if os.path.exists(project_name):
        print(f"Directory '{project_name}' already exists.")
        sys.exit(1)
    
    # Create project based on type
    if project_type == "qna":
        create_qna_templates(project_name)
        print(f"QNA project '{project_name}' created successfully!")
        print(f"\nNext steps:")
        print(f"1. cd {project_name}")
        print(f"2. Place your QNA data in data/raw/qna_data.jsonl")
        print(f"3. Run: python preprocess.py")
        print(f"4. Run: python main.py")
        
    elif project_type == "conversational":
        create_conversational_templates(project_name)
        print(f"Conversational project '{project_name}' created successfully!")
        print(f"\nNext steps:")
        print(f"1. cd {project_name}")
        print(f"2. Place your conversation data in data/raw/conversations.jsonl")
        print(f"3. Run: python preprocess.py")
        print(f"4. Run: python main.py")
        print(f"5. For interactive chat: cd inference && python interactive_chat.py")
        
    else:
        create_basic_project(project_name)
        print(f"Project '{project_name}' created.")

