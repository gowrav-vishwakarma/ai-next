import torch
import numpy as np
from sample_qllm import EnhancedQuantumLLM

def load_model(checkpoint_path):
    # Configuration matching both the loaded model and tokenizer
    config = {
        'vocab_size': 30522,  # Match BERT tokenizer vocab size
        'dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'max_sequence_length': 512,
        'tokenizer_name': 'bert-base-uncased'
    }
    
    print(f"Initializing model with config: {config}")
    model = EnhancedQuantumLLM(config)
    
    try:
        # Load PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        model.token_embeddings = torch.nn.Parameter(checkpoint['model_state']['token_embeddings'].to("mps"))
        model.position_embeddings = torch.nn.Parameter(checkpoint['model_state']['position_embeddings'].to("mps"))
        
        print("\nModel loaded successfully!")
        print(f"Token embeddings shape: {model.token_embeddings.shape}")
        print(f"Position embeddings shape: {model.position_embeddings.shape}")
        print(f"Token embeddings type: {model.token_embeddings.dtype}")
        print(f"Position embeddings type: {model.position_embeddings.dtype}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise
    
    return model

def generate(model, prompt, max_length=100):
    print(f"\nGenerating text for prompt: {prompt}")
    try:
        # Convert prompt to tensor indices
        input_ids = model.tokenizer.encode(prompt)
        print(f"Encoded input shape: {len(input_ids)}")
        print(f"Input tokens: {input_ids}")
        
        # Convert to tensor and move to MPS
        input_ids = torch.tensor(input_ids, dtype=torch.long, device="mps")
        
        # Generate text
        generated_text = model.generate(prompt, max_length=max_length)
        print(f"\nGenerated text:\n{generated_text}")
        return generated_text
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        print(f"Token embeddings device: {model.token_embeddings.device}")
        print(f"Input ids device: {input_ids.device}")
        raise

if __name__ == "__main__":
    print("Loading model from checkpoint...")
    model = load_model('quantum_model_checkpoints/checkpoint_epoch_3.pt')
    
    prompts = [
        "Quantum computing is",
        "The future of AI",
        "In the quantum realm",
    ]
    
    print("\nStarting text generation...")
    for prompt in prompts:
        generate(model, prompt) 