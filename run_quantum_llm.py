import numpy as np
from sample_qllm import EnhancedQuantumLLM

def main():
    # Configuration for a smaller test model with consistent dimensions
    config = {
        'vocab_size': 1000,
        'dim': 256,        # Must be divisible by num_heads
        'num_heads': 8,    # Changed to make dim/num_heads = 32 (head_dim)
        'num_layers': 4,
        'max_sequence_length': 512,
        'batch_size': 2,
        'epochs': 1,
        'learning_rate': 1e-4,
        'save_steps': 100
    }
    
    print("Initializing Quantum LLM...")
    model = EnhancedQuantumLLM(config)
    
    # Create sample training data
    print("Creating sample training data...")
    sample_text = """Quantum computing is an emerging technology that harnesses quantum mechanics.
    It uses quantum bits or qubits instead of classical bits.
    This allows for potentially exponential speedups in certain computations."""
    
    with open('sample_training.txt', 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    # Train model
    print("Starting training...")
    from sample_qllm import train_model
    train_model(
        model=model,
        train_data_path='sample_training.txt',
        output_dir='quantum_model_checkpoints',
        config=config
    )
    
    # Generate text
    print("\nGenerating text...")
    prompt = "Quantum computing"
    generated_text = model.generate(prompt, max_length=50)
    print(f"\nPrompt: {prompt}")
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main() 