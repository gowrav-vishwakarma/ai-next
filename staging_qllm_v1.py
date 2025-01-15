import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import os
import gc
from tqdm import tqdm
import time

# Determine the best available device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Update environment variables for MPS if applicable
if device == "mps":
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'  # Allow using more memory
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'   # Free memory more aggressively

def clear_memory():
    """Clear memory caches"""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

class SimpleQuantumState(nn.Module):
    """
    Simplified quantum state representation with just two layers:
    1. Ground state (basic meaning)
    2. Single excited state (contextual meaning)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Add layer norms
        self.input_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        
        # Even smaller initialization
        self.ground_transform = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.ground_transform.weight, gain=0.01)
        nn.init.zeros_(self.ground_transform.bias)
        
        self.excite_transform = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.excite_transform.weight, gain=0.01)
        nn.init.zeros_(self.excite_transform.bias)
        
        # Much smaller phase factor initialization
        self.phase_factor = nn.Parameter(torch.randn(dim) * 0.001)
        
        self.PHI = (1 + math.sqrt(5)) / 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        # Input normalization
        x = self.input_norm(x)
        x = x * 0.01  # Aggressive scaling
        
        # Process ground state with residual connection
        ground_state = self.ground_transform(x) + x
        
        # Very bounded phase
        phase = torch.tanh(self.phase_factor) * self.PHI * 0.01
        excited_state = self.excite_transform(x) + x
        
        # Combine states with scaling
        combined_real = ground_state + excited_state * torch.cos(phase)
        combined_imag = excited_state * torch.sin(phase)
        
        # Normalize with larger epsilon
        norm = torch.sqrt(combined_real.pow(2) + combined_imag.pow(2) + 1e-8)
        combined_real = combined_real / norm
        combined_imag = combined_imag / norm
        
        # Output normalization
        combined_real = self.output_norm(combined_real)
        combined_imag = self.output_norm(combined_imag)
        
        return combined_real, combined_imag

class BasicQuantumAttention(nn.Module):
    """
    Memory-efficient quantum attention using phase relationships
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        
        # Add more layer norms
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        
    def forward(self, q_real, q_imag, k_real, k_imag, v_real, v_imag):
        # Apply layer norm and scaling
        q_real = self.q_norm(q_real) * 0.01
        q_imag = self.q_norm(q_imag) * 0.01
        k_real = self.k_norm(k_real) * 0.01
        k_imag = self.k_norm(k_imag) * 0.01
        v_real = self.v_norm(v_real) * 0.01
        v_imag = self.v_norm(v_imag) * 0.01
        
        # Get dimensions
        B, L, D = q_real.shape
        
        # Process in smaller chunks
        chunk_size = min(L, 32)  # Smaller chunks
        output_real = torch.zeros(B, L, D, device=q_real.device)
        output_imag = torch.zeros(B, L, D, device=q_imag.device)
        
        for i in range(0, L, chunk_size):
            j = min(i + chunk_size, L)
            
            # Get current chunk
            q_real_chunk = q_real[:, i:j]
            q_imag_chunk = q_imag[:, i:j]
            
            # Compute attention scores with residual connection
            real_diff = (q_real_chunk.unsqueeze(2) - k_real.unsqueeze(1)) * self.scale
            imag_diff = (q_imag_chunk.unsqueeze(2) - k_imag.unsqueeze(1)) * self.scale
            
            # Sum with stability term
            real_sum = torch.sum(real_diff, dim=-1) + 1e-8
            imag_sum = torch.sum(imag_diff, dim=-1) + 1e-8
            
            # Compute stable phase difference
            phase_diff = torch.atan2(imag_sum, real_sum)
            
            # Compute stable attention weights
            attn_weights = torch.softmax(torch.cos(phase_diff) * 10.0, dim=-1)
            
            # Apply attention with scaled values
            output_real[:, i:j] = torch.bmm(attn_weights, v_real) * 0.1
            output_imag[:, i:j] = torch.bmm(attn_weights, v_imag) * 0.1
            
            # Clear memory
            del real_diff, imag_diff, phase_diff, attn_weights
            if device == "mps":
                torch.mps.empty_cache()
        
        # Final normalization
        output_real = self.output_norm(output_real)
        output_imag = self.output_norm(output_imag)
        
        return output_real, output_imag

class SimpleQuantumLLM(nn.Module):
    """
    Minimal quantum-inspired language model
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Initialize embeddings with smaller values
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        self.quantum_state = SimpleQuantumState(dim)
        self.attention = BasicQuantumAttention(dim)
        
        # Add layer norm before output
        self.pre_output_norm = nn.LayerNorm(dim * 2)
        self.output = nn.Linear(dim * 2, vocab_size)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings [batch_size, seq_len, dim]
        embed = self.embedding(x)
        
        # Process through quantum state
        state_real, state_imag = self.quantum_state(embed)
        
        # Ensure states have correct shape [batch_size, seq_len, dim]
        B, L, D = embed.shape
        state_real = state_real.view(B, L, D)
        state_imag = state_imag.view(B, L, D)
        
        # Apply attention
        attn_real, attn_imag = self.attention(
            state_real, state_imag,
            state_real, state_imag,
            state_real, state_imag
        )
        
        # Combine real and imaginary parts [batch_size, seq_len, dim*2]
        combined = torch.cat([attn_real, attn_imag], dim=-1)
        
        # Project to vocabulary [batch_size, seq_len, vocab_size]
        return self.output(combined)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # More aggressive clipping
        logits = torch.clamp(logits, -10, 10)
        
        # Use label smoothing and ignore padding
        ce_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            label_smoothing=0.1,
            ignore_index=0  # Assuming 0 is padding
        )
        
        # Remove phase coherence loss initially
        return ce_loss

def train_step(
    model: SimpleQuantumLLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    accumulation_steps: int = 4
) -> float:
    optimizer.zero_grad()
    
    # Even more aggressive gradient clipping
    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))
    
    try:
        logits = model(input_ids)
        loss = model.compute_loss(logits, target_ids) / accumulation_steps
        
        if not torch.isfinite(loss):
            print("Warning: Non-finite loss, skipping batch")
            return float('nan')
        
        # Skip if loss is too high
        if loss.item() > 100:
            print("Warning: Loss too high, skipping batch")
            return float('nan')
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()
        return loss.item() * accumulation_steps
        
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        return float('nan')

def generate(
    model: SimpleQuantumLLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7
) -> str:
    """
    Simple generation function with tokenization
    """
    model.eval()
    
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    with torch.no_grad():
        generated = input_ids
        
        for _ in range(max_length):
            # Get predictions
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append and continue
            generated = torch.cat([generated, next_token], dim=1)
            
            # Optional: Check for end token
            if next_token.item() == tokenizer.sep_token_id:
                break
        
        # Decode the generated tokens back to text
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text

def load_wikitext_dataset(max_samples=None):
    """Load and preprocess the Wikitext dataset"""
    dataset = load_dataset("wikitext", "wikitext-103-v1", split='train')
    
    if max_samples is not None:
        dataset = dataset.take(max_samples)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def preprocess_function(examples):
        # Tokenize with fixed sequence length
        encodings = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=512,  # Match model's dim
            return_tensors='pt'
        )
        
        # Ensure batch dimension
        if len(encodings['input_ids'].shape) == 1:
            encodings['input_ids'] = encodings['input_ids'].unsqueeze(0)
        
        return encodings
    
    # Show progress bar for dataset processing
    print("Processing dataset...")
    with tqdm(desc="Tokenizing", total=1) as pbar:
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=1,
            remove_columns=dataset.column_names,
            desc="Tokenizing"  # Removed 'leave' parameter
        )
        pbar.update(1)
    
    return tokenized_dataset

def main(args):
    # Initialize model with compatible dimensions
    config = {
        'vocab_size': 30522,  # BERT vocab size
        'dim': 512,          # Changed from 768 to match sequence length
        'num_heads': 8,
        'max_seq_length': 512
    }
    model = SimpleQuantumLLM(vocab_size=config['vocab_size'], dim=config['dim']).to(device)
    
    # Use AdamW with better defaults for stability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,  # Lower learning rate
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )

    if args.mode == 'train':
        print("Loading dataset...")
        dataset = load_wikitext_dataset(max_samples=args.max_samples)
        
        # Convert dataset to list for length calculation
        dataset = list(dataset)
        total_batches = len(dataset)
        
        print(f"\nStarting training...")
        print(f"Total batches: {total_batches}")
        
        # Training loop with progress bars
        for epoch in range(3):
            print(f"\nEpoch {epoch+1}/3")
            
            # Initialize statistics
            epoch_loss = 0
            valid_batches = 0
            start_time = time.time()
            
            # Create progress bar for this epoch
            progress_bar = tqdm(dataset, desc=f"Training", 
                              leave=True, 
                              total=total_batches)
            
            for batch in progress_bar:
                # Process batch
                input_ids = torch.tensor(batch['input_ids']).to(device)
                if len(input_ids.shape) == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                target_ids = torch.tensor(batch['input_ids']).to(device)
                if len(target_ids.shape) == 1:
                    target_ids = target_ids.unsqueeze(0)
                
                # Train step
                loss = train_step(model, optimizer, input_ids, target_ids, accumulation_steps=4)
                
                # Update statistics
                if not math.isnan(loss):
                    epoch_loss += loss
                    valid_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'avg_loss': f"{epoch_loss/valid_batches:.4f}",
                        'valid_batches': valid_batches
                    })
                
                clear_memory()
            
            # Epoch statistics
            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('nan')
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Valid Batches: {valid_batches}/{total_batches}")
            print(f"Time: {epoch_time:.2f}s")

    elif args.mode == 'generate':
        # Generate text from a prompt
        generated_text = generate(
            model, 
            AutoTokenizer.from_pretrained("bert-base-uncased"),
            args.prompt,
            max_length=args.max_length, 
            temperature=args.temperature
        )
        print("Generated text:", generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum-Inspired Language Model")
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True, help="Mode: train or generate")
    parser.add_argument('--prompt', type=str, help="Prompt for text generation")
    parser.add_argument('--max_length', type=int, default=100, help="Maximum length of generated text")
    parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument('--max_samples', type=int, default=100, help="Maximum number of samples to load from the dataset")

    args = parser.parse_args()
    main(args)