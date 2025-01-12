import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import time
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import torch.backends.mps
import gc
import argparse
import sys
from quantum_language_core import QuantumTokenizer, QuantumLanguageStructure

# Determine the best available device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Update environment variables for MPS if applicable
if device == "mps":
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.5'  # Use 50% of available memory
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.3'   # Free memory when usage goes below 30%

class FastQuantumOps:
    """Stable quantum operations using bounded functions"""
    
    @staticmethod
    def quantum_interference(x, y):
        """Stable quantum interference using bounded operations"""
        # Use hyperbolic tangent for bounded mixing
        mixed = torch.tanh((x + y) * 0.5)
        # Use cosine similarity for interference
        similarity = F.cosine_similarity(x, y, dim=-1, eps=1e-8).unsqueeze(-1)
        return mixed * similarity

    @staticmethod
    def phase_encoding(x):
        """Encode information in phases using bounded operations"""
        # Normalize input
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        # Convert to phase space using arctan2
        phase = torch.atan2(x_norm, torch.roll(x_norm, 1, dims=-1))
        return torch.sin(phase)

class FastQuantumAttention(nn.Module):
    """Optimized quantum-inspired attention"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Pre-compute interference patterns
        self.register_buffer(
            "patterns", 
            self._init_patterns()
        )
    
    def _init_patterns(self):
        """Initialize using stable operations"""
        t = torch.linspace(0, 2 * torch.pi, self.head_dim).to(device)
        patterns = []
        
        for h in range(self.num_heads):
            phase = 2 * torch.pi * h / self.num_heads
            # Create phase tensor and add to t
            phase_tensor = torch.full_like(t, phase)
            pattern = FastQuantumOps.phase_encoding(t + phase_tensor)  # Use phase encoding
            patterns.append(pattern)
            
        return torch.stack(patterns)
    
    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        
        # Reshape and scale
        x = x.view(B, L, H, -1)
        
        # Apply patterns using broadcasting
        x = x * self.patterns.view(1, 1, H, -1)
        
        # Fast approximate attention
        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        
        return torch.matmul(attn, x).view(B, L, D)

class FastQuantumState:
    """Simulate quantum states using height-map inspired techniques"""
    
    @staticmethod
    def encode_state(x, dim):
        """Memory-efficient state encoding"""
        B, L, D = x.shape
        
        # Process in smaller chunks if needed
        chunk_size = min(L, 64)  # Process 64 tokens at a time
        outputs = []
        
        for i in range(0, L, chunk_size):
            # Get chunk and ensure it's contiguous
            chunk = x[:, i:i+chunk_size, :].contiguous()
            curr_chunk_size = chunk.size(1)  # Actual size of this chunk
            
            # Process chunk
            h = torch.linspace(0, 1, D).to(device)
            v = torch.linspace(0, 1, D).to(device)
            
            h_pattern = FastQuantumOps.fast_sin(h * torch.pi)
            v_pattern = FastQuantumOps.fast_sin(v * torch.pi)
            
            # Reshape properly
            chunk_flat = chunk.reshape(B * curr_chunk_size, D)
            
            # Project through patterns
            h_proj = torch.matmul(chunk_flat, h_pattern.unsqueeze(1))  # [B*chunk_size, 1]
            v_proj = torch.matmul(chunk_flat, v_pattern.unsqueeze(1))  # [B*chunk_size, 1]
            
            # Expand projections
            h_proj = h_proj.expand(-1, D)  # [B*chunk_size, D]
            v_proj = v_proj.expand(-1, D)  # [B*chunk_size, D]
            
            # Combine projections
            output = (h_proj + v_proj) / 2.0
            
            # Reshape back to [B, chunk_size, D]
            output = output.reshape(B, curr_chunk_size, D)
            
            outputs.append(output)
            
            # Clear intermediate tensors
            del h_pattern, v_pattern, h_proj, v_proj, chunk_flat
        
        # Combine chunks
        output = torch.cat(outputs, dim=1)
        
        # Add residual connection and normalize
        output = output + x
        output = output / (torch.norm(output, dim=-1, keepdim=True) + 1e-8)
        
        return output

class FastQuantumLLM(nn.Module):
    """Optimized quantum-inspired language model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = device # Use the global device variable
        
        # Initialize with smaller values
        self.scale = 1.0 / math.sqrt(config['dim'])
        
        # Ensure vocab_size is set
        if 'vocab_size' not in config:
            config['vocab_size'] = 32000  # Default size if not specified
        
        # Initialize quantum tokenizer
        self.tokenizer = QuantumTokenizer(vocab_size=config['vocab_size'])
        
        # Add quantum language structure
        self.language_structure = QuantumLanguageStructure()
        
        # Embeddings with proper initialization
        self.token_embedding = nn.Embedding(
            config['vocab_size'], 
            config['dim']
        )
        # Initialize embeddings with smaller values
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.scale)
        
        # Initialize positional embeddings with smaller values
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config['max_sequence_length'], config['dim']) * self.scale
        ).to(self.device)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            FastQuantumAttention(config['dim'], config['num_heads'])
            for _ in range(config['num_layers'])
        ]).to(self.device)
        
        # Output projection
        self.output_proj = nn.Linear(config['dim'], config['vocab_size']).to(self.device)
        
        # Quantum state encoder
        self.quantum_encoder = FastQuantumState()
        
        # For tracking training
        self.step_count = 0
        
        # Add layer norm for stability
        self.layer_norm = nn.LayerNorm(config['dim']).to(self.device)
        
        # Add additional normalization layers
        self.input_norm = nn.LayerNorm(config['dim']).to(self.device)
        self.output_norm = nn.LayerNorm(config['dim']).to(self.device)
        
    def forward(self, input_ids, return_loss=True):
        """Forward pass with guaranteed stability"""
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.long().to(self.device)
        
        B, L = input_ids.shape
        
        # Get embeddings with bounded initialization
        x = torch.tanh(self.token_embedding(input_ids) * 0.1)
        
        # Add positional information using phases
        positions = torch.arange(L, device=self.device).float()
        pos_phase = torch.sin(positions.unsqueeze(1) * self.scale)
        x = x + pos_phase.unsqueeze(0) * 0.1
        
        # Apply layer norm for stability
        x = self.input_norm(x)
        
        # Process through attention layers with bounded operations
        for layer in self.attention_layers:
            residual = x
            
            # Apply attention with scaling
            attn = layer(x) * 0.1
            
            # Mix using quantum interference
            x = FastQuantumOps.quantum_interference(residual, attn)
            
            # Ensure values stay bounded
            x = torch.tanh(x)
            
            # Add layer norm
            x = self.layer_norm(x)
        
        # Final normalization
        x = self.output_norm(x)
        
        # Project to vocabulary with bounded activation
        logits = torch.tanh(self.output_proj(x))
        
        if return_loss and input_ids is not None:
            # Compute stable loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Use stable cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1,
                reduction='mean'
            )
            
            return loss
        
        return logits
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_k=50):
        """Generate text using the model"""
        # Encode with quantum tokenizer (using index mode)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        generated = input_ids[0].tolist()
        
        # Generate tokens
        for _ in range(max_length):
            # Get predictions
            inputs = torch.tensor([generated[-self.config['max_sequence_length']:]]).to(self.device)
            with torch.no_grad():
                outputs = self.forward(inputs, return_loss=False)
                next_token_logits = outputs[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Convert to probabilities
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample next token
            next_token_id = top_k_indices[torch.multinomial(probs, num_samples=1)]
            
            generated.append(next_token_id.item())
            
            # Stop if we exceed max length or generate EOS token
            if len(generated) >= max_length or next_token_id == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated)

# Add new dataset handling classes
class TextDataset(Dataset):
    """Custom dataset for handling text data with quantum tokenizer"""
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = list(dataset)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        
        # Use the tokenizer's __call__ method
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# Add a memory management function
def clear_memory():
    """Clear memory caches"""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

# Update the train_model function to use the new memory management
def train_model(model, config):
    """Train the model using HuggingFace dataset with memory optimizations"""
    print("Loading dataset...")
    clear_memory()  # Clear memory before starting
    
    # Load dataset with memory efficiency
    dataset = load_dataset(
        config['dataset_name'],
        config['dataset_config'],
        split='train',
        streaming=True
    )
    
    if config.get('max_samples'):
        dataset = dataset.take(config['max_samples'])
    
    # Create training dataset
    train_dataset = TextDataset(
        dataset,
        model.tokenizer,
        config['max_sequence_length']
    )
    
    # Create data loader with smaller num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,  # Use smaller learning rate
        weight_decay=0.0,  # Remove weight decay
        eps=1e-8
    )
    
    # Training loop with memory optimizations
    model.train()
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            try:
                input_ids = batch['input_ids'].to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                loss = model(input_ids)
                
                # Validate loss
                if torch.isfinite(loss) and loss.item() > 0:
                    # Use gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1  # Increment only for valid loss
                    
                    if batch_idx % config['log_every'] == 0:
                        print(f"\nBatch {batch_idx}, Loss: {loss.item():.4f}")
                else:
                    print(f"Skipping batch {batch_idx} - invalid loss: {loss.item()}")
                
                clear_memory()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                clear_memory()
                continue
        
        # Check if any batches were processed
        if num_batches > 0:
            avg_epoch_loss = total_loss / num_batches
            print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        else:
            print(f"\nEpoch {epoch+1} completed. No valid batches processed.")
        
        # Clear memory between epochs
        clear_memory()

def save_checkpoint(model, optimizer, epoch, output_dir):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    path = Path(output_dir)
    path.mkdir(exist_ok=True)
    torch.save(checkpoint, path / f'checkpoint_epoch_{epoch}.pt')

if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='''
    Quantum LLM - Train or Generate
    
    Examples:
        # Train the model:
        python sample_qllm_optimized.py --mode train
        
        # Generate text with default settings:
        python sample_qllm_optimized.py --mode generate --prompt "Once in India"
        
        # Generate with a checkpoint:
        python sample_qllm_optimized.py --mode generate --checkpoint quantum_checkpoints/checkpoint_epoch_2.pt --prompt "The future"
    ''', formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--mode', type=str, choices=['train', 'generate'],
                      help='Mode to run: train or generate')
    parser.add_argument('--prompt', type=str,
                      help='Prompt for text generation (required for generate mode)')
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint file for generation')
    parser.add_argument('--max_length', type=int, default=50,
                      help='Maximum length of generated text (default: 50)')
    parser.add_argument('--temperature', type=float, default=0.1,
                      help='Temperature for text generation (default: 0.1)')
    
    # Show help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'generate' and not args.prompt:
        parser.error("--prompt is required when mode is 'generate'")
    
    # Configuration
    config = {
        # Add vocab_size at the top of config
        'vocab_size': 32000,  # Standard vocabulary size
        
        # Dimension of the model's embeddings and hidden states
        # Higher = more expressive but more memory/compute intensive
        # Lower = faster but less capable
        # Range: 64-4096, common values: 128, 256, 512
        'dim': 128,

        # Number of attention heads for parallel processing
        # Higher = better parallel processing but more memory
        # Lower = less memory but might miss some patterns
        # Usually a power of 2: 2, 4, 8, 16
        # Should divide 'dim' evenly
        'num_heads': 8,

        # Number of transformer layers in the model
        # Higher = deeper understanding but slower and more memory
        # Lower = faster but less sophisticated
        # Range: 2-48, common values: 4, 6, 12
        'num_layers': 4,

        # Maximum length of input sequences
        # Higher = longer context but quadratic memory increase
        # Lower = less memory but can't handle long texts
        # Range: 128-4096, common values: 256, 512, 1024
        'max_sequence_length': 256,

        # Number of samples processed in parallel
        # Higher = faster training but more memory
        # Lower = less memory but slower training
        # Range: 1-32, depends heavily on GPU memory
        'batch_size': 2,

        # Number of complete passes through the training data
        # Higher = better learning but takes longer
        # Lower = faster but might underfit
        # Range: 3-100, depends on dataset size
        'epochs': 3,

        # Rate at which the model learns
        # Higher = faster learning but might be unstable
        # Lower = more stable but slower learning
        # Common range: 1e-5 to 1e-3
        'learning_rate': 1e-4,

        # Directory for saving model checkpoints
        'output_dir': 'quantum_checkpoints',

        # Pre-trained tokenizer to use
        'tokenizer_name': 'bert-base-uncased',

        # Training dataset name from HuggingFace
        'dataset_name': 'wikitext',

        # Specific configuration of the dataset
        # wikitext-103-v1 is larger than wikitext-2-v1
        'dataset_config': 'wikitext-103-v1',

        # Number of samples to use from dataset
        # Higher = better learning but slower training
        # Lower = faster but might not learn as well
        'max_samples': 200,

        # How often to print training progress
        # Lower = more frequent updates but slightly slower
        'log_every': 5,

        # Maximum gradient norm for stability
        # Higher = might be unstable but faster learning
        # Lower = more stable but might learn slower
        'gradient_clip': 0.5,

        # Number of batches to accumulate before updating
        # Higher = larger effective batch size, less memory
        # Lower = more frequent updates but might be unstable
        # Range: 1-16, helps simulate larger batch sizes
        'accumulation_steps': 8
    }
    
    # Initialize model with proper order
    model = FastQuantumLLM(config)  # First create model
    model = model.to(model.device)  # Then move to device
    
    if args.mode == 'train':
        # Train model using HuggingFace dataset
        train_model(model, config)
    else:
        # Load checkpoint if provided
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if checkpoint_path.exists():
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
        
        # Generate text
        print(f"\nGenerating text with prompt: {args.prompt}")
        print(f"Temperature: {args.temperature}, Max Length: {args.max_length}")
        
        generated_text = model.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=100
        )
        print(f"\nGenerated text:\n{generated_text}") 