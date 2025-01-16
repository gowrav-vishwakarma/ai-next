import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from typing import List, Tuple
import math
import gc
from tqdm import tqdm
import time
import argparse
from typing import List, Dict, Tuple, Optional
import os
import shutil
import pickle  # Add import for tokenizer serialization

from library.quantum_ml import BasicQuantumAttention, QuantumEmbedding, QuantumTokenizer, SimpleQuantumState, create_quantum_tokenizer

# Get best available device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        # Check if MPS is actually working
        try:
            tensor = torch.zeros(1).to("mps")
            return "mps"
        except:
            print("Warning: MPS device found but not working properly. Using CPU instead.")
            return "cpu"
    return "cpu"

device = get_device()

def clear_memory():
    """Clear memory caches safely"""
    try:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            # Only try to clear MPS memory if it's available
            try:
                torch.mps.empty_cache()
            except:
                pass
    except Exception as e:
        print(f"Warning: Memory clearing failed: {e}")

class QuantumLLM(nn.Module):
    """
    Updated quantum language model using phase-based tokenization
    """
    def __init__(self, tokenizer: QuantumTokenizer, dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.dim = dim
        
        # Quantum embedding layer
        self.embedding = QuantumEmbedding(tokenizer, dim)
        
        # Quantum state processing
        self.quantum_state = SimpleQuantumState(dim)
        self.attention = BasicQuantumAttention(dim)
        
        # Output projection
        self.pre_output_norm = nn.LayerNorm(dim * 2)
        self.output = nn.Linear(dim * 2, len(tokenizer))
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get quantum embeddings
        real_embed, imag_embed = self.embedding(x)
        
        # Process through quantum state
        state_real, state_imag = self.quantum_state(real_embed)
        
        # Apply attention
        attn_real, attn_imag = self.attention(
            state_real, state_imag,
            state_real, state_imag,
            state_real, state_imag
        )
        
        # Combine real and imaginary parts
        combined = torch.cat([attn_real, attn_imag], dim=-1)
        combined = self.pre_output_norm(combined)
        
        # Project to vocabulary
        return self.output(combined)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Clip logits for stability
        logits = torch.clamp(logits, -10, 10)
        
        # Cross entropy with label smoothing
        ce_loss = F.cross_entropy(
            logits.view(-1, len(self.tokenizer)),
            targets.view(-1),
            label_smoothing=0.1,
            ignore_index=self.tokenizer.vocab[self.tokenizer.PAD_token]
        )
        
        return ce_loss
    
    def to(self, device):
        """Override to method to ensure all components move to device"""
        super().to(device)
        self.embedding = self.embedding.to(device)
        self.quantum_state = self.quantum_state.to(device)
        self.attention = self.attention.to(device)
        self.pre_output_norm = self.pre_output_norm.to(device)
        self.output = self.output.to(device)
        return self

def load_quantum_wikitext(max_samples: Optional[int] = None):
    """Load and preprocess Wikitext with quantum tokenizer"""
    # Load raw dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split='train')
    
    if max_samples is not None:
        dataset = dataset.take(max_samples)
    
    # Extract texts
    texts = [example['text'] for example in dataset]
    
    # Create and train tokenizer
    print("Creating quantum tokenizer...")
    tokenizer = create_quantum_tokenizer(texts, dim=64)
    
    def preprocess_function(examples):
        # Tokenize with quantum tokenizer
        encodings = tokenizer.encode(examples['text'])
        
        # Ensure consistent length
        if len(encodings) > 512:
            encodings = encodings[:512]
        else:
            pad_length = 512 - len(encodings)
            encodings = torch.cat([
                encodings,
                torch.full((pad_length,), tokenizer.vocab[tokenizer.PAD_token])
            ])
        
        # Convert to tensor explicitly
        return {'input_ids': torch.tensor(encodings, dtype=torch.long)}
    
    # Process dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Convert to torch format
    tokenized_dataset.set_format(type='torch', columns=['input_ids'])
    
    return tokenized_dataset, tokenizer

def save_checkpoint(model: QuantumLLM, optimizer, epoch: int, loss: float, args, path: str = "checkpoints"):
    """Save model checkpoint"""
    import os
    import pickle
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Create checkpoint filename with epoch and loss
    checkpoint_file = os.path.join(path, f"model_epoch_{epoch}_loss_{loss:.4f}.pt")
    
    # Move model state to CPU before saving
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(model_state, checkpoint_file + '.model')
    
    # Handle tokenizer tensor devices before saving
    if hasattr(model.tokenizer, 'token_phases'):
        model.tokenizer.token_phases = model.tokenizer.token_phases.cpu()
    
    # Save tokenizer separately using pickle
    with open(checkpoint_file + '.tokenizer', 'wb') as f:
        pickle.dump(model.tokenizer, f)
    
    # Move optimizer state to CPU as well
    optimizer_state = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in optimizer.state_dict().items()
    }
    
    # Save metadata with CPU optimizer state
    metadata = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer_state,
        'loss': loss,
        'args': vars(args),
        'model_config': {
            'dim': model.dim,
            'vocab_size': len(model.tokenizer)
        },
        'model_version': '1.0'
    }
    torch.save(metadata, checkpoint_file + '.meta')
    
    # Save epoch-specific checkpoint
    print(f"\nSaving checkpoint to {checkpoint_file}")
    
    # Copy this checkpoint as the latest checkpoint
    latest_file = os.path.join(path, "checkpoint_last")
    shutil.copy2(checkpoint_file + '.model', latest_file + '.model')
    shutil.copy2(checkpoint_file + '.meta', latest_file + '.meta')
    shutil.copy2(checkpoint_file + '.tokenizer', latest_file + '.tokenizer')
    print(f"Updated latest checkpoint at {latest_file}")

def load_checkpoint(model: QuantumLLM, checkpoint_path: str, device: str):
    """Load model checkpoint"""
    import pickle
    print(f"\nLoading checkpoint from {checkpoint_path}")
    
    try:
        # Verify checkpoint files exist
        if not verify_checkpoint(checkpoint_path):
            return model, None
            
        # Define paths
        model_path = checkpoint_path + '.model'
        meta_path = checkpoint_path + '.meta'
        tokenizer_path = checkpoint_path + '.tokenizer'
            
        # Load metadata first to verify compatibility
        metadata = torch.load(meta_path, map_location=device, weights_only=True)
        
        # Version and configuration checks
        if 'model_config' in metadata:
            if metadata['model_config']['dim'] != model.dim:
                print(f"Warning: Model dimension mismatch: checkpoint has {metadata['model_config']['dim']}, but model has {model.dim}")
        
        # Load tokenizer if available
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
                # Ensure tokenizer tensors are on correct device
                if hasattr(tokenizer, 'token_phases'):
                    tokenizer.token_phases = tokenizer.token_phases.to(device)
                model.tokenizer = tokenizer
        
        # Load model weights safely
        model_state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(model_state)
        
        
        # Version check
        if 'model_version' not in metadata:
            print("Warning: Loading checkpoint from older version")
        
        return model, metadata
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return model, None

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get the path to the latest checkpoint if it exists"""
    latest_file = os.path.join(checkpoint_dir, "checkpoint_last")
    if os.path.exists(latest_file + '.model') and os.path.exists(latest_file + '.meta'):
        return latest_file
    return None

def train_model(model: QuantumLLM,
               dataset: List[dict],
               args: argparse.Namespace):
    """Train the quantum language model"""
    print(f"Training on device: {device}")
    
    # Ensure model is on correct device
    model = model.to(device)
    
    # Initialize optimizer with quantum-friendly parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoint:
        model, checkpoint = load_checkpoint(model, args.checkpoint, device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    total_batches = len(dataset)
    print(f"\nStarting quantum training...")
    print(f"Total batches: {total_batches}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()  # Ensure model is in training mode
        
        # Initialize statistics
        epoch_loss = 0
        valid_batches = 0
        start_time = time.time()
        
        # Create progress bar
        progress_bar = tqdm(dataset, desc=f"Training", 
                          leave=True, 
                          total=total_batches)
        
        for batch in progress_bar:
            try:
                # Process batch with error handling
                try:
                    input_ids = batch['input_ids'].to(device)
                except Exception as e:
                    print(f"Warning: Device allocation failed: {e}")
                    input_ids = batch['input_ids'].to("cpu")
                    model = model.to("cpu")
                
                if len(input_ids.shape) == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                # Create targets (shifted input)
                target_ids = torch.roll(input_ids, shifts=-1, dims=-1)
                target_ids[:, -1] = model.tokenizer.vocab[model.tokenizer.EOS_token]
                
                # Training step
                optimizer.zero_grad()
                
                # Forward pass with error catching
                try:
                    logits = model(input_ids)
                    loss = model.compute_loss(logits, target_ids)
                except RuntimeError as e:
                    if "MPS" in str(e):
                        print("Warning: MPS error in forward pass, falling back to CPU")
                        model = model.to("cpu")
                        input_ids = input_ids.to("cpu")
                        target_ids = target_ids.to("cpu")
                        logits = model(input_ids)
                        loss = model.compute_loss(logits, target_ids)
                    else:
                        raise e
                
                # Check loss validity
                if not torch.isfinite(loss):
                    print("Warning: Non-finite loss, skipping batch")
                    continue
                
                if loss.item() > 100:
                    print("Warning: Loss too high, skipping batch")
                    continue
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                valid_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{epoch_loss/valid_batches:.4f}",
                    'valid_batches': valid_batches
                })
                
            except RuntimeError as e:
                print(f"Runtime error: {e}")
                if "MPS" in str(e):
                    print("Falling back to CPU")
                    model = model.to("cpu")
                continue
            
            finally:
                try:
                    clear_memory()
                except:
                    pass
        
        # Epoch summary
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('nan')
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Valid Batches: {valid_batches}/{total_batches}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint at the end of each epoch
        save_checkpoint(
            model, 
            optimizer, 
            epoch, 
            avg_loss, 
            args,
            path=args.checkpoint_dir
        )

def ensure_model_on_device(model: QuantumLLM, device: str):
    """Ensure all model components are properly moved to specified device"""
    # Move base model
    model = model.to(device)
    
    # Ensure tokenizer tensors are on device
    if hasattr(model.tokenizer, 'token_phases'):
        model.tokenizer.token_phases = model.tokenizer.token_phases.to(device)
    
    # Verify all parameters are on device
    for param in model.parameters():
        if param.device != torch.device(device):
            param.data = param.data.to(device)
    
    return model

def generate_text(
    model: QuantumLLM,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7
) -> str:
    """Generate text using the quantum language model"""
    model.eval()
    
    # Encode prompt
    input_ids = model.tokenizer.encode(prompt).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = input_ids
        
        for _ in range(max_length):
            # Get predictions
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS token
            if next_token.item() == model.tokenizer.vocab[model.tokenizer.EOS_token]:
                break
            
            # Append and continue
            generated = torch.cat([generated, next_token], dim=1)
        
        # Decode the generated tokens
        return model.tokenizer.decode(generated[0])

def verify_checkpoint(checkpoint_path: str) -> bool:
    """Verify that a checkpoint is complete and valid"""
    required_extensions = ['.model', '.meta', '.tokenizer']
    for ext in required_extensions:
        if not os.path.exists(checkpoint_path + ext):
            print(f"Missing checkpoint file: {checkpoint_path + ext}")
            return False
    return True

def main(args):
    # Load dataset with quantum tokenizer
    dataset, tokenizer = load_quantum_wikitext(max_samples=args.max_samples)
    
    # Initialize quantum model
    model = QuantumLLM(
        tokenizer=tokenizer,
        dim=64
    )
    
    # Explicitly move model to device
    print(f"\nMoving model to device: {device}")
    model = model.to(device)
    
    # Ensure all model components are on the correct device
    for param in model.parameters():
        param.data = param.data.to(device)
    
    # Move tokenizer phases to device if they exist
    if hasattr(tokenizer, 'token_phases'):
        tokenizer.token_phases = tokenizer.token_phases.to(device)
    
    if args.mode == 'train':
        train_model(model, dataset, args)
    elif args.mode == 'generate':
        if not args.prompt:
            raise ValueError("Prompt is required for text generation")
        
        # Try to load specified checkpoint or fall back to latest
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
            if checkpoint_path:
                print(f"No checkpoint specified, using latest checkpoint: {checkpoint_path}")
        
        if checkpoint_path:
            model, checkpoint = load_checkpoint(model, checkpoint_path, device)
            model = ensure_model_on_device(model, device)  # Add this line
            if checkpoint is None:
                print("Warning: Failed to load checkpoint. Using untrained model.")
        else:
            print("Warning: No checkpoint available. Using untrained model.")
        
        generated_text = generate_text(
            model,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print("Generated text:", generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum-Inspired Language Model")
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True,
                      help="Mode: train or generate")
    parser.add_argument('--prompt', type=str, help="Prompt for text generation")
    parser.add_argument('--max_length', type=int, default=100,
                      help="Maximum length of generated text")
    parser.add_argument('--temperature', type=float, default=0.7,
                      help="Temperature for text generation")
    parser.add_argument('--max_samples', type=int, default=100,
                      help="Maximum number of samples to load from dataset")
    parser.add_argument('--epochs', type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument('--checkpoint', type=str,
                      help="Path to checkpoint file for loading")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                      help="Directory to save checkpoints")

    args = parser.parse_args()
    main(args)