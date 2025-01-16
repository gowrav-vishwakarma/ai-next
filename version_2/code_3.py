from collections import Counter
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
import traceback

from library.quantum_ml import BasicQuantumAttention, QuantumEmbedding, QuantumTokenizer, SimpleQuantumState, create_quantum_tokenizer, DynamicPhaseSpace

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
    """Enhanced quantum language model with proper phase preservation"""
    def __init__(self, tokenizer: QuantumTokenizer, dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.dim = dim
        
        # Dynamic phase space
        self.phase_space = DynamicPhaseSpace(dim)
        
        # Quantum embedding
        self.embedding = QuantumEmbedding(tokenizer, dim)
        
        # Multi-layer quantum processing
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': BasicQuantumAttention(dim),
                'phase_norm': nn.LayerNorm(dim, eps=1e-8)
            }) for _ in range(3)
        ])
        
        # Output projection
        self.pre_output_norm = nn.LayerNorm(dim * 2)
        self.output = nn.Linear(dim * 2, len(tokenizer))
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get quantum embeddings
        real_embed, imag_embed = self.embedding(x)
        
        # Process through dynamic phase space
        real, imag = self.phase_space(real_embed)
        
        # Process through quantum layers
        for layer in self.layers:
            # Quantum attention with interference
            attn_real, attn_imag = layer['attention'](
                real, imag, real, imag, real, imag
            )
            
            # Phase-preserving residual
            real = layer['phase_norm'](real + attn_real)
            imag = layer['phase_norm'](imag + attn_imag)
        
        # Combine for output
        combined = torch.cat([real, imag], dim=-1)
        combined = self.pre_output_norm(combined)
        
        return self.output(combined)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Enhanced loss function with phase coherence"""
        # Basic cross entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, len(self.tokenizer)),
            targets.view(-1),
            label_smoothing=0.1,
            ignore_index=self.tokenizer.vocab[self.tokenizer.PAD_token]
        )
        
        # Compute coherence loss in chunks
        chunk_size = 32
        B, L, V = logits.shape
        coherence_loss = 0.0
        num_chunks = 0
        
        for i in range(0, L, chunk_size):
            j = min(i + chunk_size, L)
            chunk = logits[:, i:j, :]
            
            chunk_phases = torch.atan2(
                chunk.unsqueeze(-2),
                chunk.unsqueeze(-1)
            )
            
            chunk_coherence = -torch.mean(torch.cos(chunk_phases))
            coherence_loss += chunk_coherence
            num_chunks += 1
        
        coherence_loss = coherence_loss / num_chunks
        
        # Combine losses
        total_loss = ce_loss + 0.1 * coherence_loss
        
        return total_loss
    
    def to(self, device):
        """Override to method to ensure all components move to device"""
        super().to(device)
        self.embedding = self.embedding.to(device)
        self.phase_space = self.phase_space.to(device)
        self.layers = self.layers.to(device)
        self.pre_output_norm = self.pre_output_norm.to(device)
        self.output = self.output.to(device)
        return self

def load_quantum_wikitext(max_samples: Optional[int] = None):
    """Load and preprocess Wikitext with quantum tokenizer"""
    # Load raw dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split='train')
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Extract texts
    texts = [example['text'] for example in dataset]
    
    # Create and train tokenizer
    print("Creating quantum tokenizer...")
    tokenizer = create_quantum_tokenizer(texts, dim=64)
    
    # Process all texts at once
    print("Tokenizing dataset...")
    all_encodings = []
    
    for text in tqdm(texts, desc="Processing texts"):
        # Tokenize with quantum tokenizer
        encodings = tokenizer.encode(text)
        
        # Ensure consistent length
        if len(encodings) > 512:
            encodings = encodings[:512]
        else:
            pad_length = 512 - len(encodings)
            encodings = torch.cat([
                encodings,
                torch.full((pad_length,), tokenizer.vocab[tokenizer.PAD_token])
            ])
        
        all_encodings.append(encodings)
    
    # Stack all encodings into a single tensor
    input_ids = torch.stack(all_encodings)
    
    return input_ids, tokenizer

def save_checkpoint(model: QuantumLLM, optimizer, epoch: int, loss: float, args, path: str = "checkpoints"):
    """Save model checkpoint"""
    import os
    import pickle  # Add import for tokenizer serialization
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Create checkpoint filename with epoch and loss
    checkpoint_file = os.path.join(path, f"model_epoch_{epoch}_loss_{loss:.4f}.pt")
    
    # Save model state dict separately
    model_state = model.state_dict()
    torch.save(model_state, checkpoint_file + '.model')
    
    # Handle tokenizer tensor devices before saving
    # if hasattr(model.tokenizer, 'token_phases'):
    #     model.tokenizer.token_phases = model.tokenizer.token_phases.cpu()
    
    # Save tokenizer separately using pickle
    with open(checkpoint_file + '.tokenizer', 'wb') as f:
        pickle.dump(model.tokenizer, f)
    
    # Update metadata to include model configuration
    metadata = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args),
        'model_config': {  # Add model configuration
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

class TokenStats:
    """Class to track token statistics"""
    def __init__(self, total_tokens: int, token_counts: Dict[int, int], avg_loss: float):
        self.total_tokens = total_tokens
        self.token_counts = token_counts
        self.avg_loss = avg_loss

def get_autocast_device():
    """Get appropriate autocast device type"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def compute_quantum_loss(model: QuantumLLM, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Enhanced loss function with explicit coherence and diversity terms"""
    # Base cross entropy
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=model.tokenizer.vocab[model.tokenizer.PAD_token]
    )
    
    # Get probability distribution
    with torch.no_grad():  # Reduce memory usage
        probs = F.softmax(logits, dim=-1)
        token_dist = probs.mean(dim=[0, 1])
    
    # Compute coherence loss in very small chunks
    chunk_size = 4  # Much smaller chunks for limited memory
    mini_batch = 2  # Process even smaller batches at a time
    B, L, V = logits.shape
    coherence_loss = 0.0
    num_chunks = 0
    
    # Get appropriate autocast type
    autocast_device = get_autocast_device()
    
    # Process in mini-batches to save memory
    for b in range(0, B, mini_batch):
        b_end = min(b + mini_batch, B)
        batch_logits = logits[b:b_end]
        
        for i in range(0, L, chunk_size):
            j = min(i + chunk_size, L)
            try:
                # Take a small chunk of the logits
                chunk = batch_logits[:, i:j, :]
                
                # Free unused memory
                if device == "mps":
                    torch.mps.empty_cache()
                
                # Compute phase differences for this small chunk
                with torch.amp.autocast(device_type=autocast_device, enabled=True):
                    chunk_phases = torch.atan2(
                        chunk.unsqueeze(-2),  # [mini_batch, chunk_size, 1, V]
                        chunk.unsqueeze(-1)   # [mini_batch, chunk_size, V, 1]
                    )
                    
                    # Compute coherence for this chunk
                    chunk_coherence = -torch.mean(torch.cos(chunk_phases))
                    coherence_loss += chunk_coherence.item()  # Convert to scalar immediately
                    num_chunks += 1
                
                # Clear memory
                del chunk_phases
                del chunk_coherence
                if device == "mps":
                    torch.mps.empty_cache()
                
            except RuntimeError as e:
                print(f"Warning: Chunk processing failed, skipping chunk: {str(e)}")
                continue
    
    # Average the coherence loss
    coherence_loss = coherence_loss / max(num_chunks, 1)
    
    # Diversity loss using entropy (computed on full distribution)
    with torch.no_grad():  # Reduce memory usage
        diversity_loss = -(token_dist * torch.log(token_dist + 1e-10)).sum()
    
    # Combine losses with weights
    total_loss = (
        ce_loss +
        0.2 * coherence_loss +  # Coherence weight
        0.1 * (1.0 - torch.clamp(diversity_loss, 0.0, 0.5))  # Diversity weight
    )
    
    return total_loss

def train_model(model: QuantumLLM, dataset, args: argparse.Namespace):
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Use smaller learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.02
    )
    
    # Create very small batches for limited memory
    batch_size = 4  # Reduced batch size for 8GB RAM
    
    # Get appropriate autocast type
    autocast_device = get_autocast_device()
    
    # Dataset is already a tensor from load_quantum_wikitext
    num_samples = len(dataset)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        # Create progress bar
        progress_bar = tqdm(range(0, num_samples, batch_size), desc="Training")
        
        for i in progress_bar:
            try:
                # Clear memory before each batch
                clear_memory()
                
                # Get batch indices
                batch_end = min(i + batch_size, num_samples)
                
                # Get batch directly from tensor
                input_ids = dataset[i:batch_end].to(device)
                
                # Create targets
                target_ids = input_ids.clone()
                target_ids = torch.roll(target_ids, shifts=-1, dims=-1)
                
                # Add EOS at sequence ends
                pad_mask = (input_ids == model.tokenizer.vocab[model.tokenizer.PAD_token])
                seq_lengths = torch.argmax(pad_mask.float(), dim=1)
                for idx, length in enumerate(seq_lengths):
                    if length == 0:  # No padding found
                        length = input_ids.size(1) - 1
                    target_ids[idx, length] = model.tokenizer.vocab[model.tokenizer.EOS_token]
                
                # Training step with memory optimization
                optimizer.zero_grad(set_to_none=True)  # More memory efficient
                
                with torch.amp.autocast(device_type=autocast_device, enabled=True):
                    logits = model(input_ids)
                    loss = compute_quantum_loss(model, logits, target_ids)
                
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    valid_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{epoch_loss/valid_batches:.4f}"
                    })
                
                # Clear memory after each batch
                del logits, loss, input_ids, target_ids
                clear_memory()
                
            except Exception as e:
                print(f"Batch error: {str(e)}")
                traceback.print_exc()
                continue
        
        # Save checkpoint and analyze epoch
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch, avg_loss, args)
    
    return model, {'avg_loss': avg_loss if valid_batches > 0 else float('inf')}

def analyze_epoch(model: QuantumLLM, stats: TokenStats, epoch: int):
    """Analyze single epoch results"""
    print(f"\nEpoch {epoch} Analysis (Avg Loss: {stats.avg_loss:.4f}):")
    
    # Special token analysis
    special_tokens = {
        'EOS': model.tokenizer.vocab[model.tokenizer.EOS_token],
        'PAD': model.tokenizer.vocab[model.tokenizer.PAD_token],
        'BOS': model.tokenizer.vocab[model.tokenizer.BOS_token],
        'UNK': model.tokenizer.vocab[model.tokenizer.UNK_token]
    }
    
    print("\nSpecial Token Usage:")
    for name, token_id in special_tokens.items():
        count = stats.token_counts.get(token_id, 0)
        percentage = (count / stats.total_tokens) * 100 if stats.total_tokens > 0 else 0
        print(f"{name}: {count} ({percentage:.2f}%)")
    
    # Regular token analysis
    regular_tokens = {
        token: count for token, count in stats.token_counts.items()
        if token not in special_tokens.values()
    }
    
    print("\nTop Regular Tokens:")
    for token_id, count in sorted(regular_tokens.items(), key=lambda x: x[1], reverse=True)[:10]:
        token = model.tokenizer.reverse_vocab.get(token_id, '<UNK>')
        percentage = (count / stats.total_tokens) * 100 if stats.total_tokens > 0 else 0
        print(f"{token}: {count} ({percentage:.2f}%)")
    
    # Coverage statistics
    vocab_coverage = len(stats.token_counts) / len(model.tokenizer) * 100
    print(f"\nVocabulary Coverage: {vocab_coverage:.2f}%")

def analyze_token_distribution(model: QuantumLLM, stats_list: List[TokenStats]):
    """Analyze token distribution across all epochs"""
    print("\nOverall Training Analysis:")
    
    for i, stats in enumerate(stats_list):
        print(f"\nEpoch {i + 1}:")
        print(f"Average Loss: {stats.avg_loss:.4f}")
        
        total_tokens = stats.total_tokens
        token_counts = stats.token_counts
        
        # Token statistics
        active_tokens = len(token_counts)
        print(f"Total Tokens: {total_tokens}")
        print(f"Active Vocabulary: {active_tokens} tokens ({active_tokens/len(model.tokenizer)*100:.2f}%)")
        
        # Special token analysis
        special_tokens = {
            'EOS': model.tokenizer.vocab[model.tokenizer.EOS_token],
            'PAD': model.tokenizer.vocab[model.tokenizer.PAD_token],
            'BOS': model.tokenizer.vocab[model.tokenizer.BOS_token],
            'UNK': model.tokenizer.vocab[model.tokenizer.UNK_token]
        }
        
        print("\nSpecial Token Usage:")
        for name, token_id in special_tokens.items():
            count = token_counts.get(token_id, 0)
            percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
            print(f"{name}: {count} ({percentage:.2f}%)")
        
        # Regular token analysis
        print("\nTop Regular Tokens:")
        regular_tokens = {
            token: count for token, count in token_counts.items()
            if token not in special_tokens.values()
        }
        
        # Sort and display top tokens
        sorted_tokens = sorted(regular_tokens.items(), key=lambda x: x[1], reverse=True)[:10]
        for token_id, count in sorted_tokens:
            token = model.tokenizer.reverse_vocab.get(token_id, '<UNK>')
            percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
            print(f"{token}: {count} ({percentage:.2f}%)")

def analyze_epoch_results(model: QuantumLLM, token_counts: torch.Tensor, epoch: int, avg_loss: float):
    """Analyze training results for an epoch"""
    total_tokens = token_counts.sum().item()
    if total_tokens == 0:
        print("No tokens predicted in this epoch!")
        return
    
    print(f"\nEpoch {epoch} Analysis (Avg Loss: {avg_loss:.4f}):")
    
    # Convert to float for probability calculations
    token_probs = token_counts.float() / total_tokens
    
    # Special token analysis
    special_tokens = {
        'EOS': model.tokenizer.vocab[model.tokenizer.EOS_token],
        'PAD': model.tokenizer.vocab[model.tokenizer.PAD_token],
        'BOS': model.tokenizer.vocab[model.tokenizer.BOS_token],
        'UNK': model.tokenizer.vocab[model.tokenizer.UNK_token]
    }
    
    print("\nSpecial Token Usage:")
    for name, token_id in special_tokens.items():
        count = token_counts[token_id].item()
        percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"{name}: {count} ({percentage:.2f}%)")
    
    # Find top regular tokens
    top_tokens = torch.topk(token_counts, min(10, len(token_counts)))
    
    print("\nTop Regular Tokens:")
    for i in range(len(top_tokens.indices)):
        token_id = top_tokens.indices[i].item()
        count = top_tokens.values[i].item()
        if token_id not in special_tokens.values():
            token = model.tokenizer.reverse_vocab.get(token_id, '<UNK>')
            percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
            print(f"{token}: {count} ({percentage:.2f}%)")

def analyze_epoch_distribution(model: QuantumLLM, token_dist: Counter, epoch: int):
    """Analyze token distribution for an epoch"""
    total_tokens = sum(token_dist.values())
    
    print(f"\nEpoch {epoch} Distribution Analysis:")
    
    # Special token analysis
    special_tokens = {
        'EOS': model.tokenizer.vocab[model.tokenizer.EOS_token],
        'PAD': model.tokenizer.vocab[model.tokenizer.PAD_token],
        'BOS': model.tokenizer.vocab[model.tokenizer.BOS_token],
        'UNK': model.tokenizer.vocab[model.tokenizer.UNK_token]
    }
    
    print("\nSpecial Token Usage:")
    for name, token_id in special_tokens.items():
        count = token_dist[token_id]
        percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"{name}: {count} ({percentage:.2f}%)")
    
    # Top tokens analysis
    print("\nTop 10 Regular Tokens:")
    regular_tokens = {token: count for token, count in token_dist.items() 
                     if token not in special_tokens.values()}
    
    for token_id, count in Counter(regular_tokens).most_common(10):
        token = model.tokenizer.reverse_vocab.get(token_id, '<UNK>')
        percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"{token}: {count} ({percentage:.2f}%)")
    
    # Distribution metrics
    vocab_coverage = len(token_dist) / len(model.tokenizer) * 100
    print(f"\nVocabulary Coverage: {vocab_coverage:.2f}%")
    
    # Entropy calculation
    probs = [count / total_tokens for count in token_dist.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(model.tokenizer))
    entropy_ratio = (entropy / max_entropy) * 100
    print(f"Distribution Entropy: {entropy:.2f} bits ({entropy_ratio:.2f}% of maximum)")



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

def generate_text(model: QuantumLLM, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
    """Generate text using the quantum language model with enhanced debugging"""
    # Ensure model is in eval mode
    model.eval()
    
    # Encode input without adding EOS token
    input_ids = model.tokenizer.encode(prompt, add_special_tokens=True)  # Add this parameter to your tokenizer
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0).to(device)
    
    print(f"\nInput Debug:")
    print(f"Input tokens: {input_ids.tolist()}")
    
    with torch.no_grad():
        generated = input_ids
        generated_tokens = []
        
        print("\nGeneration Debug:")
        for i in range(max_length):
            try:
                # Get model outputs
                logits = model(generated)
                
                # Get next token logits and apply temperature
                next_token_logits = logits[:, -1, :] / temperature
                
                # Optional: Apply repetition penalty
                for token in generated_tokens[-5:]:  # Look at last 5 tokens
                    next_token_logits[0, token] /= 1.2  # Reduce probability of recent tokens
                
                # Get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Filter out special tokens (optional)
                special_tokens = [model.tokenizer.vocab[t] for t in 
                                [model.tokenizer.PAD_token, model.tokenizer.EOS_token]
                                if hasattr(model.tokenizer, t)]
                for token in special_tokens:
                    probs[0, token] = 0
                
                # Renormalize probabilities
                probs = probs / probs.sum()
                
                # Debug probability distribution
                top_probs, top_tokens = torch.topk(probs, 5)
                print(f"\nStep {i}:")
                print(f"Top 5 tokens: {top_tokens[0].tolist()}")
                print(f"Top 5 probs: {top_probs[0].tolist()}")
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                token_value = next_token.item()
                generated_tokens.append(token_value)
                
                # Optional: Add explicit stopping conditions
                if token_value == model.tokenizer.vocab.get(model.tokenizer.EOS_token, -1):
                    print("EOS token generated, stopping")
                    break
                    
                if len(generated_tokens) >= max_length:
                    print("Max length reached, stopping")
                    break
                
                # Append and continue
                generated = torch.cat([generated, next_token], dim=1)
                
                # Show current text
                current_text = model.tokenizer.decode(generated[0].cpu(), skip_special_tokens=True)
                print(f"Current text: '{current_text}'")
                
            except Exception as e:
                print(f"Error during generation step {i}: {str(e)}")
                traceback.print_exc()
                break
        
        # Final decoding
        try:
            result = model.tokenizer.decode(generated[0].cpu(), skip_special_tokens=True)
            print(f"\nFinal Debug:")
            print(f"Generated tokens: {generated_tokens}")
            print(f"Final tensor shape: {generated.shape}")
            return result
        except Exception as e:
            print(f"Error during final decoding: {str(e)}")
            return prompt

def verify_model_state(model: QuantumLLM, device: str):
    """Verify model state and components"""
    print("\nModel Verification:")
    
    # Check model mode
    print(f"Model training mode: {model.training}")
    
    # Verify device placement
    print("\nDevice Placement:")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Embedding device: {next(model.embedding.parameters()).device}")
    print(f"Attention device: {next(model.attention.parameters()).device}")
    if hasattr(model.tokenizer, 'token_phases'):
        print(f"Tokenizer phases device: {model.tokenizer.token_phases.device}")
    
    # Verify shapes
    print("\nComponent Shapes:")
    print(f"Output layer shape: {model.output.weight.shape}")
    print(f"Embedding dimension: {model.dim}")
    
    # Return verification status
    return True

def load_model_for_generation(checkpoint_path: str, device: str):
    """Load model with enhanced verification"""
    try:
        # Load and verify tokenizer
        tokenizer_path = checkpoint_path + '.tokenizer'
        print(f"\nLoading tokenizer from: {tokenizer_path}")
        
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Debug tokenizer state
        print("\nTokenizer State:")
        print(f"Vocab size: {len(tokenizer.vocab)}")
        
        # Load model configuration
        meta_path = checkpoint_path + '.meta'
        metadata = torch.load(meta_path, map_location=device, weights_only=True)
        model_dim = metadata['model_config']['dim']
        
        # Initialize and load model
        print(f"\nInitializing model with dim={model_dim}")
        model = QuantumLLM(tokenizer=tokenizer, dim=model_dim)
        
        # Load weights
        model_path = checkpoint_path + '.model'
        model_state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(model_state)
        
        # Explicitly set eval mode
        model = model.eval()
        
        # Move to device
        model = model.to(device)
        
        # Verify model is in eval mode
        print(f"Model training mode: {model.training}")
        
        return model
        
    except Exception as e:
        print(f"Error during model loading: {str(e)}")
        traceback.print_exc()
        raise



def verify_checkpoint(checkpoint_path: str) -> bool:
    """Verify that a checkpoint is complete and valid"""
    required_extensions = ['.model', '.meta', '.tokenizer']
    for ext in required_extensions:
        if not os.path.exists(checkpoint_path + ext):
            print(f"Missing checkpoint file: {checkpoint_path + ext}")
            return False
    return True

def main(args):
    if args.mode == 'train':
        # Load dataset with quantum tokenizer
        dataset, tokenizer = load_quantum_wikitext(max_samples=args.max_samples)
        
        # Initialize quantum model
        model = QuantumLLM(
            tokenizer=tokenizer,
            dim=64
        )
        
        # Move model to device
        print(f"\nMoving model to device: {device}")
        model = ensure_model_on_device(model, device)
        
        # Train model
        model, training_stats = train_model(model, dataset, args)
        print(f"\nFinal training loss: {training_stats['avg_loss']:.4f}")

    elif args.mode == 'generate':
        if not args.prompt:
            raise ValueError("Prompt is required for text generation")
        
        # Get checkpoint path
        checkpoint_path = args.checkpoint or get_latest_checkpoint(args.checkpoint_dir)
        if not checkpoint_path:
            raise ValueError("No checkpoint found")
            
        print(f"Using checkpoint: {checkpoint_path}")
        
        # Load model with enhanced loading
        model = load_model_for_generation(checkpoint_path, device)
        
        # Generate with debug info
        print(f"Generating with prompt: '{args.prompt}'")
        generated_text = generate_text(
            model,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"Generated text: '{generated_text}'")

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