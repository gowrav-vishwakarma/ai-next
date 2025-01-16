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
from dataclasses import dataclass
from collections import defaultdict
from contextlib import nullcontext

from library.quantum_ml import (
    BasicQuantumAttention, QuantumEmbedding, QuantumTokenizer, 
    SimpleQuantumState, create_quantum_tokenizer, compute_enhanced_quantum_loss
)

# Debug configuration
DEBUG_MODE = False  # Set to False to disable detailed training analysis

@dataclass
class TokenStats:
    """Track token statistics during training"""
    epoch: int
    avg_loss: float
    token_counts: Dict[int, int]
    total_tokens: int

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
    """Clear memory caches safely for all device types"""
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

class EnhancedDynamicPhaseSpace(nn.Module):
    """Enhanced dynamic phase space with quantum collapse prevention"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.quantum_state = SimpleQuantumState(dim)  # Use existing quantum state
        
        # Anti-collapse mechanism
        self.energy_levels = nn.Parameter(torch.linspace(0, 1, dim))
        self.excitation_factor = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor, pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get base quantum state
        real, imag = self.quantum_state(x)
        
        # Add collapse prevention
        amplitude = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)
        is_collapsing = (amplitude < 0.1).float()
        
        excitation = torch.exp(self.energy_levels) * self.excitation_factor
        real = real + is_collapsing * excitation.unsqueeze(0).unsqueeze(0)
        imag = imag + is_collapsing * excitation.unsqueeze(0).unsqueeze(0)
        
        # Renormalize
        norm = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)
        real = real / norm
        imag = imag / norm
        
        return real, imag

class QuantumStatePreservingAttention(BasicQuantumAttention):
    """Enhanced quantum attention mechanism that preserves quantum coherence"""
    def __init__(self, dim: int):
        super().__init__(dim)
        
        # Additional quantum state preservation
        self.phase_preservation = nn.Parameter(torch.ones(dim) * 0.1)
        
    def forward(self, q_real: torch.Tensor, q_imag: torch.Tensor,
                k_real: torch.Tensor, k_imag: torch.Tensor,
                v_real: torch.Tensor, v_imag: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Get base attention outputs
        out_real, out_imag = super().forward(q_real, q_imag, k_real, k_imag, v_real, v_imag, pad_mask)
        
        # Additional phase preservation
        phase = torch.atan2(out_imag + 1e-8, out_real + 1e-8)
        preservation = torch.sigmoid(self.phase_preservation)
        
        # Apply phase preservation
        preserved_real = out_real * torch.cos(phase * preservation.unsqueeze(0).unsqueeze(0))
        preserved_imag = out_imag * torch.sin(phase * preservation.unsqueeze(0).unsqueeze(0))
        
        return preserved_real, preserved_imag

class QuantumLLM(nn.Module):
    """Enhanced quantum language model with proper phase preservation"""
    def __init__(self, tokenizer: QuantumTokenizer, dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.dim = dim
        
        # Replace DynamicPhaseSpace with EnhancedDynamicPhaseSpace
        self.phase_space = EnhancedDynamicPhaseSpace(dim)
        
        # Quantum embedding
        self.embedding = QuantumEmbedding(tokenizer, dim)
        
        # Multi-layer quantum processing with enhanced attention
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': QuantumStatePreservingAttention(dim),
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
        
        # Get PAD token ID for collapse prevention
        pad_token_id = self.tokenizer.vocab[self.tokenizer.PAD_token]
        
        # Process through enhanced phase space
        real, imag = self.phase_space(real_embed, pad_token_id)
        
        # Create padding mask for attention
        pad_mask = (x == pad_token_id)
        
        # Process through quantum layers with enhanced attention
        for layer in self.layers:
            # Enhanced quantum attention
            attn_real, attn_imag = layer['attention'](
                real, imag, real, imag, real, imag,
                pad_mask=pad_mask
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
    
    # Save dataset to file
    script_name = os.path.basename(__file__)  # Gets the current file name
    dataset_file = script_name.rsplit('.', 1)[0] + "_dataset.txt"  # Creates filename_dataset.txt
    
    print(f"Saving dataset to {dataset_file}...")
    with open(dataset_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + "\n")
    
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

def analyze_epoch_stats(model: QuantumLLM, logits: torch.Tensor, epoch: int, avg_loss: float) -> None:
    """Fast analysis of epoch statistics"""
    if not DEBUG_MODE:
        return
        
    with torch.no_grad():
        # Get predictions efficiently using max
        predictions = logits.argmax(dim=-1)
        
        # Count tokens using bincount
        token_counts = torch.bincount(
            predictions.flatten(),
            minlength=len(model.tokenizer)
        ).cpu()
        
        # Create stats object
        stats = TokenStats(
            epoch=epoch,
            avg_loss=avg_loss,
            token_counts=dict(enumerate(token_counts.tolist())),
            total_tokens=predictions.numel()
        )
        
        # Print analysis
        print(f"\nEpoch {epoch} Analysis:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Total Tokens: {stats.total_tokens}")
        
        # Quick vocabulary coverage
        active_vocab = (token_counts > 0).sum().item()
        vocab_size = len(model.tokenizer)
        print(f"Active Vocabulary: {active_vocab} tokens ({active_vocab/vocab_size*100:.2f}%)")
        
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
            percentage = (count / stats.total_tokens) * 100
            print(f"{name}: {count} ({percentage:.2f}%)")
        
        # Top regular tokens (fast version)
        print("\nTop Regular Tokens:")
        special_ids = set(special_tokens.values())
        regular_counts = [(i, c.item()) for i, c in enumerate(token_counts) 
                         if i not in special_ids and c > 0]
        
        for token_id, count in sorted(regular_counts, key=lambda x: x[1], reverse=True)[:10]:
            token = model.tokenizer.reverse_vocab.get(token_id, '<UNK>')
            percentage = (count / stats.total_tokens) * 100
            print(f"{token}: {count} ({percentage:.2f}%)")

def get_autocast_device():
    """Get appropriate autocast device type"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def compute_quantum_loss(model: QuantumLLM, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Enhanced loss function with explicit coherence and diversity terms"""
    return compute_enhanced_quantum_loss(
        model,
        logits,
        targets,
        pad_token_id=model.tokenizer.vocab[model.tokenizer.PAD_token]
    )

def train_model(model: QuantumLLM, dataset, args: argparse.Namespace):
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Device-specific context manager
    def get_autocast_context():
        if device == "cuda":
            return torch.cuda.amp.autocast()
        elif device == "mps":
            return nullcontext()
        else:
            return nullcontext()
    
    # Device-specific gradient scaler
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    # Reduce batch size further for MPS
    batch_size = 1 if device == "mps" else 2
    gradient_accumulation_steps = 16 if device == "mps" else 8
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    # Break dataset into smaller chunks for processing
    chunk_size = 25 if device == "mps" else 50  # Smaller chunks for MPS
    num_chunks = (len(dataset) + chunk_size - 1) // chunk_size
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.02
    )
    
    if device == "mps":
        for layer in model.layers:
            layer['attention'].max_chunk_size = 32  # Limit chunk size for attention
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        optimizer.zero_grad(set_to_none=True)
        
        # Create progress bar for chunks
        chunk_pbar = tqdm(range(num_chunks), desc=f"Processing chunks", 
                         unit="chunk", position=0, leave=True)
        
        # Process dataset in chunks
        for chunk_idx in chunk_pbar:
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, len(dataset))
            chunk_data = dataset[chunk_start:chunk_end]
            
            # Create progress bar for batches within chunk
            batch_pbar = tqdm(range(0, len(chunk_data), batch_size), 
                            desc=f"Processing batches", unit="batch",
                            position=1, leave=False)
            
            # Process each chunk in batches
            for i in batch_pbar:
                try:
                    batch_end = min(i + batch_size, len(chunk_data))
                    input_ids = chunk_data[i:batch_end].to(device)
                    
                    with get_autocast_context():
                        # Forward pass
                        logits = model(input_ids)
                        target_ids = torch.roll(input_ids.clone(), shifts=-1, dims=-1)
                        
                        # Add EOS tokens
                        pad_mask = (input_ids == model.tokenizer.vocab[model.tokenizer.PAD_token])
                        seq_lengths = torch.argmax(pad_mask.float(), dim=1)
                        for idx, length in enumerate(seq_lengths):
                            if length == 0:
                                length = input_ids.size(1) - 1
                            target_ids[idx, length] = model.tokenizer.vocab[model.tokenizer.EOS_token]
                        
                        # Compute loss
                        loss = compute_quantum_loss(model, logits, target_ids)
                        loss = loss / gradient_accumulation_steps
                    
                    # Backward pass with device-specific handling
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    if torch.isfinite(loss):
                        epoch_loss += loss.item() * gradient_accumulation_steps
                        valid_batches += 1
                    
                    # Step optimizer after accumulation
                    if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    
                    # Update progress bars
                    if valid_batches > 0:
                        avg_loss = epoch_loss / valid_batches
                        batch_pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                        chunk_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
                    
                    # Clear memory
                    clear_memory()
                    
                except Exception as e:
                    print(f"Error in batch: {str(e)}")
                    traceback.print_exc()
                    continue
            
            batch_pbar.close()
        
        chunk_pbar.close()
        
        # Print epoch summary
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            print(f"\nEpoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, avg_loss, args)
    
    return model, {'avg_loss': avg_loss if valid_batches > 0 else float('inf')}

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
    """Generate text using the quantum language model with live updating output"""
    # Ensure model is in eval mode
    model.eval()
    
    # Encode input without adding EOS token
    input_ids = model.tokenizer.encode(prompt, add_special_tokens=True)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0).to(device)
    
    if DEBUG_MODE:
        print(f"\nInput Debug:")
        print(f"Input tokens: {input_ids.tolist()}")
    
    with torch.no_grad():
        generated = input_ids
        generated_tokens = []
        last_printed_length = 0
        
        # Print initial prompt
        print("\nGenerating:", flush=True)
        print(prompt, end='', flush=True)
        last_printed_length = len(prompt)
        
        for i in range(max_length):
            try:
                # Get model outputs
                logits = model(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Optional: Apply repetition penalty
                for token in generated_tokens[-5:]:
                    next_token_logits[0, token] /= 1.2
                
                # Get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Filter out special tokens
                special_tokens = [model.tokenizer.vocab[t] for t in 
                                [model.tokenizer.PAD_token, model.tokenizer.EOS_token]
                                if hasattr(model.tokenizer, t)]
                for token in special_tokens:
                    probs[0, token] = 0
                
                # Renormalize probabilities
                probs = probs / probs.sum()
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                token_value = next_token.item()
                generated_tokens.append(token_value)
                
                # Check for stopping conditions
                if token_value == model.tokenizer.vocab.get(model.tokenizer.EOS_token, -1):
                    if DEBUG_MODE:
                        print("\nEOS token generated, stopping")
                    break
                    
                if len(generated_tokens) >= max_length:
                    if DEBUG_MODE:
                        print("\nMax length reached, stopping")
                    break
                
                # Append and continue
                generated = torch.cat([generated, next_token], dim=1)
                
                # Get current text and print only the new part
                current_text = model.tokenizer.decode(generated[0].cpu(), skip_special_tokens=True)
                new_text = current_text[last_printed_length:]
                if new_text:
                    print(new_text, end='', flush=True)
                    last_printed_length = len(current_text)
                
            except Exception as e:
                print(f"\nError during generation step {i}: {str(e)}")
                traceback.print_exc()
                break
        
        # Print final newline
        print()
        
        try:
            result = model.tokenizer.decode(generated[0].cpu(), skip_special_tokens=True)
            if DEBUG_MODE:
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
        
        # Ensure the correct module is available in the namespace
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current directory to path
        
        with open(tokenizer_path, 'rb') as f:
            # Use the correct import path for the tokenizer
            import library.quantum_ml as quantum_ml  # Ensure this matches the module where QuantumTokenizer is defined
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
        print(f"\n\n\nGenerated text: '{generated_text}'")

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