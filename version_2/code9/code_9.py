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
    SimpleQuantumState, create_quantum_tokenizer, compute_enhanced_quantum_loss,
    TokenDistributionRegulator, EnhancedDynamicPhaseSpace,
    QuantumStatePreservingAttention, QuantumCoherenceLayer
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

def get_device():
    """Get best available device with enhanced error handling"""
    if torch.cuda.is_available():
        try:
            # Test CUDA device
            torch.cuda.current_device()
            return "cuda"
        except Exception as e:
            print(f"CUDA device found but error occurred: {e}")
            print("Falling back to next available device")
    
    if torch.backends.mps.is_available():
        try:
            # Test MPS device
            test_tensor = torch.zeros(1).to("mps")
            return "mps"
        except Exception as e:
            print(f"MPS device found but error occurred: {e}")
            print("Falling back to CPU")
    
    return "cpu"

def get_device_config(device: str, batch_size: int) -> dict:
    """Get device-specific configurations"""
    configs = {
        'cuda': {
            'batch_size': min(4, batch_size),
            'grad_accum_steps': 4,
            'chunk_size': 50,
            'use_amp': True,
            'attention_chunk_size': 128,
            'learning_rate': 2e-5,
            'eps': 1e-8,
            'gradient_clip': 1.0
        },
        'mps': {
            'batch_size': min(1, batch_size),
            'grad_accum_steps': 16,
            'chunk_size': 25,
            'use_amp': False,
            'attention_chunk_size': 32,
            'learning_rate': 1e-5,
            'eps': 1e-4,
            'gradient_clip': 0.5
        },
        'cpu': {
            'batch_size': min(2, batch_size),
            'grad_accum_steps': 8,
            'chunk_size': 40,
            'use_amp': False,
            'attention_chunk_size': 64,
            'learning_rate': 2e-5,
            'eps': 1e-8,
            'gradient_clip': 1.0
        }
    }
    return configs.get(device, configs['cpu'])

def ensure_model_on_device(model: nn.Module, device: str) -> nn.Module:
    """Ensure model and all its components are on the correct device"""
    try:
        model = model.to(device)
        
        # Ensure tokenizer tensors are on device if they exist
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'token_phases'):
            model.tokenizer.token_phases = model.tokenizer.token_phases.to(device)
        
        # Verify all parameters are on device
        for param in model.parameters():
            if param.device != torch.device(device):
                param.data = param.data.to(device)
        
        print(f"Model successfully moved to {device}")
        return model
    except Exception as e:
        print(f"Error moving model to {device}: {e}")
        print("Falling back to CPU")
        return model.to('cpu')

def get_autocast_context(device: str, use_amp: bool):
    """Get appropriate autocast context for device"""
    if device == "cuda" and use_amp:
        return torch.cuda.amp.autocast()
    elif device == "mps" and use_amp:
        # MPS doesn't support AMP yet, but might in the future
        return nullcontext()
    return nullcontext()

def clear_memory(device=None):
    """Enhanced memory clearing with device-specific optimizations"""
    try:
        # Clear Python garbage collector
        gc.collect()
        
        # Get device if not provided
        if device is None:
            device = get_device()
        
        # Device-specific memory clearing
        if device == "cuda":
            # CUDA-specific memory clearing
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device == "mps":
            # MPS-specific memory clearing
            try:
                torch.mps.empty_cache()
            except:
                pass
            
        # Clear JIT cache safely
        if hasattr(torch, 'jit'):
            try:
                torch.jit._state._python_cu.clear_cache()
            except:
                pass
            
    except Exception as e:
        print(f"Warning: Memory clearing failed: {str(e)}")

def prefetch_data(dataset, indices, device):
    """Prefetch data to device memory efficiently"""
    if device == "cuda":
        # Use CUDA streams for async data transfer
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            data = dataset[indices].to(device, non_blocking=True)
        return data, stream
    else:
        # Synchronous transfer for other devices
        return dataset[indices].to(device), None

class DataPrefetcher:
    """Efficient data prefetching for all device types"""
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.index = 0
        self.length = len(dataset)
        self.stream = torch.cuda.Stream() if device == 'cuda' else None
    
    def prefetch(self):
        """Prefetch next batch"""
        if self.index >= self.length:
            return None
            
        end_idx = min(self.index + self.batch_size, self.length)
        if self.device == 'cuda':
            with torch.cuda.stream(self.stream):
                batch = self.dataset[self.index:end_idx].to(
                    self.device, non_blocking=True)
        else:
            batch = self.dataset[self.index:end_idx].to(self.device)
            
        self.index = end_idx
        return batch, self.stream
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        data = self.prefetch()
        if data is None:
            raise StopIteration
        return data

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
        
        # Add layer normalization for stability
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        
        # Fix: Convert scale to tensor before assignment
        self.register_buffer('scale', torch.tensor(dim).pow(-0.5))
        
    def forward(self, q_real: torch.Tensor, q_imag: torch.Tensor,
                k_real: torch.Tensor, k_imag: torch.Tensor,
                v_real: torch.Tensor, v_imag: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        try:
            # Apply layer normalization
            q_real, q_imag = self.q_norm(q_real), self.q_norm(q_imag)
            k_real, k_imag = self.k_norm(k_real), self.k_norm(k_imag)
            v_real, v_imag = self.v_norm(v_real), self.v_norm(v_imag)
            
            # Scale inputs
            q_real, q_imag = q_real * self.scale, q_imag * self.scale
            k_real, k_imag = k_real * self.scale, k_imag * self.scale
            
            # Compute attention scores with stability measures
            scores_real = torch.clamp(
                torch.matmul(q_real, k_real.transpose(-2, -1)) - 
                torch.matmul(q_imag, k_imag.transpose(-2, -1)),
                min=-10, max=10
            )
            
            scores_imag = torch.clamp(
                torch.matmul(q_real, k_imag.transpose(-2, -1)) + 
                torch.matmul(q_imag, k_real.transpose(-2, -1)),
                min=-10, max=10
            )
            
            # Apply padding mask if provided
            if pad_mask is not None:
                scores_real = scores_real.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
                scores_imag = scores_imag.masked_fill(pad_mask.unsqueeze(1), 0.0)
            
            # Compute attention weights with numerical stability
            weights = F.softmax(scores_real, dim=-1)
            weights = torch.clamp(weights, min=1e-6, max=1.0)
            
            # Apply attention
            out_real = torch.matmul(weights, v_real)
            out_imag = torch.matmul(weights, v_imag)
            
            # Add phase preservation with scaling
            phase = torch.atan2(out_imag + 1e-8, out_real + 1e-8)
            preservation = torch.sigmoid(self.phase_preservation) * 0.1  # Scale down preservation
            
            # Apply phase preservation with clamping
            preserved_real = torch.clamp(
                out_real * torch.cos(phase * preservation.unsqueeze(0).unsqueeze(0)),
                min=-5, max=5
            )
            preserved_imag = torch.clamp(
                out_imag * torch.sin(phase * preservation.unsqueeze(0).unsqueeze(0)),
                min=-5, max=5
            )
            
            # Add residual skip connection
            preserved_real = preserved_real + q_real * 0.1
            preserved_imag = preserved_imag + q_imag * 0.1
            
            return preserved_real, preserved_imag
            
        except Exception as e:
            print(f"Error in attention mechanism: {str(e)}")
            traceback.print_exc()
            # Return zero tensors as fallback
            return torch.zeros_like(q_real), torch.zeros_like(q_imag)

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
        
        # Add token distribution regulator
        self.token_regulator = TokenDistributionRegulator(len(tokenizer))
        
        # Output projection
        self.pre_output_norm = nn.LayerNorm(dim * 2)
        self.output = nn.Linear(dim * 2, len(tokenizer))
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # Get quantum embeddings with value checking
            real_embed, imag_embed = self.embedding(x)
            print(f"Embedding ranges - Real: {real_embed.min():.4f} to {real_embed.max():.4f}, "
                  f"Imag: {imag_embed.min():.4f} to {imag_embed.max():.4f}")
            
            # Scale down embeddings
            real_embed = real_embed * 0.1
            imag_embed = imag_embed * 0.1
            
            # Get PAD token ID for collapse prevention
            pad_token_id = self.tokenizer.vocab[self.tokenizer.PAD_token]
            
            # Process through enhanced phase space with gradient scaling
            real, imag = self.phase_space(real_embed, pad_token_id)
            
            # Add stability measures
            real = torch.clamp(real, min=-1, max=1)  # Tighter clamping
            imag = torch.clamp(imag, min=-1, max=1)  # Tighter clamping
            
            # Create padding mask for attention
            pad_mask = (x == pad_token_id)
            
            # Process through quantum layers with enhanced attention
            for i, layer in enumerate(self.layers):
                # Enhanced quantum attention with value checking
                attn_real, attn_imag = layer['attention'](
                    real, imag, real, imag, real, imag,
                    pad_mask=pad_mask
                )
                
                # Add residual connection with smaller scaling
                real = layer['phase_norm'](real + attn_real * 0.01)  # Reduced scaling
                imag = layer['phase_norm'](imag + attn_imag * 0.01)  # Reduced scaling
                
                # Add stability measures
                real = torch.clamp(real, min=-1, max=1)  # Tighter clamping
                imag = torch.clamp(real, min=-1, max=1)  # Tighter clamping
                
                print(f"Layer {i} outputs - Real: {real.min():.4f} to {real.max():.4f}, "
                      f"Imag: {imag.min():.4f} to {imag.max():.4f}")
            
            # Combine for output with scaling
            combined = torch.cat([real, imag], dim=-1)
            combined = self.pre_output_norm(combined)
            
            # Final output projection with stability measures
            logits = self.output(combined) * 0.1  # Scale down logits
            logits = torch.clamp(logits, min=-10, max=10)  # Tighter clamping
            
            return logits
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            traceback.print_exc()
            return None
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Enhanced loss function with phase coherence"""
        return compute_enhanced_quantum_loss(
            self,
            logits,
            targets,
            pad_token_id=self.tokenizer.vocab[self.tokenizer.PAD_token]
        )
    
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
        print(f"Error loading checkpoint: {str(e)}")
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

def train_model(model: nn.Module, dataset: torch.Tensor, args: argparse.Namespace) -> Tuple[nn.Module, Dict]:
    # Get device and configuration
    device = get_device()
    print(f"Training on device: {device}")
    print(f"Dataset size: {len(dataset)}")
    
    # Get device-specific configuration
    config = get_device_config(device, args.batch_size)
    
    # Move model to device
    model = ensure_model_on_device(model, device)
    
    # Initialize training parameters
    batch_size = config['batch_size']
    gradient_accumulation_steps = config['grad_accum_steps']
    chunk_size = getattr(args, 'chunk_size', config['chunk_size'])
    use_amp = config['use_amp']
    gradient_clip = config['gradient_clip']
    learning_rate = config['learning_rate']
    
    # Add gradient scaling for numerical stability
    grad_scale = 1.0 if device == 'mps' else 4.0  # Higher scaling for MPS
    
    print(f"Using configuration for {device}:")
    print(f"  batch_size: {batch_size}")
    print(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  use_amp: {use_amp}")
    print(f"  gradient_clip: {gradient_clip}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  gradient_scale: {grad_scale}")

    # Calculate number of chunks
    num_chunks = (len(dataset) + chunk_size - 1) // chunk_size

    print(f"Using batch_size: {batch_size}, chunk_size: {chunk_size}, gradient_clip: {gradient_clip}")
    print(f"Number of chunks: {num_chunks}")

    # Initialize optimizer with more stable settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=config['eps'],  # Higher eps for MPS
        weight_decay=0.01  # Reduced weight decay
    )

    # Add learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Device-specific context manager for mixed precision
    def get_autocast_context():
        if device == "cuda" and use_amp:
            return torch.cuda.amp.autocast()
        return nullcontext()

    def safe_backward(loss, optimizer, scaler=None):
        try:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            return True
        except RuntimeError as e:
            print(f"Error in backward pass: {str(e)}")
            optimizer.zero_grad(set_to_none=True)
            return False

    def process_batch(input_ids, model, optimizer, scaler=None):
        try:
            with get_autocast_context():
                # Debug input values
                print(f"Input range: {input_ids.min().item():.4f} to {input_ids.max().item():.4f}")
                
                # Forward pass with gradient scaling and value clamping
                with torch.autograd.detect_anomaly():  # Enable anomaly detection
                    logits = model(input_ids)
                    
                    # Add value clamping for stability
                    logits = torch.clamp(logits, min=-100, max=100)
                    logits = logits / grad_scale
                    
                    # Debug intermediate values
                    print(f"Pre-scaled logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
                
                # Check for NaN values early
                if torch.isnan(logits).any():
                    print("Warning: NaN values detected in logits")
                    print("Model state:", {name: param.mean().item() for name, param in model.named_parameters()})
                    return None
                
                print(f"Logits shape: {logits.shape}")
                print(f"Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")

                target_ids = torch.roll(input_ids.clone(), shifts=-1, dims=-1)
                print(f"Target shape: {target_ids.shape}")
                print(f"Target range: {target_ids.min().item():.4f} to {target_ids.max().item():.4f}")

                # Add EOS tokens
                pad_mask = (input_ids == model.tokenizer.vocab[model.tokenizer.PAD_token])
                seq_lengths = torch.argmax(pad_mask.float(), dim=1)
                for idx, length in enumerate(seq_lengths):
                    if length == 0:
                        length = input_ids.size(1) - 1
                    target_ids[idx, length] = model.tokenizer.vocab[model.tokenizer.EOS_token]

                # Compute loss with gradient clipping and stability measures
                loss = compute_quantum_loss(model, logits, target_ids)
                
                # Add stability term to loss
                stability_term = 0.01 * (logits ** 2).mean()  # L2 regularization
                loss = loss + stability_term
                
                # Scale loss for stability
                loss = loss / gradient_accumulation_steps
                
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss detected: {loss.item()}")
                    print("Loss components:")
                    print(f"  Main loss: {compute_quantum_loss(model, logits, target_ids).item()}")
                    print(f"  Stability term: {stability_term.item()}")
                    return None

                print(f"Loss: {loss.item():.4f}")
                return loss

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            traceback.print_exc()
            return None

    # Initialize training with stability measures
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set smaller initial learning rate for stability
    initial_lr = learning_rate * 0.1
    warmup_steps = 100
    
    # Initialize optimizer with more conservative settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.98),  # More conservative beta2
        eps=1e-6 if device == 'mps' else 1e-8,  # Increased epsilon
        weight_decay=0.001  # Reduced weight decay
    )

    # Add learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize AMP scaler for CUDA
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"Using AMP: {use_amp}, Scaler: {'enabled' if scaler else 'disabled'}")  # Debug info

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        for chunk_idx in tqdm(range(num_chunks), desc=f"Processing chunks"):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, len(dataset))
            chunk_data = dataset[chunk_start:chunk_end]
            
            for i in range(0, len(chunk_data), batch_size):
                try:
                    batch_end = min(i + batch_size, len(chunk_data))
                    input_ids = chunk_data[i:batch_end].to(device)
                    
                    # Process batch with error handling
                    loss = process_batch(input_ids, model, optimizer, scaler)
                    
                    if loss is not None:
                        # Backward pass with error handling
                        if safe_backward(loss, optimizer, scaler):
                            if torch.isfinite(loss):
                                epoch_loss += loss.item() * gradient_accumulation_steps
                                valid_batches += 1
                                print(f"Valid batch processed - Running average loss: {epoch_loss/valid_batches:.4f}")
                    
                    # Gradient accumulation step
                    if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                        
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    
                except Exception as e:
                    print(f"\nError in batch processing: {str(e)}")
                    traceback.print_exc()
                    continue
                
                # Clear memory with device
                clear_memory(device)
        
        # Update learning rate based on epoch loss
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch+1, avg_loss, args)
        else:
            print(f"\nEpoch {epoch+1} - No valid batches processed")

    return model, {'final_loss': epoch_loss / valid_batches if valid_batches > 0 else float('inf')}

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
    # Add argument parsing at the start of main
    parser = argparse.ArgumentParser(description='Train quantum language model')
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'eval', 'generate'],
                      help='Mode to run in (train, eval, or generate)')
    parser.add_argument('--max_samples', type=int, default=1000,
                      help='Maximum number of samples to process')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--chunk_size', type=int, default=512,
                      help='Chunk size for processing')
    parser.add_argument('--model_dim', type=int, default=64,
                      help='Model dimension')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory for saving checkpoints')
    
    # Add generation-specific arguments
    parser.add_argument('--prompt', type=str, default=None,
                      help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=100,
                      help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Sampling temperature')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Specific checkpoint to load')
    
    args = parser.parse_args()
    
    # Get device first
    device = get_device()
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        # Load dataset with quantum tokenizer
        dataset, tokenizer = load_quantum_wikitext(max_samples=args.max_samples)
        
        # Initialize quantum model
        model = QuantumLLM(
            tokenizer=tokenizer,
            dim=args.model_dim
        )
        
        # Move model to device with error handling
        model = ensure_model_on_device(model, device)
        
        # Train model
        model, training_stats = train_model(model, dataset, args)
        print(f"\nFinal training loss: {training_stats['final_loss']:.4f}")
    
    elif args.mode == 'generate':
        if not args.prompt:
            raise ValueError("Prompt is required for text generation")
        
        # Get checkpoint path
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
            
        if not checkpoint_path:
            raise ValueError("No checkpoint found. Please train the model first or specify a checkpoint path.")
            
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
        print(f"\nGenerated text: '{generated_text}'")

    elif args.mode == 'eval':
        # Evaluation mode implementation
        if not args.checkpoint:
            checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        else:
            checkpoint_path = args.checkpoint
            
        if not checkpoint_path:
            raise ValueError("No checkpoint found for evaluation")
            
        # Load model and evaluate
        model = load_model_for_evaluation(checkpoint_path, device)
        eval_results = evaluate_model(model, args)
        print("\nEvaluation Results:")
        for metric, value in eval_results.items():
            print(f"{metric}: {value:.4f}")

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
    parser.add_argument('--batch_size', type=int, default=32,
                      help="Batch size for training")
    parser.add_argument('--checkpoint', type=str,
                      help="Path to checkpoint file for loading")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                      help="Directory to save checkpoints")

    args = parser.parse_args()
    main(args)