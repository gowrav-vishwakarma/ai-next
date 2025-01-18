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
    """Dynamic phase space with layered representation and interference patterns"""
    def __init__(self, dim: int, num_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
        
        # Phase spaces for different semantic layers (semantic, context, emotion)
        self.layer_phases = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(num_layers)
        ])
        
        # Quantum interference mixers
        self.interference_weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.02) for _ in range(num_layers)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim, eps=1e-8) for _ in range(num_layers)
        ])
        
        # Energy levels for collapse prevention
        self.energy_levels = nn.Parameter(torch.linspace(0, 1, dim))
        self.excitation_factor = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor, pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_parts = []
        imag_parts = []
        
        for layer_idx in range(self.num_layers):
            # Generate dynamic phase angles using golden ratio
            phase_angles = torch.tanh(self.layer_phases[layer_idx]) * math.pi * self.PHI
            
            # Create quantum state components
            real = x * torch.cos(phase_angles)
            imag = x * torch.sin(phase_angles)
            
            # Normalize
            real = self.layer_norms[layer_idx](real)
            imag = self.layer_norms[layer_idx](imag)
            
            # Apply interference
            interference = torch.tanh(self.interference_weights[layer_idx])
            real = real * interference
            imag = imag * interference
            
            real_parts.append(real)
            imag_parts.append(imag)
        
        # Combine through quantum interference
        final_real = torch.zeros_like(x)
        final_imag = torch.zeros_like(x)
        
        for i in range(self.num_layers):
            angle = 2 * math.pi * i / self.num_layers
            phase_factor_real = torch.cos(torch.tensor(angle))
            phase_factor_imag = torch.sin(torch.tensor(angle))
            
            final_real += real_parts[i] * phase_factor_real - imag_parts[i] * phase_factor_imag
            final_imag += real_parts[i] * phase_factor_imag + imag_parts[i] * phase_factor_real
        
        # Add collapse prevention
        amplitude = torch.sqrt(final_real.pow(2) + final_imag.pow(2) + 1e-8)
        is_collapsing = (amplitude < 0.1).float()
        
        excitation = torch.exp(self.energy_levels) * self.excitation_factor
        final_real = final_real + is_collapsing * excitation.unsqueeze(0).unsqueeze(0)
        final_imag = final_imag + is_collapsing * excitation.unsqueeze(0).unsqueeze(0)
        
        # Normalize
        norm = torch.sqrt(final_real.pow(2) + final_imag.pow(2) + 1e-8)
        final_real = final_real / norm
        final_imag = final_imag / norm
        
        return final_real, final_imag

class QuantumStatePreservingAttention(BasicQuantumAttention):
    """Interference-based quantum attention with phase coherence"""
    def __init__(self, dim: int):
        super().__init__(dim)
        
        # Phase preservation and coherence
        self.phase_preservation = nn.Parameter(torch.ones(dim) * 0.1)
        self.coherence_factor = nn.Parameter(torch.ones(1) * 0.1)
        
        # Layer normalization for stability
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        
        # Attention weights - match input dimension
        self.W_attn = nn.Parameter(torch.randn(dim, dim) * 0.02)
        
        self.register_buffer('scale', torch.tensor(dim).pow(-0.5))
    
    def compute_coherence(self, phase_diffs: torch.Tensor) -> torch.Tensor:
        """Compute quantum coherence measure"""
        return torch.mean(torch.cos(phase_diffs), dim=(-1, -2))
    
    def forward(self, q_real: torch.Tensor, q_imag: torch.Tensor,
                k_real: torch.Tensor, k_imag: torch.Tensor,
                v_real: torch.Tensor, v_imag: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Get input shapes
            B, L, D = q_real.shape  # Batch, Length, Dimension
            
            # Apply layer normalization
            q_real, q_imag = self.q_norm(q_real), self.q_norm(q_imag)
            k_real, k_imag = self.k_norm(k_real), self.k_norm(k_imag)
            v_real, v_imag = self.v_norm(v_real), self.v_norm(v_imag)
            
            # Extract phases
            q_phase = torch.atan2(q_imag + 1e-8, q_real + 1e-8)  # [B, L, D]
            k_phase = torch.atan2(k_imag + 1e-8, k_real + 1e-8)  # [B, L, D]
            
            # Compute phase differences for interference - fix dimensions
            # Reshape to enable broadcasting
            q_phase_expanded = q_phase.unsqueeze(2)  # [B, L, 1, D]
            k_phase_expanded = k_phase.unsqueeze(1)  # [B, 1, L, D]
            phase_diffs = q_phase_expanded - k_phase_expanded  # [B, L, L, D]
            
            # Compute interference pattern
            interference = torch.cos(phase_diffs)  # [B, L, L, D]
            
            # Apply attention weights and scaling
            # Reshape interference to match weight matrix
            interference_flat = interference.view(B * L * L, D)  # [B*L*L, D]
            weighted = torch.matmul(interference_flat, self.W_attn)  # [B*L*L, D]
            attn_scores = weighted.view(B, L, L, D)  # [B, L, L, D]
            
            # Average over dimension D to get attention scores
            attn_scores = torch.mean(attn_scores, dim=-1) * self.scale  # [B, L, L]
            
            # Apply padding mask if provided
            if pad_mask is not None:
                # Ensure pad_mask has correct shape [B, L]
                pad_mask = pad_mask.view(B, L)
                attn_scores = attn_scores.masked_fill(
                    pad_mask.unsqueeze(1).expand(-1, L, -1),
                    float('-inf')
                )
            
            # Compute attention weights with stability
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L, L]
            attn_weights = torch.clamp(attn_weights, min=1e-6, max=1.0)
            
            # Apply attention
            out_real = torch.matmul(attn_weights, v_real)  # [B, L, D]
            out_imag = torch.matmul(attn_weights, v_imag)  # [B, L, D]
            
            # Compute and apply coherence
            coherence = self.compute_coherence(phase_diffs)  # [B, L]
            coherence_factor = torch.sigmoid(self.coherence_factor) * coherence.unsqueeze(-1)
            
            # Apply phase preservation with coherence
            phase = torch.atan2(out_imag + 1e-8, out_real + 1e-8)
            preservation = torch.sigmoid(self.phase_preservation) * coherence_factor
            
            preserved_real = out_real * torch.cos(phase * preservation)
            preserved_imag = out_imag * torch.sin(phase * preservation)
            
            return preserved_real, preserved_imag
            
        except Exception as e:
            print(f"Error in attention mechanism: {str(e)}")
            traceback.print_exc()
            # Return zero tensors with correct shape
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
            if DEBUG_MODE:  
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
                
                if DEBUG_MODE:
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

def is_colab():
    """Check if code is running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_project_root():
    """Get project root directory for both Colab and local environments"""
    if is_colab():
        return '/content/code9'
    else:
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:  # If __file__ is not defined
            return os.path.abspath('code9')

def load_quantum_wikitext(max_samples: Optional[int] = None):
    """Load and preprocess Wikitext with quantum tokenizer"""
    # Get project root and create directories
    root_dir = get_project_root()
    data_dir = os.path.join(root_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Load raw dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split='train')
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Extract texts
    texts = [example['text'] for example in dataset]
    
    # Save dataset to file
    dataset_file = os.path.join(data_dir, 'wikitext_dataset.txt')
    
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

def save_checkpoint(model: QuantumLLM, optimizer, epoch: int, loss: float, args, path: str = None):
    """Save model checkpoint"""
    if path is None:
        path = os.path.join(get_project_root(), 'checkpoints')
    
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

def process_batch(input_ids, model, optimizer, scheduler=None):
    try:
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        target_ids = torch.roll(input_ids, shifts=-1, dims=-1)
        loss = model.compute_loss(logits, target_ids)
        
        # Backward pass with retain_graph=True
        loss.backward(retain_graph=True)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step optimizer
        optimizer.step()
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        return loss
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        traceback.print_exc()
        return None

def train_model(model: nn.Module, dataset: torch.Tensor, args: argparse.Namespace) -> Tuple[nn.Module, Dict]:
    # Get device and configuration
    device = get_device()
    print(f"Training on device: {device}")
    
    # Get device-specific configuration
    config = get_device_config(device, args.batch_size)
    config['learning_rate'] = 1e-3  # Increase from 1e-5 to 1e-3
    config['gradient_clip'] = 1.0   # Increase from 0.5 to 1.0
    
    # Initialize training parameters
    batch_size = config['batch_size']
    chunk_size = getattr(args, 'chunk_size', config['chunk_size'])
    
    # Calculate number of chunks
    num_chunks = (len(dataset) + chunk_size - 1) // chunk_size
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Using batch_size: {batch_size}, chunk_size: {chunk_size}")
    print(f"Number of chunks: {num_chunks}")
    
    # Initialize optimizer with better settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01
    )
    
    # Create scheduler
    num_training_steps = args.epochs * (len(dataset) // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=num_training_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Move model to device
    model = ensure_model_on_device(model, device)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        # Create progress bar for chunks
        chunk_pbar = tqdm(range(num_chunks), desc=f"Processing chunks")
        
        for chunk_idx in chunk_pbar:
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, len(dataset))
            chunk_data = dataset[chunk_start:chunk_end]
            
            # Process batches within chunk
            batch_pbar = tqdm(
                range(0, len(chunk_data), batch_size),
                desc=f"Processing batches",
                leave=False
            )
            
            for i in batch_pbar:
                batch_end = min(i + batch_size, len(chunk_data))
                input_ids = chunk_data[i:batch_end].to(device)
                
                # Process batch with scheduler
                loss = process_batch(input_ids, model, optimizer, scheduler)
                
                if loss is not None:
                    epoch_loss += loss.item()
                    valid_batches += 1
                    batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear memory periodically
                if i % (batch_size * 10) == 0:
                    clear_memory(device)
            
            # Update chunk progress
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                chunk_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        
        # Compute epoch average loss
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
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

def generate_text(model: QuantumLLM, prompt: str, max_length: int = 100, 
                 temperature: float = 0.7, nucleus_threshold: float = 0.9) -> str:
    """Generate text using the quantum language model with improved sampling

    Args:
        model: The QuantumLLM model
        prompt: Input text to continue from
        max_length: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random, lower = more focused)
        nucleus_threshold: Controls diversity via nucleus sampling (0.1-0.99)
                         Higher values include more low-probability tokens
                         Lower values make text more focused but potentially repetitive

    Returns:
        str: Generated text including the prompt
    """
    # Get device
    device = next(model.parameters()).device
    
    # Ensure model is in eval mode
    model.eval()
    
    # Encode input without adding EOS token
    input_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0).to(device)
    
    if DEBUG_MODE:
        print(f"\nInput Debug:")
        print(f"Input tokens: {input_ids.tolist()}")
        print(f"Generation settings:")
        print(f"- Temperature: {temperature}")
        print(f"- Nucleus threshold: {nucleus_threshold}")
    
    with torch.no_grad():
        generated = input_ids
        generated_tokens = []
        last_printed_length = 0
        
        # Print initial prompt
        print("\nGenerating:", flush=True)
        print(prompt, end='', flush=True)
        last_printed_length = len(prompt)
        
        # Get special token IDs
        eos_token_id = model.tokenizer.vocab.get(model.tokenizer.EOS_token, -1)
        pad_token_id = model.tokenizer.vocab.get(model.tokenizer.PAD_token, -1)
        special_tokens = [eos_token_id, pad_token_id]
        
        consecutive_special = 0  # Track consecutive special tokens
        
        for i in range(max_length):
            try:
                # Get model outputs
                logits = model(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                for token in generated_tokens[-5:]:
                    next_token_logits[0, token] /= 1.5  # Increased penalty
                
                # Get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Prevent early EOS by reducing its probability early in generation
                min_tokens = 10  # Minimum tokens to generate
                if len(generated_tokens) < min_tokens:
                    probs[0, eos_token_id] *= 0.1  # Reduce EOS probability early on
                
                # Filter out special tokens early in generation
                if len(generated_tokens) < min_tokens:
                    for token_id in special_tokens:
                        probs[0, token_id] *= 0.1
                
                # Enhance probability of non-special tokens
                special_mask = torch.ones_like(probs)
                for token_id in special_tokens:
                    special_mask[0, token_id] = 0.1
                probs = probs * special_mask
                
                # Renormalize probabilities
                probs = probs / probs.sum()
                
                # Sample next token with nucleus sampling using provided threshold
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus_mask = cumsum_probs < nucleus_threshold
                nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
                nucleus_mask[..., 0] = True  # Always keep the top token
                
                # Apply nucleus sampling
                sorted_probs[~nucleus_mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                
                # Sample from filtered distribution
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices[0, next_token[0]]
                
                if DEBUG_MODE and i < 5:  # Show first few tokens for debugging
                    token_text = model.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                    print(f"\nToken {i}: {token_text} (id: {next_token.item()}, prob: {probs[0, next_token.item()]:.4f})")
                
                # Track consecutive special tokens
                if next_token.item() in special_tokens:
                    consecutive_special += 1
                else:
                    consecutive_special = 0
                
                # Stop if too many consecutive special tokens
                if consecutive_special >= 3:
                    if DEBUG_MODE:
                        print("\nStopping due to consecutive special tokens")
                    break
                
                # Add token to generated sequence
                generated_tokens.append(next_token.item())
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
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
                print(f"Total tokens generated: {len(generated_tokens)}")
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
    parser.add_argument('--nucleus_threshold', type=float, default=0.9,
                      help='Nucleus sampling threshold (0.1-0.99)')
    
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
            temperature=args.temperature,
            nucleus_threshold=args.nucleus_threshold
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