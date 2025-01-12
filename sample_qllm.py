import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
from collections import defaultdict
import re
import math
import os
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import itertools

class ComplexMPS:
    """Handle complex operations on MPS by splitting real and imaginary parts"""
    def __init__(self, real, imag):
        self.real = real.to("mps")
        self.imag = imag.to("mps")
    
    @classmethod
    def from_complex(cls, complex_tensor):
        return cls(complex_tensor.real, complex_tensor.imag)
    
    def matmul(self, other):
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        real = torch.matmul(self.real, other.real) - torch.matmul(self.imag, other.imag)
        imag = torch.matmul(self.real, other.imag) + torch.matmul(self.imag, other.real)
        return ComplexMPS(real, imag)
    
    def to_complex(self):
        return torch.complex(self.real, self.imag)

    @staticmethod
    def batch_matmul(tensors_real, tensors_imag, weights_real, weights_imag):
        """Optimized batch matrix multiplication for complex numbers"""
        # Compute all real and imaginary parts in parallel
        real_part = torch.matmul(tensors_real, weights_real) - torch.matmul(tensors_imag, weights_imag)
        imag_part = torch.matmul(tensors_real, weights_imag) + torch.matmul(tensors_imag, weights_real)
        return real_part, imag_part

class EnhancedQuantumTokenizer:
    """Enhanced tokenizer using Hugging Face tokenizers and quantum encoding"""
    def __init__(self, max_sequence_length=2048, tokenizer_name="bert-base-uncased"):
        self.max_sequence_length = max_sequence_length

        # Load tokenizer from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab = self.tokenizer.get_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Quantum encoding maps
        self.quantum_encodings = self._initialize_quantum_encodings()
    
    def _initialize_char_vocab(self):
        """Initialize character-level vocabulary"""
        chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_')
        for char in chars:
            if len(self.vocab) < self.vocab_size:
                self.vocab[char] = len(self.vocab)
                self.reverse_vocab[self.vocab[char]] = char
    
    def _initialize_subword_vocab(self):
        """Initialize common subword tokens"""
        common_subwords = ['ing', 'ed', 'ly', 'er', 'est', 'un', 're', 'th', 'ch', 'sh']
        for subword in common_subwords:
            if len(self.vocab) < self.vocab_size:
                self.vocab[subword] = len(self.vocab)
                self.reverse_vocab[self.vocab[subword]] = subword
    
    def _initialize_word_vocab(self):
        """Initialize common word tokens"""
        common_words = ['the', 'be', 'to', 'of', 'and', 'in', 'that', 'have', 'it', 'for']
        for word in common_words:
            if len(self.vocab) < self.vocab_size:
                self.vocab[word] = len(self.vocab)
                self.reverse_vocab[self.vocab[word]] = word
    
    def _initialize_quantum_encodings(self):
        """Initialize quantum-inspired encodings for tokens"""
        encodings = {}
        dim = 64  # Encoding dimension
        
        vocab_size = len(self.vocab)  # Get vocab size from tokenizer
        
        for token_id in range(vocab_size):
            # Create quantum state vector
            phase = 2 * np.pi * token_id / vocab_size
            state = np.zeros(dim, dtype=np.complex128)
            
            # Fill with superposition of bases
            for i in range(dim):
                theta = np.pi * i / dim
                state[i] = np.exp(1j * (phase + theta))
            
            # Normalize
            state /= np.sqrt(np.sum(np.abs(state)**2))
            encodings[token_id] = state
        
        return encodings
    
    def encode(self, text, max_length=None):
        """Encode text using the Hugging Face tokenizer"""
        if not text:
            return []

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        return encoded["input_ids"][0]
    
    def decode(self, tokens):
        """Decode token IDs back to text"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

class EnhancedQuantumAttention(nn.Module):
    """Enhanced attention mechanism with multi-scale processing"""
    def __init__(self, dim, num_heads, max_sequence_length=2048):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_sequence_length = max_sequence_length
        
        # Initialize multi-scale patterns
        self.interference_scales = self._initialize_interference_scales()
        self.expert_patterns = self._initialize_expert_patterns()
    
    def _initialize_interference_scales(self):
        """Initialize multi-scale interference patterns"""
        scales = {}
        frequencies = {'char': 1.0, 'word': 0.5, 'phrase': 0.25, 'sentence': 0.1}
        
        for name, freq in frequencies.items():
            scales[name] = self._generate_wave_patterns(freq)
        
        return scales
    
    def _generate_wave_patterns(self, frequency):
        """Generate complex wave patterns"""
        head_dim = self.head_dim
        patterns = torch.zeros((self.num_heads, self.max_sequence_length, head_dim),
                               dtype=torch.complex64, device="mps")
        
        for h in range(self.num_heads):
            phase = 2 * np.pi * h / self.num_heads
            t = torch.linspace(0, 2 * np.pi * frequency, self.max_sequence_length, device="mps")
            
            for d in range(head_dim):
                patterns[h, :, d] = (torch.exp(1j * (t + phase)) +
                                   torch.exp(1j * (2 * t + phase)) +
                                   torch.exp(1j * (0.5 * t + phase)))
        
        # Normalize across sequence length dimension
        patterns = patterns / torch.sqrt(torch.sum(torch.abs(patterns)**2, dim=1, keepdim=True))
        return patterns
    
    def _initialize_expert_patterns(self):
        """Initialize expert patterns for different aspects"""
        num_experts = 8
        patterns = []
        
        for i in range(num_experts):
            # Change: Ensure patterns match head dimensions
            head_dim = self.head_dim
            pattern = {
                'syntax': self._generate_expert_wave_pattern(0.3 + 0.1*i, head_dim),
                'semantics': self._generate_expert_wave_pattern(0.2 + 0.1*i, head_dim),
                'context': self._generate_expert_wave_pattern(0.1 + 0.1*i, head_dim)
            }
            patterns.append(pattern)
        
        return patterns
    
    def _generate_expert_wave_pattern(self, frequency, dim):
        """Generate quantum wave pattern for experts"""
        pattern = torch.zeros((self.max_sequence_length, dim), dtype=torch.complex64, device="mps")
        
        # Create quantum interference pattern
        t = torch.linspace(0, 2 * np.pi, self.max_sequence_length, device="mps")
        for d in range(dim):
            phase = 2 * np.pi * d / dim
            pattern[:, d] = torch.exp(1j * (frequency * t + phase))
        
        # Normalize the pattern
        pattern = pattern / torch.sqrt(torch.sum(torch.abs(pattern)**2, dim=0, keepdim=True))
        return pattern
    
    def _process_scale(self, Q, K, V, pattern, mask=None):
        """Optimized attention computation"""
        B, H, L, D = Q.shape
        
        # Split into real and imaginary parts once
        Q_real, Q_imag = Q.real, Q.imag  # [B, H, L, D]
        K_real, K_imag = K.real, K.imag  # [B, H, L, D]
        V_real, V_imag = V.real, V.imag  # [B, H, L, D]
        
        # Reshape inputs for batch matmul
        Q_real = Q_real.reshape(B * H, L, D)
        Q_imag = Q_imag.reshape(B * H, L, D)
        K_real = K_real.reshape(B * H, L, D)
        K_imag = K_imag.reshape(B * H, L, D)
        
        # Compute attention scores in parallel
        # [B*H, L, D] x [B*H, D, L] -> [B*H, L, L]
        scores_real = torch.bmm(Q_real, K_real.transpose(1, 2)) - torch.bmm(Q_imag, K_imag.transpose(1, 2))
        scores_imag = torch.bmm(Q_real, K_imag.transpose(1, 2)) + torch.bmm(Q_imag, K_real.transpose(1, 2))
        
        # Compute magnitudes [B*H, L, L]
        scores_magnitude = torch.sqrt(scores_real**2 + scores_imag**2)
        scores_magnitude = scores_magnitude.reshape(B, H, L, L) / math.sqrt(D)
        
        if mask is not None:
            scores_magnitude = scores_magnitude.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights [B, H, L, L]
        attention_weights = torch.softmax(scores_magnitude, dim=-1)
        
        # Reshape attention weights and values for batch matmul
        attention_weights = attention_weights.reshape(B * H, L, L)
        V_real = V_real.reshape(B * H, L, D)
        V_imag = V_imag.reshape(B * H, L, D)
        
        # Apply attention in parallel [B*H, L, L] x [B*H, L, D] -> [B*H, L, D]
        output_real = torch.bmm(attention_weights, V_real)
        output_imag = torch.bmm(attention_weights, V_imag)
        
        # Reshape output back to [B, H, L, D]
        output_real = output_real.reshape(B, H, L, D)
        output_imag = output_imag.reshape(B, H, L, D)
        
        return torch.complex(output_real, output_imag)
    
    def forward(self, Q, K, V, mask=None):
        """Forward pass with quantum interference"""
        B, H, L, D = Q.shape
        
        # Process at different scales with quantum interference
        scale_outputs = []
        for name, scale_pattern in self.interference_scales.items():
            # Ensure pattern has correct shape
            pattern = scale_pattern[:H, :L, :D]  # Slice to match dimensions
            scale_output = self._process_scale(Q, K, V, pattern, mask)
            scale_outputs.append(scale_output)
        
        # Quantum superposition of scale outputs
        combined = sum(scale_outputs) / math.sqrt(len(scale_outputs))
        
        # Apply expert patterns with quantum interference
        expert_outputs = []
        for expert in self.expert_patterns:
            expert_output = self._apply_expert(combined, expert)
            expert_outputs.append(expert_output)
        
        # Final quantum combination
        output = sum(expert_outputs) / math.sqrt(len(expert_outputs))
        return output
    
    def _apply_expert(self, x, expert):
        """Apply expert pattern using quantum interference"""
        B, H, L, D = x.shape  # Batch, Heads, Length, Dim
        
        # Split input into real and imaginary parts
        x_real = x.real.reshape(B * H, L, D)
        x_imag = x.imag.reshape(B * H, L, D)
        
        # Ensure expert patterns are on the correct device and split into real/imag
        expert_patterns = {}
        for aspect, pattern in expert.items():
            pattern = pattern.to(x.device)
            expert_patterns[f"{aspect}_real"] = pattern.real[:L]
            expert_patterns[f"{aspect}_imag"] = pattern.imag[:L]
        
        # Apply patterns separately to real and imaginary parts
        outputs_real = []
        outputs_imag = []
        for aspect in ['syntax', 'semantics', 'context']:
            # Complex multiplication in parts
            real_part = (torch.einsum('bld,ld->bld', x_real, expert_patterns[f"{aspect}_real"]) - 
                        torch.einsum('bld,ld->bld', x_imag, expert_patterns[f"{aspect}_imag"]))
            imag_part = (torch.einsum('bld,ld->bld', x_real, expert_patterns[f"{aspect}_imag"]) + 
                        torch.einsum('bld,ld->bld', x_imag, expert_patterns[f"{aspect}_real"]))
            outputs_real.append(real_part)
            outputs_imag.append(imag_part)
        
        # Combine using quantum superposition
        combined_real = sum(outputs_real) / math.sqrt(3)
        combined_imag = sum(outputs_imag) / math.sqrt(3)
        
        # Reshape back to original dimensions and combine into complex tensor
        return torch.complex(
            combined_real.reshape(B, H, L, D),
            combined_imag.reshape(B, H, L, D)
        )

class EnhancedUniversalPatterns:
    """Enhanced pattern recognition using universal constants"""
    def __init__(self, dim):
        self.dim = dim
        self.patterns = self._initialize_enhanced_patterns()
    
    def _initialize_enhanced_patterns(self):
        """Initialize patterns using multiple universal constants"""
        patterns = {}
        
        # Golden ratio patterns - using tensor operations
        phi = (1 + torch.tensor(5.0, device="mps").sqrt()) / 2
        patterns['golden'] = self._generate_golden_patterns(phi)
        
        # Pi-based patterns
        patterns['pi'] = self._generate_pi_patterns()
        
        # E-based patterns
        patterns['e'] = self._generate_e_patterns()
        
        return patterns
    
    def _generate_golden_patterns(self, phi):
        """Generate patterns based on golden ratio"""
        # Create indices tensors
        i_indices = torch.arange(self.dim, device="mps").unsqueeze(1)
        j_indices = torch.arange(self.dim, device="mps").unsqueeze(0)
        
        # Compute powers using broadcasting
        powers = (i_indices + j_indices).float() / self.dim
        patterns = phi.pow(powers).to(dtype=torch.complex64)
        
        return patterns / torch.max(torch.abs(patterns))
    
    def _generate_pi_patterns(self):
        """Generate patterns based on pi"""
        # Create indices tensors
        i_indices = torch.arange(self.dim, device="mps").unsqueeze(1)
        j_indices = torch.arange(self.dim, device="mps").unsqueeze(0)
        
        # Compute using torch operations
        angle = torch.pi * (i_indices + j_indices).float() / self.dim
        patterns = torch.sin(angle).to(dtype=torch.complex64)
        
        return patterns
    
    def _generate_e_patterns(self):
        """Generate patterns based on e"""
        # Create indices tensors
        i_indices = torch.arange(self.dim, device="mps").unsqueeze(1)
        j_indices = torch.arange(self.dim, device="mps").unsqueeze(0)
        
        # Compute using torch operations
        exponent = -(i_indices + j_indices).float() / self.dim
        patterns = torch.exp(exponent).to(dtype=torch.complex64)
        
        return patterns
    
    def forward(self, x):
        """Apply universal patterns to input"""
        # Create a new tensor instead of copying
        output = x.clone()
        
        for pattern in self.patterns.values():
            # Ensure pattern is on the same device as input
            pattern = pattern.to(x.device)
            # Apply pattern using proper broadcasting
            output = output + torch.matmul(output, pattern)
        
        return output / len(self.patterns)

class EnhancedQuantumOptimizer:
    """Enhanced optimizer with quantum-inspired updates"""
    def __init__(self, learning_rate=1e-4, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.step_count = 0
        self.m = defaultdict(float)
        self.v = defaultdict(float)
    
    def step(self, model, loss):
        """Update parameters using quantum-inspired optimization"""
        def stabilize_complex(tensor, eps=1e-8):
            magnitude = torch.sqrt(tensor.real**2 + tensor.imag**2)
            # Clip extremely small magnitudes
            magnitude = torch.clamp(magnitude, min=eps)
            phase = torch.atan2(tensor.imag.cpu(), tensor.real.cpu()).to("mps")
            return torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))

        self.step_count += 1
        params = {
            'token_embeddings': model.token_embeddings,
            'position_embeddings': model.position_embeddings
        }
        
        grads = self._compute_quantum_gradients(model, loss, params)
        
        for name, param in params.items():
            if isinstance(param, torch.Tensor):
                grad = grads[name]
                
                # Stabilize gradients
                grad = stabilize_complex(grad)
                
                # Update momentum terms with stabilization
                self.m[name] = stabilize_complex(self.beta1 * self.m[name] + (1 - self.beta1) * grad)
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad.real**2 + grad.imag**2)
                
                # Bias correction with stabilization
                m_hat = stabilize_complex(self.m[name] / (1 - self.beta1**self.step_count))
                v_hat = self.v[name] / (1 - self.beta2**self.step_count)
                
                # Compute update with gradient clipping
                update = self.learning_rate * m_hat / (torch.sqrt(v_hat + 1e-8))
                update_magnitude = torch.sqrt(update.real**2 + update.imag**2)
                max_update = 0.1  # Maximum allowed update magnitude
                
                if torch.any(update_magnitude > max_update):
                    scale = max_update / (update_magnitude + 1e-8)
                    scale = torch.minimum(torch.ones_like(scale), scale)
                    update = torch.complex(update.real * scale, update.imag * scale)
                
                # Update parameter with stabilization
                param.data = stabilize_complex(param.data - update)
            
            elif isinstance(param, EnhancedQuantumAttention):
                # Update attention layer parameters
                for expert_pattern in param.expert_patterns:
                    for aspect, pattern in expert_pattern.items():
                        name = f'{name}_expert_{aspect}'
                        grad = grads[name]
                        
                        # Update momentum terms
                        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad.real**2 + grad.imag**2)
                        
                        # Bias correction
                        m_hat = self.m[name] / (1 - self.beta1**self.step_count)
                        v_hat = self.v[name] / (1 - self.beta2**self.step_count)
                        
                        # Update parameter
                        update = self.learning_rate * m_hat / (torch.sqrt(v_hat) + 1e-8)
                        pattern -= update
            
            elif isinstance(param, EnhancedUniversalPatterns):
                # Update universal pattern parameters
                for pattern_name, pattern in param.patterns.items():
                    name = f'{name}_{pattern_name}'
                    grad = grads[name]
                    
                    # Update momentum terms
                    self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                    self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad.real**2 + grad.imag**2)
                    
                    # Bias correction
                    m_hat = self.m[name] / (1 - self.beta1**self.step_count)
                    v_hat = self.v[name] / (1 - self.beta2**self.step_count)
                    
                    # Update parameter
                    update = self.learning_rate * m_hat / (torch.sqrt(v_hat) + 1e-8)
                    pattern -= update
    
    def _compute_quantum_gradients(self, model, loss, params):
        """Efficient gradient computation using MPS-supported operations"""
        grads = {}
        eps = 1e-6  # Increased epsilon for better stability
        max_grad_norm = 0.1  # Reduced for stability
        batch_size = 32  # Smaller batch size for stability

        def stabilize_value(value):
            """Stabilize complex value computation"""
            if torch.isnan(value) or torch.isinf(value):
                return torch.zeros_like(value)
            return value

        for name, param in params.items():
            print(f"Computing gradient for: {name}")
            if isinstance(param, torch.Tensor):
                grad = torch.zeros_like(param, dtype=torch.complex64, device="mps")

                if param.dtype == torch.complex64:
                    # Sample fewer indices for stability
                    num_rows = param.shape[0]
                    sample_size = min(100, num_rows)  # Reduced sample size
                    indices = torch.randperm(num_rows, device="mps")[:sample_size]
                    
                    # Process in smaller batches
                    for batch_start in range(0, len(indices), batch_size):
                        batch_end = min(batch_start + batch_size, len(indices))
                        batch_indices = indices[batch_start:batch_end]
                        
                        with torch.no_grad():
                            original_values = param[batch_indices].clone()
                            
                            if name == 'token_embeddings':
                                # Compute gradients with stability checks
                                param.data[batch_indices] += eps
                                loss_plus = stabilize_value(model.forward(model.last_input).mean())
                                
                                param.data[batch_indices] -= 2 * eps
                                loss_minus = stabilize_value(model.forward(model.last_input).mean())
                                
                                param.data[batch_indices] = original_values
                                
                                # Compute stable gradients
                                grad_value = stabilize_value((loss_plus - loss_minus) / (2 * eps))
                                grad[batch_indices] = grad_value.view(-1, 1) * torch.ones_like(param[batch_indices])
                                
                            elif name == 'position_embeddings':
                                # Similar stabilized computation for position embeddings
                                param.data[batch_indices] += eps
                                loss_plus = stabilize_value(model.forward(model.last_input).mean())
                                
                                param.data[batch_indices] -= 2 * eps
                                loss_minus = stabilize_value(model.forward(model.last_input).mean())
                                
                                param.data[batch_indices] = original_values
                                
                                grad_value = stabilize_value((loss_plus - loss_minus) / (2 * eps))
                                grad[batch_indices] = grad_value.view(-1, 1) * torch.ones_like(param[batch_indices])

                    # Scale and clip gradients
                    grad = grad * (num_rows / sample_size)
                    grad_norm = torch.sqrt(torch.sum(torch.abs(grad)**2))
                    if grad_norm > max_grad_norm:
                        grad = grad * (max_grad_norm / grad_norm)

                grads[name] = grad

        return grads

    def _check_parameter_health(self, param, name):
        """Check parameter health and report statistics"""
        with torch.no_grad():
            # Compute statistics using real and imaginary parts
            real = param.real
            imag = param.imag
            magnitude = torch.sqrt(real**2 + imag**2)
            
            # Compute statistics on magnitude
            norm = torch.sqrt(torch.sum(magnitude**2))
            mean = torch.mean(magnitude)
            max_val = torch.max(magnitude)
            has_nan = torch.isnan(magnitude).any()
            has_inf = torch.isinf(magnitude).any()
            
            print(f"Parameter {name} stats:")
            print(f"  Magnitude norm: {norm}")
            print(f"  Magnitude mean: {mean}")
            print(f"  Magnitude max: {max_val}")
            print(f"  Has NaN: {has_nan}")
            print(f"  Has Inf: {has_inf}")
            
            # Additional phase statistics
            phase = torch.atan2(imag.cpu(), real.cpu()).to("mps")
            phase_mean = torch.mean(torch.abs(phase))
            phase_std = torch.std(phase)
            print(f"  Phase mean abs: {phase_mean}")
            print(f"  Phase std: {phase_std}")
            
            return not (has_nan or has_inf)

class EnhancedQuantumLLM(nn.Module):
    """Complete enhanced quantum-inspired language model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.tokenizer = EnhancedQuantumTokenizer(
            max_sequence_length=config['max_sequence_length'],
            tokenizer_name=config.get('tokenizer_name', 'bert-base-uncased')
        )
        
        # Update vocab_size based on the tokenizer
        self.config['vocab_size'] = len(self.tokenizer.vocab)
        
        self.attention_layers = nn.ModuleList([
            EnhancedQuantumAttention(
                dim=config['dim'],
                num_heads=config['num_heads'],
                max_sequence_length=config['max_sequence_length']
            )
            for _ in range(config['num_layers'])
        ])
        
        self.universal_patterns = EnhancedUniversalPatterns(
            dim=config['dim']
        )
        
        # Initialize embeddings
        self.token_embeddings = nn.Parameter(torch.zeros(
            (self.config['vocab_size'], self.config['dim']),
            dtype=torch.complex64, device="mps"
        ))
        self.position_embeddings = nn.Parameter(torch.zeros(
            (self.config['max_sequence_length'], self.config['dim']),
            dtype=torch.complex64, device="mps"
        ))
        
        # Initialize with quantum states
        self._initialize_enhanced_embeddings()
        
        # For training
        self.last_input = None
    
    def _initialize_enhanced_embeddings(self):
        """Initialize embeddings with quantum-inspired patterns"""
        with torch.no_grad():
            for i in range(self.config['vocab_size']):
                self.token_embeddings[i] = self._generate_quantum_state(i)
            
            for pos in range(self.config['max_sequence_length']):
                self.position_embeddings[pos] = self._generate_position_state(pos)
    
    def _generate_quantum_state(self, index):
        """Generate quantum state for token embedding"""
        state = torch.zeros(self.config['dim'], dtype=torch.complex64, device="mps")
        
        # Convert scalar values to tensors on MPS
        phase = torch.tensor(2 * np.pi * index / self.config['vocab_size'], device="mps")
        indices = torch.arange(self.config['dim'], device="mps")
        theta = torch.pi * indices / self.config['dim']
        
        # Combine phase and theta while maintaining quantum properties
        combined_phase = phase + theta
        # Create complex exponential using Euler's formula with tensors
        state = torch.exp(1j * combined_phase.to(torch.float32))
        
        # Normalize the quantum state
        norm = torch.sqrt(torch.sum(torch.abs(state)**2))
        state = state / norm
        
        return state
    
    def _generate_position_state(self, pos):
        """Generate quantum state for position embedding"""
        state = torch.zeros(self.config['dim'], dtype=torch.complex64, device="mps")
        
        # Convert scalar values to tensors on MPS
        phase = torch.tensor(2 * np.pi * pos / self.config['max_sequence_length'], device="mps")
        indices = torch.arange(self.config['dim'], device="mps")
        theta = torch.pi * indices / self.config['dim']
        
        # Combine phase and theta while maintaining quantum properties
        combined_phase = phase + theta
        # Create complex exponential using Euler's formula with tensors
        state = torch.exp(1j * combined_phase.to(torch.float32))
        
        # Normalize the quantum state
        norm = torch.sqrt(torch.sum(torch.abs(state)**2))
        state = state / norm
        
        return state
    
    def forward(self, input_ids, training=True):
        """Forward pass through the model"""
        self.last_input = input_ids

        # Move input to MPS device
        input_ids = input_ids.to("mps")

        # Get embeddings
        token_embeds = self.token_embeddings[input_ids]  # [batch, seq_len, dim]
        pos_embeds = self.position_embeddings[:input_ids.shape[1]].unsqueeze(0)  # [1, seq_len, dim]
        x = token_embeds + pos_embeds  # [batch, seq_len, dim]

        # Process through layers
        attention_mask = self._create_attention_mask(input_ids.shape[1])
        
        for i in range(len(self.attention_layers)):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            
            # Reshape for attention - split heads properly
            head_dim = self.config['dim'] // self.config['num_heads']
            
            # Reshape and permute operations for attention
            # [batch, seq_len, dim] -> [batch, seq_len, heads, head_dim]
            q = x.view(batch_size, seq_len, self.config['num_heads'], head_dim)
            k = x.view(batch_size, seq_len, self.config['num_heads'], head_dim)
            v = x.view(batch_size, seq_len, self.config['num_heads'], head_dim)
            
            # [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            # Apply attention
            attention_output = self.attention_layers[i](q, k, v, mask=attention_mask)
            
            # Reshape back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, heads, head_dim]
            attention_output = attention_output.permute(0, 2, 1, 3)
            
            # [batch, seq_len, heads, head_dim] -> [batch, seq_len, dim]
            attention_output = attention_output.reshape(batch_size, seq_len, -1)
            
            # Residual connection
            x = x + attention_output
            
            # Apply universal patterns
            x = self.universal_patterns.forward(x)  # Make sure to use .forward() explicitly
            
            if training:
                # Apply quantum dropout
                x = self._quantum_dropout(x, rate=0.1)
        
        # Project to vocabulary - handle complex values
        x_real = x.real
        x_imag = x.imag
        embeddings_real = self.token_embeddings.real
        embeddings_imag = self.token_embeddings.imag
        
        # Compute complex matrix multiplication manually
        logits_real = torch.matmul(x_real, embeddings_real.T) - torch.matmul(x_imag, embeddings_imag.T)
        logits_imag = torch.matmul(x_real, embeddings_imag.T) + torch.matmul(x_imag, embeddings_real.T)
        
        logits = torch.complex(logits_real, logits_imag)
        return logits
    
    def _create_attention_mask(self, seq_length):
        """Create causal attention mask"""
        # Create mask on correct device
        mask = torch.tril(torch.ones((seq_length, seq_length), device="mps"))
        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask
    
    def _calculate_angle(self, x):
        """Calculate angle of complex number using MPS-supported operations"""
        # Get real and imaginary parts
        real = x.real
        imag = x.imag
        
        # Use atan2 on CPU (temporarily) as it's not available on MPS
        angle = torch.atan2(imag.cpu(), real.cpu())
        return angle.to("mps")
    
    def _quantum_dropout(self, x, rate=0.1):
        """Apply quantum-inspired dropout"""
        if rate == 0:
            return x
        
        # Split into real and imaginary parts for MPS-compatible operations
        real = x.real
        imag = x.imag
        
        # Calculate phase using our helper function
        phase = self._calculate_angle(x)
        
        # Create dropout mask on the same device
        mask = (torch.rand(x.shape, device="mps") > rate) / (1 - rate)
        
        # Apply mask and recombine with phase
        magnitude = torch.sqrt(real**2 + imag**2) * mask
        
        # Recombine using magnitude and phase
        new_real = magnitude * torch.cos(phase)
        new_imag = magnitude * torch.sin(phase)
        
        return torch.complex(new_real, new_imag)
    
    def generate(self, prompt, max_length=100, temperature=0.8):
        """Generate text using the model"""
        def safe_normalize(x, dim=-1, eps=1e-8):
            """Safely normalize vector to unit length."""
            x_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True)) + eps
            return x / x_norm

        def get_next_token_probs(logits, temp=1.0):
            """Get stable probability distribution for next token."""
            # Get magnitudes of complex logits
            magnitudes = torch.sqrt(logits.real**2 + logits.imag**2)
            
            # Apply temperature and subtract max for stability
            scaled_magnitudes = magnitudes / temp
            scaled_magnitudes = scaled_magnitudes - scaled_magnitudes.max()
            
            # Compute stable softmax
            exp_magnitudes = torch.exp(scaled_magnitudes)
            probs = exp_magnitudes / (exp_magnitudes.sum() + 1e-10)
            
            # Ensure valid probability distribution
            probs = torch.nan_to_num(probs, 0.0)
            if torch.sum(probs) < 1e-10:
                # If all probs are too small, use uniform distribution
                probs = torch.ones_like(probs) / probs.shape[0]
            else:
                probs = safe_normalize(probs, dim=-1)
            
            return probs

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, device="mps").unsqueeze(0)
        generated = list(input_ids[0].cpu().numpy())
        
        # Generation loop
        for i in range(max_length):
            # Get predictions
            if len(generated) > self.config['max_sequence_length']:
                context = generated[-self.config['max_sequence_length']:]
            else:
                context = generated
            
            # Forward pass
            with torch.no_grad():
                input_ids = torch.tensor(context, device="mps").unsqueeze(0)
                logits = self.forward(input_ids, training=False)
                next_token_logits = logits[0, -1]
                
                # Get probability distribution
                probs = get_next_token_probs(next_token_logits, temp=temperature)
                
                # Sample next token with fallbacks
                try:
                    # Try multinomial sampling
                    next_token = torch.multinomial(probs, 1).item()
                except RuntimeError:
                    try:
                        # Fallback 1: Try with re-normalized probabilities
                        probs = safe_normalize(torch.abs(probs))
                        next_token = torch.multinomial(probs, 1).item()
                    except RuntimeError:
                        # Fallback 2: Use argmax
                        print("Warning: Falling back to argmax sampling")
                        next_token = torch.argmax(probs).item()
                
                # Debug info for first token
                if i == 0:
                    print("\nGeneration debugging:")
                    print(f"Probability sum: {probs.sum().item():.6f}")
                    print(f"Max probability: {probs.max().item():.6f}")
                    print(f"Number of non-zero probs: {(probs > 0).sum().item()}")
                    top_k = 5
                    top_probs, top_indices = torch.topk(probs, top_k)
                    print("\nTop {} tokens:".format(top_k))
                    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                        token = self.tokenizer.decode([idx])
                        print(f"Token: {token}, Probability: {prob:.6f}")
                
                # Stop if EOS token or probability is degenerate
                if next_token == self.tokenizer.vocab.get('<EOS>', -1) or probs.max() > 0.9:
                    break
                
                generated.append(next_token)
        
        # Decode and return generated text
        return self.tokenizer.decode(generated)

def train_model(model, train_data_path, output_dir, config):
    """Train the enhanced quantum-inspired model"""
    print("Starting training...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize optimizer
    optimizer = EnhancedQuantumOptimizer(
        learning_rate=config['learning_rate']
    )
    
    # Load training data
    with open(train_data_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    # Training loop
    global_step = 0
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        for i in tqdm(range(0, len(texts), config['batch_size'])):
            batch_texts = texts[i:i + config['batch_size']]
            
            # Prepare batch
            print("Tokenizing batch...")
            start_time = time.time()
            # Convert tokenizer output directly to tensor
            batch_tokens = [model.tokenizer.encode(text) for text in batch_texts]
            batch_tokens = [t for t in batch_tokens if len(t) > 1]
            end_time = time.time()
            print(f"Tokenization time: {end_time - start_time:.4f} seconds")

            if not batch_tokens:
                continue

            # Pad sequences using PyTorch operations
            max_len = min(max(len(t) for t in batch_tokens), model.config['max_sequence_length'] - 1)
            batch = torch.zeros((len(batch_tokens), max_len + 1), dtype=torch.long, device="mps")
            
            # Convert and assign tokens properly
            for j, tokens in enumerate(batch_tokens):
                token_tensor = torch.tensor(tokens[:max_len + 1], dtype=torch.long, device="mps")
                batch[j, :len(token_tensor)] = token_tensor

            print(f"Batch shape: {batch.shape}")

            # Forward pass and optimize
            print("Forward pass...")
            start_time = time.time()
            logits = model.forward(batch[:, :-1])
            end_time = time.time()
            print(f"Forward pass time: {end_time - start_time:.4f} seconds")

            print(f"Logits shape: {logits.shape}")

            # Compute loss with proper tensor types
            targets = batch[:, 1:].to(device="mps")
            loss = compute_loss(logits, targets)

            print("Optimizer step...")
            start_time = time.time()
            optimizer.step(model, loss)
            end_time = time.time()
            print(f"Optimizer step time: {end_time - start_time:.4f} seconds")

            if global_step % 100 == 0:
                print(f"\nStep {global_step}, Loss: {loss:.4f}")

            if global_step % config['save_steps'] == 0:
                save_checkpoint(model, optimizer, global_step, output_dir)

            if global_step % 100 == 0:
                print("\nChecking parameter health...")
                optimizer._check_parameter_health(model.token_embeddings, "token_embeddings")
                optimizer._check_parameter_health(model.position_embeddings, "position_embeddings")

            global_step += 1

        # Save epoch checkpoint
        save_checkpoint(model, optimizer, f"epoch_{epoch+1}", output_dir)

    print("Training completed!")

def compute_loss(logits, targets):
    """Compute cross entropy loss with quantum phase alignment and stabilization"""
    def stabilize_complex(tensor, eps=1e-8):
        magnitude = torch.sqrt(tensor.real**2 + tensor.imag**2)
        magnitude = torch.clamp(magnitude, min=eps)
        phase = torch.atan2(tensor.imag.cpu(), tensor.real.cpu()).to("mps")
        return torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    
    # Ensure inputs are on the correct device and stabilized
    logits = stabilize_complex(logits.to("mps"))
    targets = targets.to("mps")
    
    # Flatten logits and targets
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.flatten()
    
    # Split complex values
    real = logits_flat.real
    imag = logits_flat.imag
    
    # Compute probabilities using quantum-inspired softmax
    probs = quantum_softmax(logits_flat)
    
    # Move to CPU for complex operations
    probs_cpu = probs.to("cpu")
    targets_cpu = targets_flat.to("cpu")
    indices = torch.arange(len(targets_flat)).to("cpu")
    
    # Get probabilities for target tokens
    target_probs = probs_cpu[indices, targets_cpu]
    
    # Compute cross-entropy loss
    loss = -torch.mean(torch.log(torch.abs(target_probs) + 1e-10))
    
    # Align quantum phases using CPU for atan2
    phase_diff = (torch.atan2(logits_flat.imag.cpu(), logits_flat.real.cpu()) - 
                 torch.atan2(probs_cpu.imag, probs_cpu.real))
    phase_factor = torch.mean(torch.abs(torch.cos(phase_diff)))
    
    # Move back to MPS for final computation
    loss = loss.to("mps") * phase_factor.to("mps")
    
    return loss

def quantum_softmax(logits):
    """Apply quantum-inspired softmax with enhanced stability"""
    def safe_normalize(x, dim=-1, eps=1e-8):
        x_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True)) + eps
        return x / x_norm
    
    # Get real and imaginary parts
    real = logits.real
    imag = logits.imag
    
    # Compute magnitudes with stability
    magnitudes = torch.sqrt(real**2 + imag**2 + 1e-8)
    phases = torch.atan2(imag.cpu(), real.cpu()).to("mps")
    
    # Stable softmax computation
    max_mag = torch.max(magnitudes)
    exp_mags = torch.exp(magnitudes - max_mag)
    probs = safe_normalize(exp_mags)
    
    # Combine with phases
    real_probs = probs * torch.cos(phases)
    imag_probs = probs * torch.sin(phases)
    
    return torch.complex(real_probs, imag_probs)

def save_checkpoint(model, optimizer, step, output_dir):
    """Save model checkpoint"""
    checkpoint = {
        'step': step,
        'model_state': {
            'token_embeddings': model.token_embeddings.to("cpu"),  # Move to CPU for saving
            'position_embeddings': model.position_embeddings.to("cpu"),
            'config': model.config
        },
        'optimizer_state': {
            key: value.to("cpu") if isinstance(value, torch.Tensor) else value
            for key, value in optimizer.__dict__.items()
        }
    }
    
    save_path = output_dir / f'checkpoint_{step}.pt'  # Use .pt extension for PyTorch
    torch.save(checkpoint, save_path)

### Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'vocab_size': 32000,
        'dim': 1024,
        'num_heads': 16,
        'num_layers': 24,
        'max_sequence_length': 2048,
        'batch_size': 32,
        'epochs': 3,
        'learning_rate': 1e-4,
        'save_steps': 1000
    }
    
    # Create sample training data
    sample_text = """This is a sample text for training.
    It contains multiple lines of text.
    The quantum-inspired model will learn from this data."""
    
    with open('sample_training.txt', 'w') as f:
        f.write(sample_text)
    
    # Initialize model
    model = EnhancedQuantumLLM(config)
    
    # Train model
    train_model(
        model=model,
        train_data_path='sample_training.txt',
        output_dir='quantum_model_checkpoints',
        config=config
    )
    
    # Generate text
    prompt = "Once upon a time"
    generated_text = model.generate(prompt, max_length=100)
    print(f"\nGenerated text:\n{generated_text}")