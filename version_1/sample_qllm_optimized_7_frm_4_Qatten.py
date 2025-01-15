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
import re
# from quantum_language_core import QuantumTokenizer, QuantumLanguageStructure

# Determine the best available device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Update environment variables for MPS if applicable
if device == "mps":
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.5'  # Use 50% of available memory
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.3'   # Free memory when usage goes below 30%

# Model Architecture Configuration
MODEL_DIM = 256  # Base model dimension
NUM_HEADS = 8  # Total number of attention heads
HEAD_DIM = MODEL_DIM // NUM_HEADS  # Dimension per head
QUANTUM_HEADS = 2  # Number of quantum attention heads
TRADITIONAL_HEADS = NUM_HEADS - QUANTUM_HEADS  # Number of traditional attention heads
NUM_LAYERS = 4  # Number of transformer layers

# Sequence and Processing Configuration
MAX_SEQUENCE_LENGTH = 256  # Maximum sequence length
CHUNK_SIZE = 64  # Size of chunks for processing
TRUNCATION_SIDE = 'right'  # Which side to truncate sequences ('left' or 'right')
PADDING_SIDE = 'right'  # Which side to add padding ('left' or 'right')

# Training Configuration
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_SAMPLES = 200
GRADIENT_CLIP = 0.5
LOG_EVERY = 5

def ensure_tensor_size(tensor, target_size, mode='pad'):
    """Ensure tensor matches target size by padding or truncating.
    
    Args:
        tensor (torch.Tensor): Input tensor
        target_size (tuple): Desired size
        mode (str): 'pad' or 'truncate'
    
    Returns:
        torch.Tensor: Resized tensor
    """
    current_size = tensor.size()
    
    if len(current_size) != len(target_size):
        raise ValueError(f"Dimension mismatch: {current_size} vs {target_size}")
        
    result = tensor
    for dim, (current, target) in enumerate(zip(current_size, target_size)):
        if current < target:
            if mode == 'pad':
                pad_size = [0] * (2 * len(current_size))
                pad_size[2*dim + 1] = target - current
                result = F.pad(result, pad_size)
        elif current > target:
            if mode == 'truncate':
                slicing = [slice(None)] * len(current_size)
                slicing[dim] = slice(0, target)
                result = result[slicing]
    
    return result

def validate_attention_sizes(q, k, v):
    """Validate and adjust attention input sizes.
    
    Args:
        q, k, v (torch.Tensor): Query, Key, Value tensors
    
    Returns:
        tuple: Adjusted (q, k, v) tensors
    """
    # Check batch sizes match
    if not (q.size(0) == k.size(0) == v.size(0)):
        raise ValueError(f"Batch sizes don't match: {q.size(0)}, {k.size(0)}, {v.size(0)}")
    
    # Ensure sequence lengths are compatible
    seq_len = min(q.size(1), k.size(1), v.size(1))
    q = q[:, :seq_len]
    k = k[:, :seq_len]
    v = v[:, :seq_len]
    
    # Ensure head dimensions match
    head_dim = min(q.size(-1), k.size(-1), v.size(-1))
    q = q[..., :head_dim]
    k = k[..., :head_dim]
    v = v[..., :head_dim]
    
    return q, k, v

def chunk_sequence(tensor, chunk_size):
    """Split sequence into chunks for efficient processing.
    
    Args:
        tensor (torch.Tensor): Input tensor [B, L, D]
        chunk_size (int): Size of chunks
    
    Returns:
        list: List of tensor chunks
    """
    B, L, D = tensor.shape
    chunks = []
    
    for i in range(0, L, chunk_size):
        end_idx = min(i + chunk_size, L)
        chunk = tensor[:, i:end_idx].contiguous()
        chunks.append(chunk)
    
    return chunks

class QuantumTokenizer:
    """Quantum-inspired tokenizer using wave patterns and universal constants"""
    def __init__(self, vocab_size=None):
        # Initialize core components
        self.vocab = self._initialize_vocabulary()
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Constants
        self.phi = (1 + np.sqrt(5)) / 2
        self.constants = {'e': np.e, 'pi': np.pi}
        
        # Configuration
        self.eos_token_id = 3
        self.padding_side = PADDING_SIDE
        self.truncation_side = TRUNCATION_SIDE
        self.model_max_length = MAX_SEQUENCE_LENGTH
        
        # Initialize patterns once
        self._patterns = None
    
    @property
    def patterns(self):
        if self._patterns is None:
            self._patterns = self._initialize_quantum_patterns()
        return self._patterns
    
    def _initialize_quantum_patterns(self):
        patterns = {}
        # Use first 100 prime numbers for base patterns
        primes = self._generate_primes(100)
        
        for i, prime in enumerate(primes):
            # Create unique phase using golden ratio and prime numbers
            phase = 2 * self.constants['pi'] * (i / len(primes)) * self.phi
            patterns[prime] = np.exp(1j * phase)
        return patterns
    
    def _generate_primes(self, n):
        """Generate first n prime numbers"""
        primes = []
        num = 2
        while len(primes) < n:
            if all(num % prime != 0 for prime in primes):
                primes.append(num)
            num += 1
        return primes
    
    def _initialize_vocabulary(self):
        """Initialize vocabulary using wave-like patterns"""
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<MASK>': 4
        }
        
        # Add common words - this will be our initial known vocabulary
        common_words = """the be to of and a in that have I it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us""".split()
        
        # Add each word to vocabulary
        for word in common_words:
            vocab[word] = len(vocab)
            
        return vocab
    
    def extend_vocabulary(self, new_words):
        """Extend vocabulary with new words"""
        for word in new_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Update vocab size and reverse vocab
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        return self.vocab_size
    
    def _initialize_char_waves(self):
        """Initialize wave patterns for characters"""
        waves = {}
        for i, char in enumerate(self.vocab):
            # Create unique wave pattern for each character
            freq = (i + 1) * self.phi
            phase = 2 * self.constants['pi'] * (i / len(self.vocab))
            waves[char] = lambda x, f=freq, p=phase: np.sin(f * x + p)
        return waves
    
    def encode(self, text, return_tensors='pt', **kwargs):
        """Enhanced encode method with word-level tokenization"""
        tokens = self._quantum_tokenize(text)
        
        # Get token indices
        indices = []
        for token in tokens:
            idx = self.vocab.get(token, self.vocab['<UNK>'])  # Use UNK token id for unknown words
            indices.append(idx)
        
        # Convert to tensor if requested
        if return_tensors == 'pt':
            return torch.tensor([indices], dtype=torch.long)
        elif return_tensors == 'wave':
            # Return wave patterns for quantum processing
            encoded = torch.zeros((len(tokens), self.vocab_size), dtype=torch.float32)
            for i, token in enumerate(tokens):
                if token in self.vocab:
                    pattern = self._create_token_pattern(token)
                    encoded[i] = torch.tensor(pattern)
                else:
                    encoded[i] = self._create_unknown_pattern(token)
            return encoded
        return indices
    
    def decode(self, token_ids):
        """Decode token ids back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
            
        text = []
        for idx in token_ids:
            token = self.reverse_vocab.get(idx, '<UNK>')
            if token not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']:
                text.append(token)
        
        return ' '.join(text)
    
    def _quantum_tokenize(self, text):
        """Tokenize text using word-level tokenization with known words"""
        # Normalize text
        text = text.lower().strip()
        
        # Split into words
        words = text.split()
        
        # Convert to known tokens or UNK
        tokens = []
        for word in words:
            # Only use known words from vocabulary, otherwise use UNK
            if word in self.vocab:
                tokens.append(word)
            else:
                tokens.append('<UNK>')
        
        return tokens
    
    def _fibonacci_group(self, tokens):
        """Group tokens using Fibonacci sequence"""
        fib = [1, 1]
        while fib[-1] < len(tokens):
            fib.append(fib[-1] + fib[-2])
        
        grouped = []
        i = 0
        while i < len(tokens):
            # Use Fibonacci numbers to determine group sizes
            size = min(fib[int(np.log(len(tokens)-i)/np.log(self.phi))], len(tokens)-i)
            group = ''.join(tokens[i:i+size]).strip()
            if group:
                grouped.append(group)
            i += size
        
        return grouped
    
    def _create_token_pattern(self, token):
        """Create quantum wave pattern for token"""
        pattern = np.zeros(self.vocab_size)
        idx = self.vocab.get(token, 1)  # Use 1 for UNK
        
        # Create interference pattern
        for i in range(self.vocab_size):
            phase = 2 * self.constants['pi'] * (i - idx) / self.vocab_size
            pattern[i] = np.cos(phase * self.phi) * np.exp(-abs(i-idx)/self.vocab_size)
        
        return pattern
    
    def _create_unknown_pattern(self, token):
        """Create pattern for unknown tokens using quantum superposition"""
        pattern = np.zeros(self.vocab_size)
        
        # Create superposition of similar known tokens
        for known_token, idx in self.vocab.items():
            similarity = self._quantum_similarity(token, known_token)
            phase = 2 * self.constants['pi'] * similarity * self.phi
            pattern[idx] = np.cos(phase) * similarity
        
        return torch.tensor(pattern / np.sqrt(np.sum(pattern**2) + 1e-8))
    
    def _quantum_similarity(self, token1, token2):
        """Compute quantum-inspired similarity between tokens"""
        # Use Levenshtein distance with quantum weights
        distance = self._weighted_levenshtein(str(token1), str(token2))
        return np.exp(-distance * self.phi)
    
    def _weighted_levenshtein(self, s1, s2):
        """Compute weighted Levenshtein distance"""
        if len(s1) < len(s2):
            return self._weighted_levenshtein(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Use golden ratio for insertions/deletions
                insertions = previous_row[j + 1] + 1/self.phi
                deletions = current_row[j] + 1/self.phi
                # Use e for substitutions
                substitutions = previous_row[j] + (0 if c1 == c2 else 1/self.constants['e'])
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def __call__(self, text, padding=True, truncation=True, max_length=None, return_tensors=None):
        """Make the tokenizer callable like HuggingFace tokenizers"""
        if isinstance(text, str):
            text = [text]
        
        # Process each text in the batch
        batch_encoding = {
            'input_ids': [],
            'attention_mask': []
        }
        
        max_length = max_length or self.model_max_length
        
        for t in text:
            # Encode text to get token ids
            token_ids = self.encode(t, return_tensors=None)  # Get raw indices
            
            # Truncate if needed
            if truncation and len(token_ids) > max_length:
                if self.truncation_side == 'right':
                    token_ids = token_ids[:max_length]
                else:
                    token_ids = token_ids[-max_length:]
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Pad if needed
            if padding and len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                if self.padding_side == 'right':
                    token_ids = token_ids + [0] * pad_length
                    attention_mask = attention_mask + [0] * pad_length
                else:
                    token_ids = [0] * pad_length + token_ids
                    attention_mask = [0] * pad_length + attention_mask
            
            batch_encoding['input_ids'].append(token_ids)
            batch_encoding['attention_mask'].append(attention_mask)
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            batch_encoding['input_ids'] = torch.tensor(batch_encoding['input_ids'], dtype=torch.long)
            batch_encoding['attention_mask'] = torch.tensor(batch_encoding['attention_mask'], dtype=torch.long)
        
        return batch_encoding

class QuantumLanguageStructure:
    """Handle natural language structure using quantum-inspired patterns"""
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.fib_sequence = self._generate_fibonacci(512)
    
    def _generate_fibonacci(self, n):
        """Generate Fibonacci sequence"""
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return torch.tensor(fib, dtype=torch.float32)
    
    def apply_structure(self, tokens):
        """Apply natural language patterns based on Fibonacci ratios"""
        B, L, D = tokens.shape
        
        # Move sequence to correct device
        self.fib_sequence = self.fib_sequence.to(tokens.device)
        
        # Ensure we have enough Fibonacci numbers
        if L > len(self.fib_sequence):
            self.fib_sequence = self._generate_fibonacci(L * 2).to(tokens.device)
        
        # Get pattern for this sequence length with scaling
        pattern = (self.fib_sequence[:L] / (self.fib_sequence[L-1] + 1e-8))
        pattern = torch.clamp(pattern, -5, 5)  # Prevent extreme values
        
        # Reshape pattern to match token dimensions
        pattern = pattern.view(1, L, 1).expand(B, L, D)
        
        # Apply pattern with scaling and ensure output dimension matches input
        return (tokens * pattern * 0.1).to(tokens.device)

class QuantumOps:
    """Unified quantum operations with caching"""
    def __init__(self, vocab_size=None, dim=MODEL_DIM):
        self.cache = {}
        self.vocab_size = vocab_size
        self.dim = dim
        self._init_lookup_tables()
    
    def _init_lookup_tables(self):
        """Initialize common lookup tables"""
        steps = 2048
        self.angles = torch.linspace(0, 2 * math.pi, steps)
        self.sin_table = torch.sin(self.angles)
        self.cos_table = torch.cos(self.angles)
        self.phase_norm = torch.sqrt(torch.tensor([self.dim]))
    
    @staticmethod
    def fast_sin(x):
        return torch.sin(x)
    
    @staticmethod
    def quantum_interference(x, y):
        mixed = torch.tanh((x + y) * 0.5)
        similarity = F.cosine_similarity(x, y, dim=-1, eps=1e-8).unsqueeze(-1)
        return mixed * similarity
    
    @staticmethod
    def phase_encoding(x):
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        phase = torch.atan2(x_norm, torch.roll(x_norm, 1, dims=-1))
        return torch.sin(phase)
    
    def get_cached_state(self, key):
        return self.cache.get(key)
    
    def set_cached_state(self, key, value):
        self.cache[key] = value
        if len(self.cache) > 10000:  # Prevent memory issues
            self.cache.clear()

# Add this helper function for safe reshaping
def safe_reshape(tensor, shape):
    """Safely reshape tensor by making it contiguous first"""
    return tensor.contiguous().view(*shape)

class FastHybridQuantumAttention(nn.Module):
    def __init__(self, dim=MODEL_DIM, num_heads=NUM_HEADS, quantum_heads=QUANTUM_HEADS, 
                 chunk_size=CHUNK_SIZE, use_cuda_graphs=True):
        super().__init__()
        assert num_heads >= quantum_heads, "quantum_heads must be <= num_heads"
        
        # Initialize dimensions
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.quantum_heads = quantum_heads
        self.traditional_heads = num_heads - quantum_heads
        self.chunk_size = chunk_size
        
        self._init_components(use_cuda_graphs)
    
    def _init_components(self, use_cuda_graphs):
        """Initialize quantum and traditional components"""
        self.chunk_size = CHUNK_SIZE
        self.truncation_side = TRUNCATION_SIDE
        self.use_cuda_graphs = use_cuda_graphs and torch.cuda.is_available()
        
        if self.quantum_heads > 0:
            self._init_quantum_components()
        
        if self.traditional_heads > 0:
            self._init_traditional_components()
        
        self.to_out = nn.Linear(self.dim, self.dim)
    
    def _init_quantum_components(self):
        """Initialize quantum-specific components"""
        self.q_phase_shifts = nn.Parameter(torch.randn(self.quantum_heads) * 0.02)
        self.q_frequencies = nn.Parameter(torch.randn(self.quantum_heads) * 0.02)
        self.to_quantum_state = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 2),
            nn.LayerNorm(self.head_dim * 2),
            nn.Tanh()
        )
        self._init_quantum_lookup()
    
    def _init_traditional_components(self):
        """Initialize traditional attention components"""
        self.q_proj = nn.Linear(self.dim, self.traditional_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.traditional_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.traditional_heads * self.head_dim, bias=False)

    def _init_quantum_lookup(self):
        steps = 2048
        angles = torch.linspace(0, 2 * math.pi, steps)
        self.register_buffer('angle_steps', angles)
        self.register_buffer('sin_table', torch.sin(angles))
        self.register_buffer('cos_table', torch.cos(angles))
        self.register_buffer('phase_norm', torch.sqrt(torch.tensor([self.head_dim])))

    def _init_cuda_graphs(self):
        if not self.use_cuda_graphs:
            return
            
        self.cuda_graphs = {}
        common_lengths = [128, 256, 512, 1024]
        
        for seq_len in common_lengths:
            static_input = torch.zeros(1, seq_len, self.dim, device='cuda')
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.forward(static_input)
                with torch.cuda.graph(g):
                    static_output = self.forward(static_input)
            
            torch.cuda.current_stream().wait_stream(s)
            self.cuda_graphs[seq_len] = (g, static_input, static_output)

    def _process_quantum_chunk(self, q_chunk, k, v, head_idx):
        """Process quantum chunks with proper size handling"""
        # Ensure input sizes match
        q_chunk, k, v = validate_attention_sizes(q_chunk, k, v)
        B, L = q_chunk.size(0), q_chunk.size(1)
        
        # Reshape inputs to correct dimensions
        q_chunk = ensure_tensor_size(q_chunk, (B, L, self.head_dim))
        k = ensure_tensor_size(k, (B, L, self.head_dim))
        v = ensure_tensor_size(v, (B, L, self.head_dim))
        
        # Convert to quantum states - use safe_reshape
        q_flat = safe_reshape(q_chunk, (B * L, -1))
        k_flat = safe_reshape(k, (B * L, -1))
        
        quantum_chunk = self.to_quantum_state(q_flat)
        quantum_k = self.to_quantum_state(k_flat)
        
        # Split into amplitude and phase
        q_amp, q_phase = quantum_chunk.chunk(2, dim=-1)
        k_amp, k_phase = quantum_k.chunk(2, dim=-1)
        
        # Reshape back to 3D - use safe_reshape
        q_amp = safe_reshape(q_amp, (B, L, -1))
        q_phase = safe_reshape(q_phase, (B, L, -1))
        k_amp = safe_reshape(k_amp, (B, L, -1))
        k_phase = safe_reshape(k_phase, (B, L, -1))
        
        # Compute attention with proper dimensions
        attention = self._fast_quantum_interference(
            q_amp, q_phase, k_amp, k_phase, head_idx
        )
        
        # Ensure output has correct dimensions
        attention = F.softmax(attention / math.sqrt(self.head_dim), dim=-1)
        v_flat = safe_reshape(v, (B, L, -1))
        output = torch.matmul(attention, v_flat)
        
        return ensure_tensor_size(output, (B, L, self.head_dim))
    
    def _fast_quantum_interference(self, q_amp, q_phase, k_amp, k_phase, head_idx):
        """Compute quantum interference with proper reshaping"""
        # Compute phase difference
        phase_diff = (q_phase - k_phase.transpose(-2, -1)) * self.q_frequencies[head_idx]
        phase_diff = phase_diff + self.q_phase_shifts[head_idx]
        
        # Get interference pattern
        phase_indices = ((phase_diff / (2 * math.pi) * 2048) % 2048).long()
        interference = self.cos_table[phase_indices]
        
        # Compute attention scores
        scores = torch.matmul(q_amp, k_amp.transpose(-2, -1))
        return scores * interference
    
    def _process_traditional_attention(self, x, mask=None):
        """Process traditional attention heads"""
        B, L, _ = x.shape
        
        # Project inputs - use safe_reshape
        q = safe_reshape(self.q_proj(x), (B, L, self.traditional_heads, self.head_dim))
        k = safe_reshape(self.k_proj(x), (B, L, self.traditional_heads, self.head_dim))
        v = safe_reshape(self.v_proj(x), (B, L, self.traditional_heads, self.head_dim))
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        
        # Reshape output - use safe_reshape
        return safe_reshape(output.transpose(1, 2), (B, L, -1))
    
    def forward(self, x, mask=None):
        """Forward pass with proper size handling"""
        # Validate and adjust input size
        B, L, D = x.shape
        x = ensure_tensor_size(x, (B, min(L, self.chunk_size), self.dim))
        
        # Initialize output tensor
        output = torch.zeros(B, x.size(1), self.dim, device=x.device)
        
        # Process quantum heads
        if self.quantum_heads > 0:
            quantum_outputs = []
            for head in range(self.quantum_heads):
                head_dim_slice = slice(head * self.head_dim, (head + 1) * self.head_dim)
                x_head = x[..., head_dim_slice].contiguous()
                
                # Process in chunks
                chunks = chunk_sequence(x_head, self.chunk_size)
                chunk_outputs = []
                
                for chunk in chunks:
                    chunk_output = self._process_quantum_chunk(chunk, x_head, x_head, head)
                    chunk_outputs.append(chunk_output)
                
                # Combine chunk outputs
                head_output = torch.cat(chunk_outputs, dim=1)
                quantum_outputs.append(head_output)
            
            # Combine quantum head outputs - use safe_reshape for final combination
            quantum_output = torch.cat(quantum_outputs, dim=-1)
            output[..., :self.quantum_heads * self.head_dim] = quantum_output
        
        # Process traditional heads
        if self.traditional_heads > 0:
            traditional_output = self._process_traditional_attention(x, mask)
            output[..., self.quantum_heads * self.head_dim:] = traditional_output
        
        return self.to_out(output)

class QuantumStateCache:
    """Cache for quantum states to avoid recomputation"""
    def __init__(self):
        self.token_states = {}
        self.phase_states = {}
        self.max_cache_size = 10000
    
    def get_token_state(self, token_id):
        return self.token_states.get(token_id)
    
    def set_token_state(self, token_id, state):
        if len(self.token_states) > self.max_cache_size:
            self.token_states.clear()
        self.token_states[token_id] = state
    
    def clear(self):
        self.token_states.clear()
        self.phase_states.clear()

class QuantumPreprocessor:
    """Prepare quantum states from input tokens"""
    def __init__(self, vocab_size, dim, device):
        self.vocab_size = vocab_size
        self.dim = dim
        self.device = device
        self.phi = (1 + math.sqrt(5)) / 2
        
        # Initialize basis states with correct dimension
        self.basis_states = self._initialize_basis_states()
    
    def _initialize_basis_states(self):
        """Initialize quantum basis states for each token"""
        states = torch.zeros(self.vocab_size, self.dim, device=self.device)
        for i in range(self.vocab_size):
            phase = 2 * math.pi * (i / self.vocab_size) * self.phi
            # Generate state directly in correct dimension
            states[i] = torch.sin(torch.linspace(0, phase, self.dim, device=self.device))
        return states
    
    def prepare_state(self, token_ids):
        """Convert token IDs to quantum states"""
        B, L = token_ids.shape
        states = torch.zeros(B, L, self.dim, device=self.device)
        
        for b in range(B):
            for l in range(L):
                token_id = token_ids[b, l].item()
                if token_id < self.vocab_size:
                    states[b, l] = self.basis_states[token_id]
        
        # Normalize states
        states = states / (torch.norm(states, dim=-1, keepdim=True) + 1e-8)
        return states

class QuantumPipeline:
    """Process quantum states through multiple stages"""
    def __init__(self, dim, num_heads, device):
        self.dim = dim
        self.num_heads = num_heads
        self.device = device
        self.quantum_structure = QuantumLanguageStructure()
    
    def process_batch(self, states, cache):
        """Process quantum states through pipeline"""
        B, L, D = states.shape
        
        # Apply quantum structure while preserving dimensions
        states = self.quantum_structure.apply_structure(states)  # [B, L, D]
        
        # Apply quantum operations
        states = FastQuantumOps.phase_encoding(states)  # [B, L, D]
        
        # Ensure output dimension matches model dimension
        if D != self.dim:
            states = F.linear(
                states,
                torch.eye(self.dim, device=states.device)[:D]
            )  # Project to correct dimension
        
        return states

class QuantumStageProcessor:
    """Manages multi-stage quantum processing"""
    def __init__(self, vocab_size, dim, num_heads, device):
        self.preprocessor = QuantumPreprocessor(vocab_size, dim, device)
        self.cache = QuantumStateCache()
        self.pipeline = QuantumPipeline(dim, num_heads, device)
        self.device = device
        self.dim = dim
    
    def process_batch(self, batch_data):
        """Process a batch through all quantum stages"""
        B, L = batch_data.shape
        
        # Stage 1: Prepare base states
        base_states = self.preprocessor.prepare_state(batch_data)  # [B, L, dim]
        
        # Stage 2: Process through quantum pipeline
        processed_states = self.pipeline.process_batch(base_states, self.cache)
        
        return processed_states

class FastQuantumLLMOptimized(nn.Module):
    """Optimized quantum language model with shared cache"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = device
        
        # Initialize tokenizer and quantum ops
        self.tokenizer = QuantumTokenizer()
        self.quantum_ops = QuantumOps(vocab_size=self.tokenizer.vocab_size, dim=config['dim'])
        
        # Initialize model components
        self.token_embedding = nn.Embedding(self.tokenizer.vocab_size, config['dim'])
        self.quantum_structure = QuantumLanguageStructure()
        
        # Initialize attention layers
        self.attention_layers = nn.ModuleList([
            FastHybridQuantumAttention(
                dim=config['dim'],
                num_heads=config['num_heads'],
                quantum_heads=config['quantum_heads'],
                chunk_size=config['chunk_size']
            )
            for _ in range(config['num_layers'])
        ])
        
        # Initialize normalization and output layers
        self.layer_norm = nn.LayerNorm(config['dim'])
        self.input_norm = nn.LayerNorm(config['dim'])
        self.output_norm = nn.LayerNorm(config['dim'])
        self.output_proj = nn.Linear(config['dim'], self.tokenizer.vocab_size)
        
        # Move model to correct device
        self.to(self.device)

    def forward(self, input_ids, return_loss=True):
        """Forward pass with quantum processing"""
        # Get embeddings
        x = self.token_embedding(input_ids)
        
        # Process through quantum pipeline
        quantum_states = self.quantum_ops.phase_encoding(x)
        x = x + quantum_states * 0.1
        
        # Process through attention layers
        x = self.input_norm(x)
        for layer in self.attention_layers:
            residual = x
            attn = layer(x)
            x = self.quantum_ops.quantum_interference(residual, attn)
            x = self.layer_norm(x)
        
        # Final processing
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        if return_loss and input_ids is not None:
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1
            )
            return loss
        
        return logits

# Add new dataset handling classes
class TextDataset(Dataset):
    """Custom dataset for handling text data with quantum tokenizer"""
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = list(dataset)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = tokenizer.vocab_size
    
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
        
        # Ensure token ids are within vocabulary bounds
        input_ids = encoding['input_ids'].squeeze(0)
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        return {
            'input_ids': input_ids,
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
    """Train the model with hybrid quantum-classical processing"""
    print("Loading dataset...")
    clear_memory()
    
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
    
    # Optimized data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        eps=1e-8
    )
    
    # Training loop with checkpointing
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            try:
                input_ids = batch['input_ids'].to(device)
                
                optimizer.zero_grad()
                loss = model(input_ids)
                
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Save checkpoint if best loss
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        save_checkpoint(model, optimizer, epoch, config['output_dir'], is_best=True)
                    
                    if batch_idx % config['log_every'] == 0:
                        print(f"\nBatch {batch_idx}, Loss: {loss.item():.4f}")
                
                # Regular checkpoint
                if batch_idx % 1000 == 0:
                    save_checkpoint(model, optimizer, epoch, config['output_dir'])
                
                clear_memory()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                clear_memory()
                continue
        
        # Epoch summary
        if num_batches > 0:
            avg_epoch_loss = total_loss / num_batches
            print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        clear_memory()

def save_checkpoint(model, optimizer, epoch, output_dir, is_best=False):
    """Save model checkpoint with quantum attention states"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config,
        'quantum_heads': model.attention_layers[0].quantum_heads,
        'traditional_heads': model.attention_layers[0].traditional_heads
    }
    
    path = Path(output_dir)
    path.mkdir(exist_ok=True)
    
    # Save regular checkpoint
    torch.save(checkpoint, path / f'checkpoint_epoch_{epoch}.pt')
    
    # Save best model if applicable
    if is_best:
        torch.save(checkpoint, path / 'best_model.pt')

def load_checkpoint(model, checkpoint_path):
    """Load checkpoint with quantum attention states"""
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Verify quantum attention configuration
    if (checkpoint.get('quantum_heads') != model.attention_layers[0].quantum_heads or
        checkpoint.get('traditional_heads') != model.attention_layers[0].traditional_heads):
        print("Warning: Checkpoint has different quantum/traditional head configuration")
    
    return checkpoint['epoch']

def main(mode, config, prompt=None, checkpoint=None, max_length=50, temperature=0.1):
    # Initialize model
    model = FastQuantumLLMOptimized(config)
    model = model.to(model.device)
    
    if mode == 'train':
        print("Training model with hybrid quantum-classical processing...")
        train_model(model, config)
    else:
        # Load checkpoint if provided
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                print(f"Loading checkpoint from {checkpoint_path}")
                epoch = load_checkpoint(model, checkpoint_path)
                print(f"Resumed from epoch {epoch}")
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
        
        if not prompt:
            print("Error: --prompt is required when mode is 'generate'")
            return
        
        print(f"\nGenerating text with prompt: {prompt}")
        print(f"Temperature: {temperature}, Max Length: {max_length}")
        
        # Generate text
        with torch.no_grad():
            input_ids = model.tokenizer.encode(prompt, return_tensors='pt').to(device)
            generated = input_ids[0].tolist()
            
            for _ in range(max_length):
                # Get predictions
                inputs = torch.tensor([generated[-model.config['max_sequence_length']:]]).to(device)
                outputs = model(inputs, return_loss=False)
                next_token_logits = outputs[0, -1, :] / temperature
                
                # Apply top-k filtering
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample next token
                next_token_id = top_k_indices[torch.multinomial(probs, num_samples=1)]
                generated.append(next_token_id.item())
                
                # Stop if we generate EOS token
                if next_token_id == model.tokenizer.eos_token_id:
                    break
        
        # Decode and print generated text
        generated_text = model.tokenizer.decode(generated)
        print(f"\nGenerated text:\n{generated_text}")

def get_default_config():
    """Return default configuration for the model"""
    return {
        # Model dimensions
        'dim': MODEL_DIM,
        'num_heads': NUM_HEADS,
        'head_dim': HEAD_DIM,
        'num_layers': NUM_LAYERS,
        'max_sequence_length': MAX_SEQUENCE_LENGTH,
        'quantum_heads': QUANTUM_HEADS,
        
        # Training parameters
        'batch_size': BATCH_SIZE,
        'epochs': 3,
        'learning_rate': LEARNING_RATE,
        'output_dir': 'quantum_checkpoints_7',
        'tokenizer_name': 'bert-base-uncased',
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-103-v1',
        'max_samples': MAX_SAMPLES,
        'log_every': LOG_EVERY,
        'gradient_clip': GRADIENT_CLIP,
        
        # Quantum processing
        'quantum_dim': MODEL_DIM,
        'chunk_size': CHUNK_SIZE,
    }

def run_from_cli():
    """Function to handle command-line interface"""
    parser = argparse.ArgumentParser(description='Quantum Language Model Training and Generation')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True,
                      help='Mode to run the model: train or generate')
    
    # Generation arguments
    parser.add_argument('--prompt', type=str,
                      help='Text prompt for generation (required if mode is generate)')
    parser.add_argument('--max_length', type=int, default=50,
                      help='Maximum length of generated text (default: 50)')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for text generation (default: 0.7)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate (default: 1e-4)')
    parser.add_argument('--max_samples', type=int, default=200,
                      help='Maximum number of training samples (default: 200)')
    
    # Model loading/saving
    parser.add_argument('--checkpoint', type=str,
                      help='Path to model checkpoint file')
    parser.add_argument('--output_dir', type=str, default='quantum_checkpoints_7',
                      help='Directory for saving checkpoints (default: quantum_checkpoints)')
    
    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'generate' and not args.prompt:
        parser.error("--prompt is required when mode is 'generate'")

    # Get default config and update with CLI arguments
    config = get_default_config()
    config.update({
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'output_dir': args.output_dir,
        'max_samples': args.max_samples,
    })

    # Run the model
    if args.mode == 'train':
        print("Training model...")
        print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}, Max samples: {args.max_samples}")
        main(mode='train', config=config, checkpoint=args.checkpoint)
    else:
        print("Generating text...")
        print(f"Prompt: {args.prompt}")
        print(f"Temperature: {args.temperature}, Max length: {args.max_length}")
        main(mode='generate', 
             config=config,
             prompt=args.prompt,
             checkpoint=args.checkpoint,
             max_length=args.max_length,
             temperature=args.temperature)

def process_batch_efficiently(batch, model, optimizer=None, training=True):
    """Process batch with automatic memory management"""
    try:
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            if training and optimizer:
                optimizer.zero_grad()
                loss = model(batch['input_ids'].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                return loss.item()
            else:
                with torch.no_grad():
                    return model(batch['input_ids'].to(device), return_loss=False)
    finally:
        # Clean up memory
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

if __name__ == "__main__":
    # Only run CLI handling when script is run directly
    try:
        run_from_cli()
    except SystemExit:
        # Ignore SystemExit in interactive environments
        pass 