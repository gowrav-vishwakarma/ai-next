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

class QuantumTokenizer:
    """Quantum-inspired tokenizer using wave patterns and universal constants"""
    def __init__(self, vocab_size=None):  # Make vocab_size optional
        # Initialize vocabulary first to determine actual size
        self.vocab = self._initialize_vocabulary()
        # Set vocab_size based on actual vocabulary
        self.vocab_size = len(self.vocab)
        
        self.phi = (1 + np.sqrt(5)) / 2
        self.e = np.e
        self.pi = np.pi
        
        # Initialize quantum patterns
        self.token_patterns = self._initialize_quantum_patterns()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.char_waves = self._initialize_char_waves()
        self.eos_token_id = 3  # <EOS> token
        self.padding_side = 'right'
        self.truncation_side = 'right'
        self.model_max_length = 512
    
    def _initialize_quantum_patterns(self):
        patterns = {}
        # Use first 100 prime numbers for base patterns
        primes = self._generate_primes(100)
        
        for i, prime in enumerate(primes):
            # Create unique phase using golden ratio and prime numbers
            phase = 2 * self.pi * (i / len(primes)) * self.phi
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
            phase = 2 * self.pi * (i / len(self.vocab))
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
            phase = 2 * self.pi * (i - idx) / self.vocab_size
            pattern[i] = np.cos(phase * self.phi) * np.exp(-abs(i-idx)/self.vocab_size)
        
        return pattern
    
    def _create_unknown_pattern(self, token):
        """Create pattern for unknown tokens using quantum superposition"""
        pattern = np.zeros(self.vocab_size)
        
        # Create superposition of similar known tokens
        for known_token, idx in self.vocab.items():
            similarity = self._quantum_similarity(token, known_token)
            phase = 2 * self.pi * similarity * self.phi
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
                substitutions = previous_row[j] + (0 if c1 == c2 else 1/self.e)
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
        self.phi = (1 + np.sqrt(5)) / 2
        self.fib_sequence = self._generate_fibonacci(512)  # Increase size to handle longer sequences
        # Move to the correct device during initialization
        self.fib_sequence = self.fib_sequence.to(device)
    
    def _generate_fibonacci(self, n):
        """Generate Fibonacci sequence"""
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return torch.tensor(fib, dtype=torch.float32)
    
    def apply_structure(self, tokens):
        """Apply natural language patterns based on Fibonacci ratios"""
        B, L, D = tokens.shape
        
        # Ensure we have enough Fibonacci numbers
        if L > len(self.fib_sequence):
            self.fib_sequence = self._generate_fibonacci(L * 2).to(tokens.device)
        
        # Get pattern for this sequence length with scaling
        pattern = (self.fib_sequence[:L] / (self.fib_sequence[L-1] + 1e-8))
        pattern = torch.clamp(pattern, -5, 5)  # Prevent extreme values
        pattern = pattern.to(tokens.device)
        
        # Reshape pattern to match token dimensions
        pattern = pattern.view(1, L, 1).expand(B, -1, D)
        
        # Apply pattern with scaling
        return tokens * pattern * 0.1 

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

class QuantumPreprocessor:
    """Handles initial quantum state preparation and caching"""
    def __init__(self, vocab_size, dim, device):
        self.vocab_size = vocab_size
        self.dim = dim
        self.device = device
        self.phi = (1 + np.sqrt(5)) / 2
        
        # Pre-compute common values
        self.phase_angles = torch.linspace(0, 2 * torch.pi, vocab_size, device=device)
        self.position_encodings = self._init_position_encodings()
    
    def _init_position_encodings(self):
        """Pre-compute position encodings for common sequence lengths"""
        encodings = {}
        common_lengths = [32, 64, 128, 256, 512, 1024]
        for length in common_lengths:
            positions = torch.arange(length, device=self.device).float()
            encodings[length] = torch.sin(positions.unsqueeze(1) * (1.0 / math.sqrt(self.dim)))
        return encodings
    
    def prepare_state(self, token_ids):
        """Prepare initial quantum states for tokens"""
        B, L = token_ids.shape
        states = torch.zeros((B, L, self.dim), device=self.device)
        
        # Vectorized state preparation
        token_phases = self.phase_angles[token_ids]
        states = torch.sin(token_phases.unsqueeze(-1) * torch.arange(self.dim, device=self.device))
        
        return states

class QuantumStateCache:
    """Manages caching of quantum states and patterns"""
    def __init__(self, max_cache_size=1000):
        self.max_cache_size = max_cache_size
        self.state_cache = {}
        self.pattern_cache = {}
        self.access_count = {}
    
    def get_state(self, key, compute_fn):
        """Get cached state or compute if not available"""
        if key not in self.state_cache:
            if len(self.state_cache) >= self.max_cache_size:
                self._evict_least_used()
            self.state_cache[key] = compute_fn()
        self.access_count[key] = self.access_count.get(key, 0) + 1
        return self.state_cache[key]
    
    def _evict_least_used(self):
        """Evict least recently used items from cache"""
        if not self.access_count:
            return
        min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        del self.state_cache[min_key]
        del self.access_count[min_key]

class QuantumPipeline:
    """Manages the quantum processing pipeline"""
    def __init__(self, dim, num_heads, device):
        self.dim = dim
        self.num_heads = num_heads
        self.device = device
        self.head_dim = dim // num_heads
    
    def process_batch(self, states, cache):
        """Process a batch through the quantum pipeline"""
        # Stage 1: Generate interference patterns
        interference = self._generate_interference(states)
        
        # Stage 2: Apply attention mechanism
        attention = self._apply_attention(interference)
        
        # Stage 3: Transform states
        transformed = self._transform_states(attention)
        
        return transformed
    
    def _generate_interference(self, states):
        """Generate interference patterns between quantum states"""
        B, L, D = states.shape
        states_flat = states.view(B * L, D)
        
        # Compute interference using matrix operations
        interference = torch.matmul(states_flat, states_flat.transpose(-2, -1))
        interference = interference.view(B, L, L)
        
        return interference
    
    def _apply_attention(self, interference):
        """Apply quantum-inspired attention mechanism"""
        attention = torch.softmax(interference / math.sqrt(self.head_dim), dim=-1)
        return attention
    
    def _transform_states(self, attention):
        """Apply quantum transformations to states"""
        # Apply non-linear transformation
        transformed = torch.tanh(attention)
        return transformed

class QuantumStageProcessor:
    """Manages multi-stage quantum processing"""
    def __init__(self, vocab_size, dim, num_heads, device):
        self.preprocessor = QuantumPreprocessor(vocab_size, dim, device)
        self.cache = QuantumStateCache()
        self.pipeline = QuantumPipeline(dim, num_heads, device)
        self.device = device
    
    def process_batch(self, batch_data):
        """Process a batch through all quantum stages"""
        # Stage 1: Prepare base states
        base_states = self.preprocessor.prepare_state(batch_data)
        
        # Stage 2: Process through quantum pipeline
        processed_states = self.pipeline.process_batch(base_states, self.cache)
        
        return processed_states

class FastQuantumLLMOptimized(nn.Module):
    """Optimized quantum language model using staged processing"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = device
        
        # Initialize tokenizer and get vocab size
        self.tokenizer = QuantumTokenizer()
        self.config['vocab_size'] = self.tokenizer.vocab_size
        
        # Initialize quantum processor for staged processing
        self.quantum_processor = QuantumStageProcessor(
            vocab_size=self.config['vocab_size'],
            dim=config['dim'],
            num_heads=config['num_heads'],
            device=self.device
        )
        
        # Initialize rest of the model with actual vocab size
        self.scale = 1.0 / math.sqrt(config['dim'])
        
        # Embeddings with proper initialization
        self.token_embedding = nn.Embedding(
            self.config['vocab_size'], 
            config['dim']
        ).to(self.device)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.scale)
        
        # Initialize positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config['max_sequence_length'], config['dim']) * self.scale
        ).to(self.device)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            FastQuantumAttention(config['dim'], config['num_heads'])
            for _ in range(config['num_layers'])
        ]).to(self.device)
        
        # Output projection
        self.output_proj = nn.Linear(config['dim'], self.config['vocab_size']).to(self.device)
        
        # Quantum state encoder
        self.quantum_encoder = FastQuantumState()
        
        # Normalization layers
        self.layer_norm = nn.LayerNorm(config['dim']).to(self.device)
        self.input_norm = nn.LayerNorm(config['dim']).to(self.device)
        self.output_norm = nn.LayerNorm(config['dim']).to(self.device)
    
    def forward(self, input_ids, return_loss=True):
        """Forward pass with staged processing"""
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.long().to(self.device)
        
        B, L = input_ids.shape
        
        # Stage 1: Initial state preparation using quantum processor
        quantum_states = self.quantum_processor.process_batch(input_ids)
        
        # Stage 2: Apply embeddings and position encoding
        x = torch.tanh(self.token_embedding(input_ids) * 0.1)
        x = x + quantum_states  # Add quantum states to embeddings
        
        # Add positional information
        positions = torch.arange(L, device=self.device).float()
        pos_phase = torch.sin(positions.unsqueeze(1) * self.scale)
        x = x + pos_phase.unsqueeze(0) * 0.1
        
        # Apply layer norm
        x = self.input_norm(x)
        
        # Process through attention layers
        for layer in self.attention_layers:
            residual = x
            attn = layer(x) * 0.1
            x = FastQuantumOps.quantum_interference(residual, attn)
            x = torch.tanh(x)
            x = self.layer_norm(x)
        
        # Final processing
        x = self.output_norm(x)
        logits = torch.tanh(self.output_proj(x))
        
        if return_loss and input_ids is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_labels = torch.clamp(shift_labels, 0, self.config['vocab_size'] - 1)
            
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
        # Keep the original generate function unchanged
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = input_ids[0].tolist()
        
        for _ in range(max_length):
            inputs = torch.tensor([generated[-self.config['max_sequence_length']:]]).to(self.device)
            with torch.no_grad():
                outputs = self.forward(inputs, return_loss=False)
                next_token_logits = outputs[0, -1, :]
            
            next_token_logits = next_token_logits / temperature
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_id = top_k_indices[torch.multinomial(probs, num_samples=1)]
            generated.append(next_token_id.item())
            
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
    """Train the model using HuggingFace dataset with memory optimizations"""
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
    
    # Scan dataset to build vocabulary
    print("Scanning dataset to build vocabulary...")
    vocab_words = set()
    for item in tqdm(dataset):
        text = item['text'].lower().split()
        vocab_words.update(text)
    
    # Update vocabulary
    model.tokenizer.extend_vocabulary(vocab_words)
    print(f"Final vocabulary size: {model.tokenizer.vocab_size}")
    
    # Update model config with new vocab size
    model.config['vocab_size'] = model.tokenizer.vocab_size
    
    # Resize token embedding and output projection
    new_embeddings = nn.Embedding(
        model.tokenizer.vocab_size,
        config['dim']
    ).to(model.device)
    new_embeddings.weight.data[:model.token_embedding.num_embeddings] = model.token_embedding.weight.data
    nn.init.normal_(
        new_embeddings.weight.data[model.token_embedding.num_embeddings:],
        mean=0.0,
        std=model.scale
    )
    model.token_embedding = new_embeddings
    
    # Update output projection
    model.output_proj = nn.Linear(
        config['dim'],
        model.tokenizer.vocab_size
    ).to(model.device)
    
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

def main(mode, config, prompt=None, checkpoint=None, max_length=50, temperature=0.1):
    # Initialize optimized model instead of old version
    model = FastQuantumLLMOptimized(config)
    model = model.to(model.device)
    
    if mode == 'train':
        print("Training model with optimized implementation...")
        # Use optimized training function
        train_model_optimized(model, config)
    else:
        # Load checkpoint if provided
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
        
        # Generate text using optimized model
        if not prompt:
            print("Error: --prompt is required when mode is 'generate'")
            return
        
        print(f"\nGenerating text with prompt: {prompt}")
        print(f"Temperature: {temperature}, Max Length: {max_length}")
        
        # Convert prompt to input_ids using tokenizer
        input_ids = model.tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate text
        with torch.no_grad():
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
        
        # Decode generated text
        generated_text = model.tokenizer.decode(generated)
        print(f"\nGenerated text:\n{generated_text}")

def get_default_config():
    """Return default configuration for the model"""
    return {
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
        'output_dir': 'quantum_checkpoints_4',

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
    parser.add_argument('--output_dir', type=str, default='quantum_checkpoints_4',
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
        # main(mode='train', config=config, checkpoint=args.checkpoint)
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

# Add new optimization-focused classes
class QuantumCache:
    """Manages pre-computed and cached quantum operations"""
    def __init__(self, vocab_size, dim, device):
        self.device = device
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Initialize caches
        self.token_pattern_cache = {}  # Cache for token patterns
        self.phase_cache = {}  # Cache for phase calculations
        self.attention_pattern_cache = {}  # Cache for attention patterns
        
        # Pre-compute common values
        self._init_common_values()
    
    def _init_common_values(self):
        """Pre-compute values used across multiple operations"""
        # Phase angles for vocab indices
        self.vocab_phases = torch.linspace(0, 2 * torch.pi, self.vocab_size, device=self.device)
        
        # Common trigonometric values
        self.sin_cache = torch.sin(self.vocab_phases)
        self.cos_cache = torch.cos(self.vocab_phases)
        
        # Position encodings for common sequence lengths
        self.pos_encodings = {}
        common_lengths = [32, 64, 128, 256, 512, 1024]
        for length in common_lengths:
            self.pos_encodings[length] = self._compute_position_encoding(length)
    
    def _compute_position_encoding(self, length):
        """Compute position encoding for a given sequence length"""
        positions = torch.arange(length, device=self.device).float()
        pos_enc = torch.sin(positions.unsqueeze(1) * (1.0 / math.sqrt(self.dim)))
        return pos_enc.to(self.device)
    
    def get_position_encoding(self, length):
        """Get cached position encoding or compute if not available"""
        if length not in self.pos_encodings:
            self.pos_encodings[length] = self._compute_position_encoding(length)
        return self.pos_encodings[length]
    
    def get_token_pattern(self, token_id):
        """Get cached token pattern or compute if not available"""
        if token_id not in self.token_pattern_cache:
            pattern = self._compute_token_pattern(token_id)
            self.token_pattern_cache[token_id] = pattern
        return self.token_pattern_cache[token_id]
    
    def _compute_token_pattern(self, token_id):
        """Compute token pattern efficiently using pre-computed values"""
        # Use vectorized operations instead of loops
        indices = torch.arange(self.vocab_size, device=self.device)
        phase_diff = self.vocab_phases[indices] - self.vocab_phases[token_id]
        pattern = self.cos_cache[indices] * torch.exp(-torch.abs(indices - token_id) / self.vocab_size)
        return pattern

class FastQuantumOpsOptimized:
    """Optimized quantum operations using batched calculations"""
    
    @staticmethod
    def batch_quantum_interference(x, y, cache):
        """Compute quantum interference for entire batch at once"""
        # Reshape inputs for batched computation
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        y_flat = y.view(-1, D)
        
        # Compute interference using pre-computed values
        mixed = torch.tanh((x_flat + y_flat) * 0.5)
        similarity = F.cosine_similarity(x_flat, y_flat, dim=-1, eps=1e-8)
        
        # Reshape back to original dimensions
        return (mixed * similarity.unsqueeze(-1)).view(B, L, D)
    
    @staticmethod
    def batch_phase_encoding(x, cache):
        """Encode phases for entire batch at once"""
        # Use pre-computed trigonometric values
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        phase = torch.atan2(x_norm, torch.roll(x_norm, 1, dims=-1))
        return torch.sin(phase)

class FastQuantumAttentionOptimized(nn.Module):
    """Optimized quantum attention using cached computations"""
    def __init__(self, dim, num_heads, cache):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.cache = cache
        
        # Pre-compute attention patterns with correct dimensions
        self.attention_patterns = self._init_attention_patterns()
        self.register_buffer("patterns", self.attention_patterns)
    
    def _init_attention_patterns(self):
        """Initialize attention patterns using cached values"""
        patterns = []
        for h in range(self.num_heads):
            phase = 2 * torch.pi * h / self.num_heads
            # Create pattern with the correct dimension (head_dim)
            pattern = FastQuantumOpsOptimized.batch_phase_encoding(
                torch.linspace(0, 2 * torch.pi, self.head_dim, device=self.cache.device) + phase,
                self.cache
            )
            patterns.append(pattern)
        return torch.stack(patterns)
    
    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        
        # Reshape input to (B, L, H, head_dim)
        x = x.view(B, L, H, self.head_dim)
        
        # Apply patterns (patterns should be shape [H, head_dim])
        x = x * self.patterns.view(1, 1, H, self.head_dim)
        
        # Compute attention scores efficiently
        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        
        return torch.matmul(attn, x).view(B, L, D)

class FastQuantumLLMOptimized(nn.Module):
    """Optimized quantum language model with shared cache"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = device
        
        # Initialize tokenizer first to get vocab size
        self.tokenizer = QuantumTokenizer()
        
        # Update config with actual vocab size
        self.config['vocab_size'] = self.tokenizer.vocab_size
        
        # Initialize cache with updated vocab size
        self.cache = QuantumCache(
            vocab_size=self.config['vocab_size'],
            dim=config['dim'],
            device=self.device
        )
        
        # Initialize components with cache
        self.token_embedding = nn.Embedding(
            self.config['vocab_size'], 
            config['dim']
        ).to(self.device)
        
        self.attention_layers = nn.ModuleList([
            FastQuantumAttentionOptimized(
                config['dim'],
                config['num_heads'],
                self.cache
            )
            for _ in range(config['num_layers'])
        ]).to(self.device)
        
        # Other layers
        self.layer_norm = nn.LayerNorm(config['dim']).to(self.device)
        self.input_norm = nn.LayerNorm(config['dim']).to(self.device)
        self.output_norm = nn.LayerNorm(config['dim']).to(self.device)
        self.output_proj = nn.Linear(config['dim'], self.config['vocab_size']).to(self.device)
    
    def forward(self, input_ids, return_loss=True):
        B, L = input_ids.shape
        
        # Get embeddings using cached patterns
        x = torch.zeros((B, L, self.config['dim']), device=self.device)
        for b in range(B):
            for l in range(L):
                token_id = input_ids[b, l].item()
                # Get pattern and project it to the model dimension
                pattern = self.cache.get_token_pattern(token_id)
                # Project pattern to model dimension using token embedding
                x[b, l] = self.token_embedding(torch.tensor([token_id], device=self.device)).squeeze(0)
        
        # Add position encoding from cache
        pos_enc = self.cache.get_position_encoding(L)
        x = x + pos_enc.unsqueeze(0)
        
        # Process through attention layers
        x = self.input_norm(x)
        for layer in self.attention_layers:
            residual = x
            attn = layer(x)
            x = FastQuantumOpsOptimized.batch_quantum_interference(residual, attn, self.cache)
            x = self.layer_norm(x)
        
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        if return_loss and input_ids is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1
            )
            return loss
        
        return logits

def train_model_optimized(model, config):
    """Optimized training loop with efficient batch processing"""
    dataset = load_dataset(
        config['dataset_name'],
        config['dataset_config'],
        split='train',
        streaming=True
    )
    
    if config.get('max_samples'):
        dataset = dataset.take(config['max_samples'])
    
    # Create training dataset with optimized tokenizer
    train_dataset = TextDataset(
        dataset,
        model.tokenizer,
        config['max_sequence_length']
    )
    
    # Use efficient data loading
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
    
    # Training loop with optimized batch processing
    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            
            # Process batch
            optimizer.zero_grad()
            loss = model(input_ids)
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Clear cache periodically
            if num_batches % 100 == 0:
                model.cache.token_pattern_cache.clear()
                clear_memory()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Only run CLI handling when script is run directly
    try:
        run_from_cli()
    except SystemExit:
        # Ignore SystemExit in interactive environments
        pass 