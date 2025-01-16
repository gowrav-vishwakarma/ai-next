import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import numpy as np

class ConceptMapper:
    """Maps words to quantum-inspired concept space"""
    def __init__(self, dim: int):
        self.dim = dim
        self.PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.concepts: Dict[str, torch.Tensor] = {}
        self.word_to_concept: Dict[str, str] = {}
        self.concept_counter = 0
        
    def _compute_phase_encoding(self, word: str) -> torch.Tensor:
        """Generate stable phase encoding for a word based on its characters"""
        # Use character positions for phase calculation
        phases = []
        for i, char in enumerate(word):
            # Generate phase based on character position and value
            phase = (ord(char) * self.PHI + i) % (2 * math.pi)
            phases.append(phase)
        
        # Ensure we have dim phases by either truncating or repeating
        while len(phases) < self.dim:
            phases.extend(phases[:self.dim - len(phases)])
        phases = phases[:self.dim]
        
        # Convert to tensor and apply bounded transformation
        phase_tensor = torch.tensor(phases, dtype=torch.float32)
        return torch.sin(phase_tensor)  # Ensure bounded output
    
    def _find_similar_concept(self, encoding: torch.Tensor, threshold: float = 0.85) -> Optional[str]:
        """Find if a similar concept already exists"""
        for concept_id, concept_encoding in self.concepts.items():
            similarity = F.cosine_similarity(encoding.unsqueeze(0), 
                                          concept_encoding.unsqueeze(0)).item()
            if similarity > threshold:
                return concept_id
        return None

    def get_or_create_concept(self, word: str) -> str:
        """Get existing concept or create new one for a word"""
        if word in self.word_to_concept:
            return self.word_to_concept[word]
        
        # Generate phase encoding for the word
        encoding = self._compute_phase_encoding(word)
        
        # Check for similar existing concept
        similar_concept = self._find_similar_concept(encoding)
        if similar_concept:
            self.word_to_concept[word] = similar_concept
            return similar_concept
        
        # Create new concept if none found
        new_concept_id = f"c{self.concept_counter}"
        self.concept_counter += 1
        self.concepts[new_concept_id] = encoding
        self.word_to_concept[word] = new_concept_id
        return new_concept_id

class QuantumTokenizer(nn.Module):
    """Optimized quantum tokenizer"""
    def __init__(self, dim: int = 64, max_vocab_size: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_vocab_size = max_vocab_size
        self.concept_mapper = ConceptMapper(dim)
        
        # Special tokens
        self.PAD_token = "<PAD>"
        self.UNK_token = "<UNK>"
        self.BOS_token = "<BOS>"
        self.EOS_token = "<EOS>"
        
        # Initialize vocabulary with special tokens
        self.special_tokens = [self.PAD_token, self.UNK_token, self.BOS_token, self.EOS_token]
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        self.reverse_vocab = {i: token for token, i in self.vocab.items()}
        
        # Phase space encodings for tokens
        self.token_phases = nn.Parameter(torch.zeros(max_vocab_size, dim))
        
        # Pre-compute special token phases
        self._initialize_special_token_phases()
    
    def _initialize_special_token_phases(self):
        """Initialize special token phases without JIT"""
        phases = torch.zeros(len(self.special_tokens), self.dim)
        for i, _ in enumerate(self.special_tokens):
            phases[i] = torch.sin(torch.linspace(0, 2*math.pi, self.dim) * (i + 1) / len(self.special_tokens))
        self.register_buffer('special_token_phases', phases)
    
    def _split_into_subwords(self, word: str) -> List[str]:
        """Split word into subwords based on common patterns"""
        # Basic subword splitting rules
        if len(word) <= 3:
            return [word]
            
        subwords = []
        current_pos = 0
        
        while current_pos < len(word):
            # Try different subword lengths
            found_subword = False
            for length in range(min(8, len(word) - current_pos), 2, -1):
                subword = word[current_pos:current_pos + length]
                if subword in self.vocab:
                    subwords.append(subword)
                    current_pos += length
                    found_subword = True
                    break
            
            if not found_subword:
                # If no known subword found, take 3 characters
                subwords.append(word[current_pos:current_pos + 3])
                current_pos += 3
        
        return subwords

    def train_on_texts(self, texts: List[str]):
        """Train tokenizer on a corpus of texts"""
        # Collect word frequencies
        word_freqs = Counter()
        for text in texts:
            # Basic preprocessing
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            word_freqs.update(words)
        
        # Create concepts for frequent words
        vocab_size = len(self.special_tokens)
        
        for word, freq in word_freqs.most_common(self.max_vocab_size - vocab_size):
            if vocab_size >= self.max_vocab_size:
                break
                
            # Get or create concept for word
            concept_id = self.concept_mapper.get_or_create_concept(word)
            
            # Add to vocabulary if not already present
            if word not in self.vocab:
                self.vocab[word] = vocab_size
                self.reverse_vocab[vocab_size] = word
                
                # Initialize phase encoding for the token
                with torch.no_grad():
                    self.token_phases[vocab_size] = self.concept_mapper.concepts[concept_id]
                
                vocab_size += 1
    
    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """Optimized encoding"""
        # Use pre-tokenized cache if available
        cache_key = hash(text)
        if hasattr(self, '_token_cache') and cache_key in self._token_cache:
            return self._token_cache[cache_key]
        
        # Fast tokenization using regex
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        
        # Pre-allocate token list
        max_tokens = len(words) + 2 if add_special_tokens else len(words)
        tokens = torch.zeros(max_tokens, dtype=torch.long)
        idx = 0
        
        if add_special_tokens:
            tokens[idx] = self.vocab[self.BOS_token]
            idx += 1
        
        # Vectorized vocabulary lookup
        for word in words:
            tokens[idx] = self.vocab.get(word, self.vocab[self.UNK_token])
            idx += 1
        
        if add_special_tokens:
            tokens[idx] = self.vocab[self.EOS_token]
            idx += 1
        
        # Cache result
        if not hasattr(self, '_token_cache'):
            self._token_cache = {}
        self._token_cache[cache_key] = tokens[:idx]
        
        return tokens[:idx]
    
    def decode(self, tokens: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text"""
        words = []
        for token in tokens.tolist():
            word = self.reverse_vocab.get(token, self.UNK_token)
            if skip_special_tokens and word in self.special_tokens:
                continue
            words.append(word)
        return ' '.join(words)
    
    def get_phase_encoding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get phase encoding for tokens"""
        return F.embedding(token_ids, self.token_phases)
    
    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)

class FastPhaseProcessor(nn.Module):
    """Optimized phase processing with cached lookups"""
    def __init__(self, resolution: int = 1024, max_dim: int = 512):
        super().__init__()
        self.resolution = resolution
        
        # Pre-compute common angles and transformations
        angles = torch.linspace(0, 2*math.pi, resolution)
        self.register_buffer('sin_table', torch.sin(angles))
        self.register_buffer('cos_table', torch.cos(angles))
        
        # Phase cache for common operations
        pos_angles = torch.linspace(0, 2*math.pi, max_dim).unsqueeze(1)
        dim_angles = torch.linspace(0, 2*math.pi, max_dim).unsqueeze(0)
        self.register_buffer('phase_cache', torch.sin(pos_angles + dim_angles))
        
        # Common quantum phase factors
        self.register_buffer('quantum_phases', 
            torch.exp(torch.complex(torch.zeros(resolution), angles)))
        
        # Cache for common rotations
        rotations = torch.linspace(0, math.pi/2, resolution)
        self.register_buffer('rotation_sin', torch.sin(rotations))
        self.register_buffer('rotation_cos', torch.cos(rotations))
    
    def fast_phase_transform(self, angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast phase transformation using lookup"""
        indices = ((angles / (2*math.pi)) * (self.resolution-1)).long()
        return self.sin_table[indices], self.cos_table[indices]
    
    def get_quantum_phase(self, phase_index: torch.Tensor) -> torch.Tensor:
        """Get quantum phase factor from cache"""
        return self.quantum_phases[phase_index % self.resolution]
    
    def fast_rotation(self, angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get rotation components from cache"""
        indices = ((angles / (math.pi/2)) * (self.resolution-1)).long()
        return self.rotation_cos[indices], self.rotation_sin[indices]

class QuantumEmbedding(nn.Module):
    """Quantum-inspired embedding layer"""
    def __init__(self, tokenizer: QuantumTokenizer, embedding_dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.phase_processor = FastPhaseProcessor()
        
        # Create learnable amplitude parameters
        self.amplitude = nn.Parameter(torch.randn(len(tokenizer), embedding_dim) * 0.02)
        
        # Cache common phase patterns
        self.register_buffer('token_phase_patterns', 
            self._precompute_phase_patterns())
    
    def _precompute_phase_patterns(self) -> torch.Tensor:
        """Pre-compute common phase patterns for tokens"""
        vocab_size = len(self.tokenizer)
        patterns = torch.zeros(vocab_size, self.embedding_dim)
        for i in range(vocab_size):
            patterns[i] = torch.linspace(0, 2*math.pi * (i+1)/vocab_size, self.embedding_dim)
        return patterns
    
    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get phase patterns for tokens
        phases = self.token_phase_patterns[token_ids]
        
        # Get amplitudes
        amplitudes = F.embedding(token_ids, self.amplitude)
        
        # Fast phase transformation
        sin_phases, cos_phases = self.phase_processor.fast_phase_transform(phases)
        
        # Create quantum state components
        real_part = amplitudes * cos_phases
        imag_part = amplitudes * sin_phases
        
        # Fast normalization
        norm = torch.rsqrt(real_part.pow(2) + imag_part.pow(2) + 1e-8)
        real_part = real_part * norm
        imag_part = imag_part * norm
        
        return real_part, imag_part

# Example usage:
def create_quantum_tokenizer(texts: List[str], dim: int = 64, max_vocab_size: int = 8192) -> QuantumTokenizer:
    """Create and train a quantum tokenizer on texts"""
    tokenizer = QuantumTokenizer(dim=dim, max_vocab_size=max_vocab_size)
    tokenizer.train_on_texts(texts)
    return tokenizer

# Add FastTrigLookup class for optimized trigonometric operations
class FastTrigLookup(nn.Module):
    """Pre-computed trigonometric lookup tables for fast computation"""
    def __init__(self, resolution: int = 1024):
        super().__init__()
        self.resolution = resolution
        angles = torch.linspace(0, 2*math.pi, resolution)
        self.register_buffer('sin_lookup', torch.sin(angles))
        self.register_buffer('cos_lookup', torch.cos(angles))
        
        # Pre-compute phase factors for common operations
        self.register_buffer('phase_factors', 
            torch.exp(torch.complex(torch.zeros(resolution), angles)))
    
    def sin(self, x: torch.Tensor) -> torch.Tensor:
        """Fast sine approximation"""
        indices = ((x % (2*math.pi)) / (2*math.pi) * self.resolution).long()
        return self.sin_lookup[indices]
    
    def cos(self, x: torch.Tensor) -> torch.Tensor:
        """Fast cosine approximation"""
        indices = ((x % (2*math.pi)) / (2*math.pi) * self.resolution).long()
        return self.cos_lookup[indices]

# Update SimpleQuantumState to use FastPhaseProcessor
class SimpleQuantumState(nn.Module):
    """Optimized quantum state with fast operations"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.phase_processor = FastPhaseProcessor()
        
        # Input is dim, output is 2*dim (real+imag)
        self.fused_transform = nn.Linear(dim, dim * 2)  # Changed back to original dimensions
        torch.nn.init.xavier_uniform_(self.fused_transform.weight)
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim * 2, eps=1e-5)
        
        # Register buffers for faster computation
        self.register_buffer('scale_factor', torch.sqrt(torch.tensor(1.0 / dim)))
        self.register_buffer('gelu_coef1', torch.tensor(0.044715))
        self.register_buffer('gelu_coef2', torch.tensor(math.sqrt(2.0 / math.pi)))
    
    def _fused_gelu(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized GELU implementation without JIT"""
        return x * 0.5 * (1.0 + torch.tanh(
            self.gelu_coef2 * (x + self.gelu_coef1 * torch.pow(x, 3))
        ))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: [B, L, dim]
        batch_size, seq_len = x.shape[:2]
        
        # Fused transformation
        transformed = self.fused_transform(x)  # [B, L, 2*dim]
        activated = self._fused_gelu(transformed)
        normalized = self.norm(activated)  # [B, L, 2*dim]
        
        # Split into real and imaginary parts
        real_part, imag_part = normalized.chunk(2, dim=-1)  # Each [B, L, dim]
        
        # Get phase components
        phase_angles = torch.atan2(imag_part + 1e-8, real_part + 1e-8)
        sin_phases, cos_phases = self.phase_processor.fast_phase_transform(phase_angles)
        
        # Compute state components
        real = real_part * cos_phases
        imag = imag_part * sin_phases
        
        # Fast normalization
        norm = torch.rsqrt(real.pow(2) + imag.pow(2) + 1e-8)
        real = real * norm * self.scale_factor
        imag = imag * norm * self.scale_factor
        
        return real, imag  # Each [B, L, dim]

class DynamicPhaseSpace(nn.Module):
    """Implements dynamic phase space representation"""
    def __init__(self, dim: int, num_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.PHI = (1 + math.sqrt(5)) / 2
        
        # Phase spaces for different semantic layers
        self.layer_phases = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(num_layers)
        ])
        
        # Quantum interference mixers
        self.interference_weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.02) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim, eps=1e-8) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        real_parts = []
        imag_parts = []
        
        for layer_idx in range(self.num_layers):
            # Generate dynamic phase angles
            phase_angles = torch.tanh(self.layer_phases[layer_idx]) * math.pi
            
            # Create quantum state
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
            # Calculate phase factor components separately using PyTorch operations
            angle = 2 * math.pi * i / self.num_layers
            phase_factor_real = torch.cos(torch.tensor(angle))
            phase_factor_imag = torch.sin(torch.tensor(angle))
            
            # Apply phase factor to real and imaginary parts
            final_real += real_parts[i] * phase_factor_real - imag_parts[i] * phase_factor_imag
            final_imag += real_parts[i] * phase_factor_imag + imag_parts[i] * phase_factor_real
        
        # Normalize
        norm = torch.sqrt(final_real.pow(2) + final_imag.pow(2) + 1e-8)
        final_real = final_real / norm
        final_imag = final_imag / norm
        
        return final_real, final_imag

class QuantumCoherenceLayer(nn.Module):
    """Maintains quantum coherence through active phase management"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.PHI = (1 + math.sqrt(5)) / 2
        
        # Phase management parameters
        self.phase_matrix = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.coherence_factor = nn.Parameter(torch.ones(1) * 0.1)
        
        # Quantum state normalization
        self.state_norm = nn.LayerNorm(dim, eps=1e-8)
    
    def compute_coherence_factor(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> torch.Tensor:
        """Compute quantum coherence measure"""
        # Extract phases
        phases = torch.atan2(x_imag + 1e-8, x_real + 1e-8)
        
        # Compute phase differences between all pairs
        phase_diffs = phases.unsqueeze(-2) - phases.unsqueeze(-1)
        
        # Calculate coherence using quantum-inspired cosine measure
        coherence = torch.mean(torch.cos(phase_diffs), dim=(-1, -2))
        return coherence.unsqueeze(-1)
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute current coherence
        coherence = self.compute_coherence_factor(x_real, x_imag)
        
        # Dynamic phase adjustment based on coherence
        phase_adjustment = torch.matmul(x_real, self.phase_matrix) * self.coherence_factor * (1 - coherence)
        
        # Apply phase rotation
        new_real = x_real * torch.cos(phase_adjustment) - x_imag * torch.sin(phase_adjustment)
        new_imag = x_real * torch.sin(phase_adjustment) + x_imag * torch.cos(phase_adjustment)
        
        # Normalize while preserving phase relationships
        norm = torch.sqrt(new_real.pow(2) + new_imag.pow(2) + 1e-8)
        new_real = new_real / norm
        new_imag = new_imag / norm
        
        return self.state_norm(new_real), self.state_norm(new_imag)

class TokenDistributionRegulator(nn.Module):
    """Actively maintains token distribution through quantum interference"""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Target distribution parameters
        self.register_buffer('target_dist', self._initialize_target_distribution())
        
        # Phase-based regulation parameters
        self.regulation_strength = nn.Parameter(torch.ones(1) * 0.1)
        self.phase_factors = nn.Parameter(torch.randn(vocab_size) * 0.02)
    
    def _initialize_target_distribution(self) -> torch.Tensor:
        """Initialize target distribution following Zipf's law"""
        ranks = torch.arange(1, self.vocab_size + 1, dtype=torch.float32)
        dist = 1 / (ranks * torch.log(ranks + 1))
        return dist / dist.sum()
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Get current token distribution
        token_probs = torch.softmax(logits, dim=-1)
        batch_dist = token_probs.mean(dim=[0, 1])
        
        # Compute distribution divergence
        kl_div = F.kl_div(
            batch_dist.log(),
            self.target_dist.to(logits.device),
            reduction='batchmean'
        )
        
        # Apply phase-based regulation
        phase_adjustment = torch.tanh(self.phase_factors) * self.regulation_strength * kl_div
        
        # Adjust logits through phase rotation
        adjusted_logits = logits + phase_adjustment.unsqueeze(0).unsqueeze(0)
        
        return adjusted_logits

# Update BasicQuantumAttention for speed
class BasicQuantumAttention(nn.Module):
    """Speed-optimized quantum attention"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer('scale', torch.tensor(dim).pow(-0.5))
        self.trig = FastTrigLookup()
        
        # Fix QKV projection dimensions
        self.qkv = nn.Linear(dim * 6, dim * 6)
        
        # Output projection
        self.out_proj = nn.Linear(dim * 2, dim * 2)
        
        # Initialize
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Add dropout
        self.attn_dropout = nn.Dropout(0.1)
        
        # Device-specific settings
        self.use_memory_efficient = True
    
    def configure_device(self, device_type: str):
        """Configure for specific device"""
        if device_type == "mps":
            self.use_memory_efficient = False
    
    def forward(self, q_real: torch.Tensor, q_imag: torch.Tensor,
                k_real: torch.Tensor, k_imag: torch.Tensor,
                v_real: torch.Tensor, v_imag: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, L = q_real.shape[:2]
        
        # Combine real and imaginary parts for QKV projection
        # Each component has shape [B, L, dim]
        qkv_input = torch.cat([
            q_real, q_imag,  # [B, L, dim] each
            k_real, k_imag,  # [B, L, dim] each
            v_real, v_imag   # [B, L, dim] each
        ], dim=-1)  # Result: [B, L, 6*dim]
        
        # Project and reshape
        qkv = self.qkv(qkv_input)  # [B, L, 6*dim]
        qkv = qkv.view(B, L, 6, -1)  # [B, L, 6, dim]
        qkv = qkv.transpose(1, 2)  # [B, 6, L, dim]
        
        # Split back into components
        chunks = qkv.chunk(6, dim=1)  # 6 tensors of shape [B, 1, L, dim]
        q_r, q_i, k_r, k_i, v_r, v_i = [x.squeeze(1) for x in chunks]  # Each [B, L, dim]
        
        # Compute attention scores
        scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * self.scale  # [B, L, L]
        
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
        
        # Apply attention
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))  # [B, L, L]
        
        # Apply attention to both real and imaginary parts
        out_real = torch.matmul(attn, v_r)  # [B, L, dim]
        out_imag = torch.matmul(attn, v_i)  # [B, L, dim]
        
        # Combine and project
        out = torch.cat([out_real, out_imag], dim=-1)  # [B, L, 2*dim]
        out = self.out_proj(out)  # [B, L, 2*dim]
        
        # Split back into real and imaginary
        out_real, out_imag = out.chunk(2, dim=-1)  # Each [B, L, dim]
        
        return out_real, out_imag

def compute_enhanced_quantum_loss(
    model: 'QuantumLLM',
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int
) -> torch.Tensor:
    """Optimized quantum loss computation with device flexibility"""
    device = logits.device
    
    # Compute cross entropy efficiently
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=pad_token_id,
        reduction='mean'
    )
    
    # Fast probability computation using native autocast
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
    
    # Efficient pad penalty
    pad_probs = probs[..., pad_token_id]
    pad_penalty = torch.mean(pad_probs.pow(2)) * 10.0
    
    # Fast token diversity using log probabilities
    token_dist = probs.mean(dim=[0, 1])
    diversity_loss = -(token_dist * log_probs.mean(dim=[0, 1])).sum()
    
    # Optimized coherence calculation without complex numbers
    B, L, V = logits.shape
    coherence_loss = torch.tensor(0.0, device=device)
    
    if L > 1:  # Only compute coherence if sequence length > 1
        # Ensure the split is even
        V_half = V // 2
        real = logits[..., :V_half]
        imag = logits[..., V_half:V_half*2]
        
        # Make sure real and imag have the same size
        min_size = min(real.size(-1), imag.size(-1))
        real = real[..., :min_size]
        imag = imag[..., :min_size]
        
        # Compute phases using real and imaginary parts
        phases = torch.atan2(imag + 1e-8, real + 1e-8)
        mean_phase = phases.mean(dim=1, keepdim=True)
        phase_diffs = (phases - mean_phase).abs().mean()
        coherence_loss = -torch.cos(phase_diffs)
    
    # Combine losses with adjusted weights
    total_loss = (
        ce_loss +
        pad_penalty +
        0.1 * coherence_loss +
        0.05 * (1.0 - torch.clamp(diversity_loss, 0.0, 0.5))
    )
    
    return total_loss

class MemoryEfficientQuantumAttention(BasicQuantumAttention):
    """Memory-efficient version of quantum attention"""
    def __init__(self, dim: int):
        super().__init__(dim)
        self.gradient_checkpointing = False
    
    def _memory_efficient_forward(self, q: torch.Tensor, k: torch.Tensor, 
                                v: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Memory efficient attention forward pass"""
        B, H, L, D = q.shape
        output = q.new_zeros(B, H, L, D)
        
        # Process in chunks with gradient checkpointing if enabled
        for i in range(0, L, chunk_size):
            chunk_end = min(i + chunk_size, L)
            
            if self.gradient_checkpointing and self.training:
                chunk_output = torch.utils.checkpoint.checkpoint(
                    self._chunk_forward,
                    q[:, :, i:chunk_end],
                    k,
                    v,
                    preserve_rng_state=False
                )
            else:
                chunk_output = self._chunk_forward(
                    q[:, :, i:chunk_end],
                    k,
                    v
                )
            
            output[:, :, i:chunk_end] = chunk_output
            
            # Clear intermediate memory
            torch.cuda.empty_cache() if q.device.type == 'cuda' else None
            
        return output
    
    def _chunk_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Single chunk forward pass"""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(attn, v)

# Update QuantumLLM to use memory optimizations
class QuantumLLM(nn.Module):
    """Memory-optimized quantum language model"""
    def __init__(self, tokenizer: QuantumTokenizer, dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.dim = dim
        
        # Create components without JIT compilation initially
        self.embedding = QuantumEmbedding(tokenizer, dim)
        self.phase_space = EnhancedDynamicPhaseSpace(dim)
        
        # Multi-layer quantum processing
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': BasicQuantumAttention(dim),
                'phase_norm': nn.LayerNorm(dim, eps=1e-8)
            }) for _ in range(3)
        ])
        
        # Memory optimization settings
        self.gradient_checkpointing_enabled = True
        self.chunk_size = 512  # Will be adjusted per device
        self.use_memory_efficient_attention = True
        
        # Output projection
        self.pre_output_norm = nn.LayerNorm(dim * 2)
        self.output = nn.Linear(dim * 2, len(tokenizer))
        
        # Initialize
        with torch.no_grad():
            nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.output.bias)
        
        # Optional: JIT compile only if not on MPS
        if device != "mps":
            try:
                self.embedding = torch.jit.script(self.embedding)
                self.phase_space = torch.jit.script(self.phase_space)
            except Exception as e:
                print(f"Warning: JIT compilation skipped: {str(e)}")
    
    def configure_memory_settings(self, device_type: str):
        """Configure memory settings based on device"""
        if device_type == 'cuda':
            self.chunk_size = 512
            self.use_memory_efficient_attention = True
            for layer in self.layers:
                layer['attention'].gradient_checkpointing = True
        elif device_type == 'mps':
            self.chunk_size = 128
            self.use_memory_efficient_attention = True
            # MPS-specific optimizations
            for layer in self.layers:
                layer['attention'].gradient_checkpointing = False
        else:  # CPU
            self.chunk_size = 256
            self.use_memory_efficient_attention = False
            for layer in self.layers:
                layer['attention'].gradient_checkpointing = False
    
    def _chunked_forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Process input in chunks to save memory"""
        outputs = []
        for i in range(0, x.size(1), chunk_size):
            chunk = x[:, i:i+chunk_size]
            if self.gradient_checkpointing_enabled and self.training:
                chunk_output = torch.utils.checkpoint.checkpoint(
                    self._forward_impl,
                    chunk,
                    preserve_rng_state=False
                )
            else:
                chunk_output = self._forward_impl(chunk)
            outputs.append(chunk_output)
            
            # Clear cache after each chunk
            if x.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif x.device.type == 'mps':
                try:
                    torch.mps.empty_cache()
                except:
                    pass
        
        return torch.cat(outputs, dim=1)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Memory-efficient forward implementation
        with torch.cuda.amp.autocast() if x.device.type == 'cuda' else nullcontext():
            # Get embeddings
            real_embed, imag_embed = self.embedding(x)
            
            # Process through phase space
            pad_token_id = self.tokenizer.vocab[self.tokenizer.PAD_token]
            real, imag = self.phase_space(real_embed, pad_token_id)
            
            # Create padding mask
            pad_mask = (x == pad_token_id)
            
            # Process through layers with memory optimization
            for layer in self.layers:
                if self.use_memory_efficient_attention:
                    real, imag = layer['attention']._memory_efficient_forward(
                        real.unsqueeze(1), imag.unsqueeze(1),
                        real.unsqueeze(1), imag.unsqueeze(1),
                        real.unsqueeze(1), imag.unsqueeze(1),
                        pad_mask
                    )
                    real, imag = real.squeeze(1), imag.squeeze(1)
                else:
                    real, imag = layer['attention'](
                        real, imag, real, imag, real, imag,
                        pad_mask=pad_mask
                    )
                
                # Apply normalization
                real = layer['phase_norm'](real)
                imag = layer['phase_norm'](imag)
                
                # Clear intermediate tensors
                torch.cuda.empty_cache() if x.device.type == 'cuda' else None
            
            # Final output computation
            combined = self.pre_output_norm(torch.cat([real, imag], dim=-1))
            return self.output(combined)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Configure memory settings for device
        self.configure_memory_settings(x.device.type)
        
        # Use chunked forward if sequence is long
        if x.size(1) > self.chunk_size:
            return self._chunked_forward(x, self.chunk_size)
        
        # Regular forward pass with memory optimization
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x,
                preserve_rng_state=False
            )
        return self._forward_impl(x)

class EnhancedDynamicPhaseSpace(nn.Module):
    """Enhanced dynamic phase space with quantum collapse prevention and fast trig"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.quantum_state = SimpleQuantumState(dim)
        self.trig = FastTrigLookup()
        
        # Anti-collapse mechanism
        self.register_buffer('energy_levels', torch.linspace(0, 1, dim))
        self.excitation_factor = nn.Parameter(torch.ones(1) * 0.1)
        
        # Pre-compute common phase factors
        angles = torch.linspace(0, 2*math.pi, dim)
        self.register_buffer('phase_cache', torch.exp(
            torch.complex(torch.zeros(dim), angles)
        ))
    
    def forward(self, x: torch.Tensor, pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get base quantum state
        real, imag = self.quantum_state(x)
        
        # Add collapse prevention using fast trig
        amplitude = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)
        is_collapsing = (amplitude < 0.1).float()
        
        # Use pre-computed phase factors for excitation
        excitation_real = self.trig.cos(self.energy_levels * 2 * math.pi)
        excitation_imag = self.trig.sin(self.energy_levels * 2 * math.pi)
        
        # Apply excitation where needed
        real = real + is_collapsing * excitation_real.unsqueeze(0).unsqueeze(0) * self.excitation_factor
        imag = imag + is_collapsing * excitation_imag.unsqueeze(0).unsqueeze(0) * self.excitation_factor
        
        # Fast normalization
        norm = torch.rsqrt(real.pow(2) + imag.pow(2) + 1e-8)
        real = real * norm
        imag = imag * norm
        
        # Apply padding mask if needed
        if pad_token_id is not None:
            pad_mask = (x == pad_token_id).unsqueeze(-1)
            real = real.masked_fill(pad_mask, 0.0)
            imag = imag.masked_fill(pad_mask, 0.0)
        
        return real, imag

class QuantumStatePreservingAttention(BasicQuantumAttention):
    """Enhanced quantum attention with fast trig and state preservation"""
    def __init__(self, dim: int):
        super().__init__(dim)
        self.trig = FastTrigLookup()
        
        # Additional quantum state preservation
        self.phase_preservation = nn.Parameter(torch.ones(dim) * 0.1)
        
        # Pre-compute common phase adjustments
        angles = torch.linspace(0, 2*math.pi, dim)
        self.register_buffer('preservation_phases', 
            torch.stack([self.trig.cos(angles), self.trig.sin(angles)], dim=-1))
    
    def forward(self, q_real: torch.Tensor, q_imag: torch.Tensor,
                k_real: torch.Tensor, k_imag: torch.Tensor,
                v_real: torch.Tensor, v_imag: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Get base attention outputs using parent class
        out_real, out_imag = super().forward(q_real, q_imag, k_real, k_imag, v_real, v_imag, pad_mask)
        
        # Fast phase computation using lookup
        phase_indices = ((torch.atan2(out_imag + 1e-8, out_real + 1e-8) + math.pi) 
                        / (2*math.pi) * self.trig.resolution).long()
        
        # Apply phase preservation using pre-computed values
        preservation = torch.sigmoid(self.phase_preservation)
        preserved_real = out_real * self.preservation_phases[phase_indices, 0]
        preserved_imag = out_imag * self.preservation_phases[phase_indices, 1]
        
        # Fast normalization
        norm = torch.rsqrt(preserved_real.pow(2) + preserved_imag.pow(2) + 1e-8)
        preserved_real = preserved_real * norm
        preserved_imag = preserved_imag * norm
        
        return preserved_real, preserved_imag
