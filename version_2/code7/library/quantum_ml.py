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

class QuantumEmbedding(nn.Module):
    """Quantum-inspired embedding layer"""
    def __init__(self, tokenizer: QuantumTokenizer, embedding_dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        
        # Create learnable amplitude parameters
        self.amplitude = nn.Parameter(torch.randn(len(tokenizer), embedding_dim) * 0.02)
        
    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert tokens to quantum state representations
        Returns: (real_part, imaginary_part)
        """
        # Get phase encodings from tokenizer
        phases = self.tokenizer.get_phase_encoding(token_ids)
        
        # Get amplitudes for tokens
        amplitudes = F.embedding(token_ids, self.amplitude)
        
        # Create quantum state components
        real_part = amplitudes * torch.cos(phases)
        imag_part = amplitudes * torch.sin(phases)
        
        # Normalize
        norm = torch.sqrt(real_part.pow(2) + imag_part.pow(2) + 1e-8)
        real_part = real_part / norm
        imag_part = imag_part / norm
        
        return real_part, imag_part

# Example usage:
def create_quantum_tokenizer(texts: List[str], dim: int = 64, max_vocab_size: int = 8192) -> QuantumTokenizer:
    """Create and train a quantum tokenizer on texts"""
    tokenizer = QuantumTokenizer(dim=dim, max_vocab_size=max_vocab_size)
    tokenizer.train_on_texts(texts)
    return tokenizer


class SimpleQuantumState(nn.Module):
    """Optimized quantum state representation"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Combine transforms into single operation
        self.unified_transform = nn.Linear(dim, dim * 2)
        nn.init.xavier_normal_(self.unified_transform.weight, gain=0.01)
        nn.init.zeros_(self.unified_transform.bias)
        
        # Single normalization layer
        self.norm = nn.LayerNorm(dim * 2)
        
        # Constant phase factor
        self.register_buffer('phase_factor', torch.tensor([(1 + math.sqrt(5)) / 2 * 0.01]))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Single transform operation
        combined = self.unified_transform(x)
        combined = self.norm(combined)
        
        # Split into real and imaginary efficiently
        real, imag = torch.chunk(combined, 2, dim=-1)
        
        # Fast normalization using fused operation
        norm = torch.rsqrt(real.pow(2) + imag.pow(2) + 1e-8)
        real = real * norm
        imag = imag * norm
        
        return real, imag

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

# Update BasicQuantumAttention to use coherence
class BasicQuantumAttention(nn.Module):
    """Optimized quantum attention"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        
        # Separate projections for better dimension control
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Initialize with small weights
        for layer in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        
        # Single normalization layer
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, q_real: torch.Tensor, q_imag: torch.Tensor,
                k_real: torch.Tensor, k_imag: torch.Tensor,
                v_real: torch.Tensor, v_imag: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Get batch size and sequence length
        B = q_real.size(0)
        L = q_real.size(1)
        
        # Project each component separately
        q_real = self.q_proj(q_real)
        q_imag = self.q_proj(q_imag)
        k_real = self.k_proj(k_real)
        k_imag = self.k_proj(k_imag)
        v_real = self.v_proj(v_real)
        v_imag = self.v_proj(v_imag)
        
        # Compute attention scores
        attn_real = torch.matmul(q_real, k_real.transpose(-2, -1)) * self.scale
        attn_imag = torch.matmul(q_imag, k_imag.transpose(-2, -1)) * self.scale
        
        # Apply padding mask if provided
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1)  # [B, 1, L]
            attn_real = attn_real.masked_fill(pad_mask, float('-inf'))
            attn_imag = attn_imag.masked_fill(pad_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(attn_real + attn_imag, dim=-1)
        
        # Compute output
        out_real = torch.matmul(attn_weights, v_real)
        out_imag = torch.matmul(attn_weights, v_imag)
        
        # Apply normalization
        out_real = self.norm(out_real)
        out_imag = self.norm(out_imag)
        
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
