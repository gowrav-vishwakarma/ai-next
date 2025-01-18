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
    def __init__(self, vocab_size: int, min_freq_threshold: float = 0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.min_freq_threshold = min_freq_threshold
        
        # Register target distribution (Zipf-like for natural language)
        self.register_buffer('target_dist', self._init_zipf_distribution())
        
        # Token frequency tracking
        self.register_buffer('token_counts', torch.zeros(vocab_size))
        self.register_buffer('total_tokens', torch.zeros(1))
        
        # Adaptive scaling for common words
        self.common_word_boost = nn.Parameter(torch.ones(vocab_size))
    
    def _init_zipf_distribution(self):
        """Initialize target distribution following Zipf's law"""
        ranks = torch.arange(1, self.vocab_size + 1, dtype=torch.float32)
        dist = 1 / (ranks * torch.log(ranks + 1))
        return dist / dist.sum()
    
    def update_distribution(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update token distribution and adjust logits"""
        # Update token counts
        new_counts = torch.bincount(
            targets.view(-1), 
            minlength=self.vocab_size
        )
        self.token_counts += new_counts
        self.total_tokens += targets.numel()
        
        # Calculate current distribution
        current_dist = self.token_counts / self.total_tokens.clamp(min=1)
        
        # Identify underrepresented common words
        dist_ratio = current_dist / self.target_dist.clamp(min=1e-8)
        underrep_mask = dist_ratio < self.min_freq_threshold
        
        # Boost common words that are underrepresented
        self.common_word_boost.data = torch.where(
            underrep_mask,
            self.common_word_boost * 1.1,  # Increase boost
            self.common_word_boost * 0.99   # Gradually decrease
        )
        
        return logits + self.common_word_boost.log().unsqueeze(0).unsqueeze(0)

# Update BasicQuantumAttention for speed
class BasicQuantumAttention(nn.Module):
    """Speed-optimized quantum attention with block-sparse patterns"""
    def __init__(self, dim: int, block_size: int = 64, sparsity: float = 0.9):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.sparsity = sparsity
        self.register_buffer('scale', torch.tensor(dim).pow(-0.5))
        self.trig = FastTrigLookup()
        
        # QKV projection with block-sparse structure
        self.qkv = nn.Linear(dim * 6, dim * 6)
        
        # Output projection
        self.out_proj = nn.Linear(dim * 2, dim * 2)
        
        # Initialize
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Add dropout
        self.attn_dropout = nn.Dropout(0.1)
        
        # Memory efficiency settings
        self.max_chunk_size = 128
        self.use_memory_efficient = True
    
    def _get_block_sparse_mask(self, seq_length: int) -> torch.Tensor:
        """Generate block-sparse attention mask for given sequence length"""
        num_blocks = (seq_length + self.block_size - 1) // self.block_size
        block_mask = torch.rand(num_blocks, num_blocks, device=self.scale.device) > self.sparsity
        
        # Ensure causal masking in blocks
        block_mask = torch.tril(block_mask)
        
        # Expand mask to full size
        mask = block_mask.repeat_interleave(self.block_size, dim=0)
        mask = mask.repeat_interleave(self.block_size, dim=1)
        
        # Trim to exact sequence length
        mask = mask[:seq_length, :seq_length]
        
        return mask
    
    def _chunk_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        chunk_size: Optional[int] = None) -> torch.Tensor:
        """Compute attention in memory-efficient chunks"""
        B, L = q.shape[:2]
        chunk_size = chunk_size or self.max_chunk_size
        
        # Generate block-sparse mask for this sequence length
        block_sparse_mask = self._get_block_sparse_mask(L)
        
        output = []
        for i in range(0, L, chunk_size):
            chunk_q = q[:, i:i+chunk_size]
            end_idx = min(i + chunk_size, L)
            
            # Compute scores for this chunk
            scores = torch.matmul(chunk_q, k.transpose(-2, -1)) * self.scale
            
            # Apply block-sparse mask for this chunk
            chunk_mask = block_sparse_mask[i:end_idx, :L]
            scores = scores.masked_fill(~chunk_mask, float('-inf'))
            
            # Apply attention
            attn = self.attn_dropout(torch.softmax(scores, dim=-1))
            chunk_output = torch.matmul(attn, v)
            output.append(chunk_output)
            
            # Clear GPU memory if needed
            if q.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.cat(output, dim=1)
    
    def forward(self, q_real: torch.Tensor, q_imag: torch.Tensor,
                k_real: torch.Tensor, k_imag: torch.Tensor,
                v_real: torch.Tensor, v_imag: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, L = q_real.shape[:2]
        
        # Combine real and imaginary parts for QKV projection
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
        
        # Apply memory-efficient attention separately to real and imaginary parts
        if self.use_memory_efficient:
            out_real = self._chunk_attention(q_r, k_r, v_r)
            out_imag = self._chunk_attention(q_i, k_i, v_i)
        else:
            # Generate block-sparse mask for this sequence length
            block_sparse_mask = self._get_block_sparse_mask(L)
            
            # Compute attention scores
            scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * self.scale
            
            # Apply block-sparse mask and padding mask
            scores = scores.masked_fill(~block_sparse_mask, float('-inf'))
            if pad_mask is not None:
                scores = scores.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
            
            # Apply attention
            attn = self.attn_dropout(torch.softmax(scores, dim=-1))
            out_real = torch.matmul(attn, v_r)
            out_imag = torch.matmul(attn, v_i)
        
        # Combine and project output
        out = torch.cat([out_real, out_imag], dim=-1)  # [B, L, 2*dim]
        out = self.out_proj(out)  # [B, L, 2*dim]
        
        # Split back into real and imaginary
        out_real, out_imag = out.chunk(2, dim=-1)  # Each [B, L, dim]
        
        return out_real, out_imag
    
    def configure_device(self, device_type: str):
        """Configure for specific device"""
        if device_type == "mps":
            self.use_memory_efficient = False
            self.max_chunk_size = 64
        elif device_type == "cuda":
            self.use_memory_efficient = True
            self.max_chunk_size = 128
        else:  # CPU
            self.use_memory_efficient = True
            self.max_chunk_size = 256

def compute_enhanced_quantum_loss(
    model: 'QuantumLLM',
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int
) -> torch.Tensor:
    """Enhanced quantum loss computation with better gradient flow"""
    device = logits.device
    
    # Compute cross entropy with label smoothing
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=pad_token_id,
        reduction='mean',
        label_smoothing=0.1  # Add label smoothing
    )
    
    # Get probabilities
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Enhanced token diversity loss
    token_dist = probs.mean(dim=[0, 1])
    target_dist = torch.ones_like(token_dist) / token_dist.size(0)
    diversity_loss = F.kl_div(
        token_dist.log(), 
        target_dist, 
        reduction='batchmean'
    )
    
    # Compute coherence loss
    B, L, V = logits.shape
    coherence_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    if L > 1:
        V_half = V // 2
        real = logits[..., :V_half]
        imag = logits[..., V_half:V_half*2]
        
        # Enhanced phase coherence
        phases = torch.atan2(imag + 1e-8, real + 1e-8)
        phase_diffs = phases[:, 1:] - phases[:, :-1]
        coherence_loss = 1 - torch.cos(phase_diffs).mean()
    
    # Combine losses with dynamic weighting
    total_loss = (
        ce_loss +
        0.2 * coherence_loss +  # Increased weight
        0.1 * diversity_loss    # Increased weight
    )
    
    # Add L2 regularization
    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters()) * 0.0001
    total_loss = total_loss + l2_reg
    
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
            
            # Scale embeddings for better gradient flow
            real_embed = real_embed * 0.1
            imag_embed = imag_embed * 0.1
            
            # Get PAD token ID
            pad_token_id = self.tokenizer.vocab[self.tokenizer.PAD_token]
            
            # Process through phase space with gradient tracking
            real, imag = self.phase_space(real_embed, pad_token_id)
            
            # Create padding mask
            pad_mask = (x == pad_token_id)
            
            # Add skip connection from embeddings
            real = real + real_embed * 0.1
            imag = imag + imag_embed * 0.1
            
            # Process through layers with gradient accumulation
            layer_outputs = []
            for layer in self.layers:
                real_out, imag_out = layer['attention']._memory_efficient_forward(
                    real.unsqueeze(1), imag.unsqueeze(1),
                    real.unsqueeze(1), imag.unsqueeze(1),
                    real.unsqueeze(1), imag.unsqueeze(1),
                    pad_mask
                )
                real_out, imag_out = real_out.squeeze(1), imag_out.squeeze(1)
                
                # Apply normalization
                real_out = layer['phase_norm'](real_out)
                imag_out = layer['phase_norm'](imag_out)
                
                # Clear intermediate tensors
                torch.cuda.empty_cache() if x.device.type == 'cuda' else None
                
                # Add skip connection from embeddings
                real_out = real_out + real_embed * 0.1
                imag_out = imag_out + imag_embed * 0.1
                
                layer_outputs.append((real_out, imag_out))
                real, imag = real_out, imag_out
            
            # Combine outputs with residual connections
            final_real = sum(out[0] for out in layer_outputs) / len(self.layers)
            final_imag = sum(out[1] for out in layer_outputs) / len(self.layers)
            
            # Combine for output with scaling
            combined = torch.cat([final_real, final_imag], dim=-1)
            combined = self.pre_output_norm(combined)
            
            # Add a small amount of noise during training for exploration
            if self.training:
                noise = torch.randn_like(combined) * 0.01
                combined = combined + noise
            
            # Final output projection
            logits = self.output(combined)
            
            return logits
    
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
        
        # Pre-compute phase patterns for efficiency
        angles = torch.linspace(0, 2*math.pi, dim)
        self.register_buffer('base_phases', angles)
        self.register_buffer('phase_cache', torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=0))
        
        # Add quantum coherence preservation
        self.coherence_layer = QuantumCoherenceLayer(dim)
        
        # Add adaptive phase adjustment
        self.phase_scale = nn.Parameter(torch.ones(dim) * 0.1)
        self.register_buffer('phase_mask', torch.tril(torch.ones(dim, dim)))
    
    def _compute_excitation(self, collapse_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute excitation components efficiently"""
        # Use pre-computed phase patterns
        excitation_real = self.phase_cache[0] * self.excitation_factor
        excitation_imag = self.phase_cache[1] * self.excitation_factor
        
        # Scale by collapse prevention mask
        excitation_real = excitation_real.unsqueeze(0).unsqueeze(0) * collapse_mask
        excitation_imag = excitation_imag.unsqueeze(0).unsqueeze(0) * collapse_mask
        
        return excitation_real, excitation_imag
    
    def _apply_phase_preservation(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply phase preservation using adaptive scaling"""
        # Compute phase relationships
        phase_diff = torch.atan2(imag + 1e-8, real + 1e-8)
        phase_coherence = torch.matmul(phase_diff, self.phase_mask) * self.phase_scale.unsqueeze(0)
        
        # Apply coherent phase adjustment
        cos_adj = torch.cos(phase_coherence)
        sin_adj = torch.sin(phase_coherence)
        
        new_real = real * cos_adj - imag * sin_adj
        new_imag = real * sin_adj + imag * cos_adj
        
        return new_real, new_imag
    
    def forward(self, x: torch.Tensor, pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get base quantum state with gradient tracking
        real, imag = self.quantum_state(x)
        
        # Detect potential collapse states
        amplitude = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)
        collapse_mask = (amplitude < 0.1).float()
        
        # Add excitation with better gradient flow
        excitation = torch.exp(self.energy_levels) * torch.sigmoid(self.excitation_factor)
        real = real + collapse_mask * excitation.unsqueeze(0).unsqueeze(0)
        imag = imag + collapse_mask * excitation.unsqueeze(0).unsqueeze(0)
        
        # Normalize while preserving gradients
        norm = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)
        real = real / norm
        imag = imag / norm
        
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
        
        # Remove retain_grad calls as they're not needed
        # Apply layer normalization
        q_real, q_imag = self.q_norm(q_real), self.q_norm(q_imag)
        k_real, k_imag = self.k_norm(k_real), self.k_norm(k_imag)
        v_real, v_imag = self.v_norm(v_real), self.v_norm(v_imag)
        
        # Compute attention scores
        scores_real = torch.matmul(q_real, k_real.transpose(-2, -1)) - \
                     torch.matmul(q_imag, k_imag.transpose(-2, -1))
        scores_imag = torch.matmul(q_real, k_imag.transpose(-2, -1)) + \
                     torch.matmul(q_imag, k_real.transpose(-2, -1))
        
        # Scale scores
        scores_real = scores_real * self.scale
        scores_imag = scores_imag * self.scale
        
        # Apply padding mask if provided
        if pad_mask is not None:
            scores_real = scores_real.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
            scores_imag = scores_imag.masked_fill(pad_mask.unsqueeze(1), 0.0)
        
        # Compute attention weights with stable softmax
        weights = F.softmax(scores_real, dim=-1)
        
        # Apply attention
        out_real = torch.matmul(weights, v_real)
        out_imag = torch.matmul(weights, v_imag)
        
        # Apply phase preservation
        phase = torch.atan2(out_imag + 1e-8, out_real + 1e-8)
        preservation = torch.sigmoid(self.phase_preservation)
        
        preserved_real = out_real * torch.cos(phase * preservation.unsqueeze(0).unsqueeze(0))
        preserved_imag = out_imag * torch.sin(phase * preservation.unsqueeze(0).unsqueeze(0))
        
        # Add residual connection with scaling
        preserved_real = preserved_real + q_real * 0.1
        preserved_imag = preserved_imag + q_imag * 0.1
        
        return preserved_real, preserved_imag

class ReversibleQuantumLayer(nn.Module):
    """Memory-efficient reversible quantum layer"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # F and G functions for reversible architecture
        self.F = nn.ModuleDict({
            'attention': QuantumStatePreservingAttention(dim),
            'norm': nn.LayerNorm(dim, eps=1e-8)
        })
        
        self.G = nn.ModuleDict({
            'coherence': QuantumCoherenceLayer(dim),
            'norm': nn.LayerNorm(dim, eps=1e-8)
        })
        
        # Memory management
        self.chunk_size = 512
        self.gradient_checkpointing = True
    
    def _forward_F(self, x_real: torch.Tensor, x_imag: torch.Tensor,
                  pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """F function of reversible layer"""
        # Apply attention
        attn_real, attn_imag = self.F['attention'](
            x_real, x_imag, x_real, x_imag, x_real, x_imag,
            pad_mask=pad_mask
        )
        
        # Apply normalization
        return self.F['norm'](attn_real), self.F['norm'](attn_imag)
    
    def _forward_G(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """G function of reversible layer"""
        # Apply coherence preservation
        coh_real, coh_imag = self.G['coherence'](x_real, x_imag)
        
        # Apply normalization
        return self.G['norm'](coh_real), self.G['norm'](coh_imag)
    
    def _chunked_forward(self, x_real: torch.Tensor, x_imag: torch.Tensor,
                        chunk_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input in memory-efficient chunks"""
        chunk_size = chunk_size or self.chunk_size
        B, L, D = x_real.shape
        
        out_real, out_imag = [], []
        
        for i in range(0, L, chunk_size):
            end_idx = min(i + chunk_size, L)
            
            # Process chunk
            chunk_real = x_real[:, i:end_idx]
            chunk_imag = x_imag[:, i:end_idx]
            
            # Apply F and G functions
            F_real, F_imag = self._forward_F(chunk_real, chunk_imag)
            y1_real = chunk_real + F_real
            y1_imag = chunk_imag + F_imag
            
            G_real, G_imag = self._forward_G(y1_real, y1_imag)
            y2_real = y1_real + G_real
            y2_imag = y1_imag + G_imag
            
            out_real.append(y2_real)
            out_imag.append(y2_imag)
            
            # Clear memory
            if x_real.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.cat(out_real, dim=1), torch.cat(out_imag, dim=1)
    
    def _backward_compute(self, y2_real: torch.Tensor, y2_imag: torch.Tensor,
                         grad_real: torch.Tensor, grad_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute backward pass for reversible layer"""
        # Detach tensors for memory efficiency
        y2_real = y2_real.detach()
        y2_imag = y2_imag.detach()
        
        # Compute G gradient
        G_real, G_imag = self._forward_G(y2_real - grad_real, y2_imag - grad_imag)
        
        # Compute F gradient
        y1_real = y2_real - G_real
        y1_imag = y2_imag - G_imag
        F_real, F_imag = self._forward_F(y1_real - grad_real, y1_imag - grad_imag)
        
        # Return input gradients
        return y1_real - F_real, y1_imag - F_imag
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.gradient_checkpointing:
            # Use gradient checkpointing for memory efficiency
            def create_custom_forward(module):
                def custom_forward(*args):
                    return module._chunked_forward(*args)
                return custom_forward
            
            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self),
                x_real, x_imag,
                preserve_rng_state=True
            )
        else:
            return self._chunked_forward(x_real, x_imag)

class MemoryEfficientQuantumLLM(QuantumLLM):
    """Memory-optimized version of QuantumLLM with reversible layers"""
    def __init__(self, tokenizer: QuantumTokenizer, dim: int):
        super().__init__(tokenizer, dim)
        
        # Replace standard layers with reversible layers
        self.layers = nn.ModuleList([
            ReversibleQuantumLayer(dim)
            for _ in range(3)
        ])
        
        # Memory management settings
        self.chunk_size = 512
        self.gradient_checkpointing = True
    
    def configure_memory(self, device_type: str):
        """Configure memory settings based on device"""
        if device_type == "cuda":
            self.chunk_size = 512
            self.gradient_checkpointing = True
        elif device_type == "mps":
            self.chunk_size = 128
            self.gradient_checkpointing = False
        else:  # CPU
            self.chunk_size = 256
            self.gradient_checkpointing = True
        
        # Configure layers
        for layer in self.layers:
            layer.chunk_size = self.chunk_size
            layer.gradient_checkpointing = self.gradient_checkpointing
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get quantum embeddings
        real_embed, imag_embed = self.embedding(x)
        
        # Scale embeddings for better gradient flow
        real_embed = real_embed * 0.1
        imag_embed = imag_embed * 0.1
        
        # Get PAD token ID
        pad_token_id = self.tokenizer.vocab[self.tokenizer.PAD_token]
        
        # Process through phase space with gradient tracking
        real, imag = self.phase_space(real_embed, pad_token_id)
        
        # Create padding mask
        pad_mask = (x == pad_token_id)
        
        # Add skip connection from embeddings
        real = real + real_embed * 0.1
        imag = imag + imag_embed * 0.1
        
        # Process through layers with gradient accumulation
        layer_outputs = []
        for layer in self.layers:
            real_out, imag_out = layer(real, imag, pad_mask=pad_mask)
            layer_outputs.append((real_out, imag_out))
            real, imag = real_out, imag_out
        
        # Combine outputs with residual connections
        final_real = sum(out[0] for out in layer_outputs) / len(self.layers)
        final_imag = sum(out[1] for out in layer_outputs) / len(self.layers)
        
        # Combine for output with scaling
        combined = torch.cat([final_real, final_imag], dim=-1)
        combined = self.pre_output_norm(combined)
        
        # Add a small amount of noise during training for exploration
        if self.training:
            noise = torch.randn_like(combined) * 0.01
            combined = combined + noise
        
        # Final output projection
        logits = self.output(combined)
        
        return logits
