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

class QuantumTokenizer:
    """Quantum-inspired tokenizer that uses concepts and subwords"""
    def __init__(self, dim: int = 64, max_vocab_size: int = 8192):
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
        self._initialize_special_token_phases()
    
    def _initialize_special_token_phases(self):
        """Initialize phase encodings for special tokens"""
        with torch.no_grad():
            for i, token in enumerate(self.special_tokens):
                # Generate unique, stable phases for special tokens
                phases = torch.linspace(0, 2*math.pi, self.dim) * (i + 1) / len(self.special_tokens)
                self.token_phases[i] = torch.sin(phases)
    
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
        """Convert text to token IDs with phase information"""
        # Preprocess text
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        
        tokens = []
        
        # Add BOS token if requested
        if add_special_tokens:
            tokens.append(self.vocab[self.BOS_token])
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Split unknown words into subwords
                subwords = self._split_into_subwords(word)
                for subword in subwords:
                    tokens.append(self.vocab.get(subword, self.vocab[self.UNK_token]))
        
        # Add EOS token if requested
        if add_special_tokens:
            tokens.append(self.vocab[self.EOS_token])
        
        return torch.tensor(tokens, dtype=torch.long)
    
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
    """
    Simplified quantum state representation with just two layers:
    1. Ground state (basic meaning)
    2. Single excited state (contextual meaning)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Add layer norms
        self.input_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        
        # Even smaller initialization
        self.ground_transform = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.ground_transform.weight, gain=0.01)
        nn.init.zeros_(self.ground_transform.bias)
        
        self.excite_transform = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.excite_transform.weight, gain=0.01)
        nn.init.zeros_(self.excite_transform.bias)
        
        # Much smaller phase factor initialization
        self.phase_factor = nn.Parameter(torch.randn(dim) * 0.001)
        
        self.PHI = (1 + math.sqrt(5)) / 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        # Input normalization
        x = self.input_norm(x)
        x = x * 0.01  # Aggressive scaling
        
        # Process ground state with residual connection
        ground_state = self.ground_transform(x) + x
        
        # Very bounded phase
        phase = torch.tanh(self.phase_factor) * self.PHI * 0.01
        excited_state = self.excite_transform(x) + x
        
        # Combine states with scaling
        combined_real = ground_state + excited_state * torch.cos(phase)
        combined_imag = excited_state * torch.sin(phase)
        
        # Normalize with larger epsilon
        norm = torch.sqrt(combined_real.pow(2) + combined_imag.pow(2) + 1e-8)
        combined_real = combined_real / norm
        combined_imag = combined_imag / norm
        
        # Output normalization
        combined_real = self.output_norm(combined_real)
        combined_imag = self.output_norm(combined_imag)
        
        return combined_real, combined_imag

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
    """Enhanced quantum attention with coherence preservation"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        
        # Phase-aware projections
        self.q_phase = nn.Linear(dim, dim, bias=False)
        self.k_phase = nn.Linear(dim, dim, bias=False)
        self.v_phase = nn.Linear(dim, dim, bias=False)
        
        # Initialize with small weights
        for layer in [self.q_phase, self.k_phase, self.v_phase]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
    
    def compute_interference(self, q_real: torch.Tensor, q_imag: torch.Tensor, 
                           k_real: torch.Tensor, k_imag: torch.Tensor) -> torch.Tensor:
        """Compute interference pattern between query and key states"""
        B, L, D = q_real.shape  # batch size, sequence length, dimension
        
        # Extract phases
        q_phase = torch.atan2(q_imag + 1e-8, q_real + 1e-8)
        k_phase = torch.atan2(k_imag + 1e-8, k_real + 1e-8)
        
        # Reshape for attention computation
        q_phase = q_phase.view(B, L, 1, D)  # [B, L, 1, D]
        k_phase = k_phase.view(B, 1, L, D)  # [B, 1, L, D]
        
        # Phase difference
        phase_diff = q_phase - k_phase  # [B, L, L, D]
        
        # Interference pattern (sum over dimension)
        interference = torch.cos(phase_diff).sum(dim=-1) * self.scale  # [B, L, L]
        coherence = torch.exp(-torch.abs(phase_diff).sum(dim=-1))  # [B, L, L]
        
        return interference * coherence
    
    def forward(self, q_real: torch.Tensor, q_imag: torch.Tensor,
                k_real: torch.Tensor, k_imag: torch.Tensor,
                v_real: torch.Tensor, v_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with proper shape handling"""
        # Project and normalize
        q_real = self.q_norm(self.q_phase(q_real))
        q_imag = self.q_norm(self.q_phase(q_imag))
        k_real = self.k_norm(self.k_phase(k_real))
        k_imag = self.k_norm(self.k_phase(k_imag))
        v_real = self.v_norm(self.v_phase(v_real))
        v_imag = self.v_norm(self.v_phase(v_imag))
        
        # Compute attention through interference [B, L, L]
        attn_weights = self.compute_interference(q_real, q_imag, k_real, k_imag)
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights / 0.1, dim=-1)  # [B, L, L]
        
        # Apply attention to values
        out_real = torch.bmm(attn_weights, v_real)  # [B, L, D]
        out_imag = torch.bmm(attn_weights, v_imag)  # [B, L, D]
        
        # Final normalization
        out_real = self.output_norm(out_real)
        out_imag = self.output_norm(out_imag)
        
        return out_real, out_imag

def compute_enhanced_quantum_loss(
    model: 'QuantumLLM',
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int
) -> torch.Tensor:
    """
    Enhanced quantum loss function that prevents collapse to PAD token
    with memory-efficient implementation
    """
    # Regular cross entropy with greater penalty for PAD predictions
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=pad_token_id,
        reduction='none'
    )
    
    # Get probabilities
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
    
    # Penalize PAD token predictions heavily
    pad_probs = probs[:, :, pad_token_id]
    pad_penalty = torch.mean(pad_probs.pow(2)) * 10.0
    
    # Encourage token diversity
    with torch.no_grad():
        token_dist = probs.mean(dim=[0, 1])
        diversity_loss = -(token_dist * torch.log(token_dist + 1e-10)).sum()
    
    # Compute coherence loss in small chunks to save memory
    coherence_loss = 0.0
    num_chunks = 0
    chunk_size = 32  # Smaller chunks
    B, L, V = logits.shape
    
    with torch.no_grad():
        for i in range(0, L, chunk_size):
            j = min(i + chunk_size, L)
            chunk = logits[:, i:j, :]
            
            # Process each batch item separately
            for b in range(B):
                # Get chunk for this batch
                batch_chunk = chunk[b:b+1]  # Keep dimension
                
                # Compute phase angles for this small chunk
                chunk_real = batch_chunk.unsqueeze(-2)  # [1, chunk_size, 1, V]
                chunk_imag = batch_chunk.unsqueeze(-1)  # [1, chunk_size, V, 1]
                
                # Compute phase difference
                phase_diff = torch.atan2(chunk_real, chunk_imag + 1e-8)
                
                # Compute coherence for this chunk
                chunk_coherence = -torch.mean(torch.cos(phase_diff))
                coherence_loss += chunk_coherence.item()
                num_chunks += 1
                
                # Clear unnecessary tensors
                del phase_diff, chunk_coherence
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Average coherence loss
    coherence_loss = coherence_loss / max(num_chunks, 1)
    
    # Combine losses with weights
    total_loss = (
        ce_loss.mean() +
        pad_penalty +  # Heavy penalty for PAD collapse
        0.2 * coherence_loss +  # Maintain quantum coherence
        0.1 * (1.0 - torch.clamp(diversity_loss, 0.0, 0.5))  # Encourage diversity
    )
    
    return total_loss
