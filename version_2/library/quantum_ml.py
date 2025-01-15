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
    
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to token IDs with phase information"""
        # Preprocess text
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        
        # Start with BOS token
        tokens = [self.vocab[self.BOS_token]]
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Split unknown words into subwords
                subwords = self._split_into_subwords(word)
                for subword in subwords:
                    tokens.append(self.vocab.get(subword, self.vocab[self.UNK_token]))
        
        # Add EOS token
        tokens.append(self.vocab[self.EOS_token])
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Convert token IDs back to text"""
        words = []
        for token in tokens.tolist():
            word = self.reverse_vocab.get(token, self.UNK_token)
            if word not in self.special_tokens:
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

class BasicQuantumAttention(nn.Module):
    """
    Memory-efficient quantum attention using phase relationships
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        
        # Add more layer norms
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        
    def forward(self, q_real, q_imag, k_real, k_imag, v_real, v_imag):
        # Apply layer norm and scaling
        q_real = self.q_norm(q_real) * 0.01
        q_imag = self.q_norm(q_imag) * 0.01
        k_real = self.k_norm(k_real) * 0.01
        k_imag = self.k_norm(k_imag) * 0.01
        v_real = self.v_norm(v_real) * 0.01
        v_imag = self.v_norm(v_imag) * 0.01
        
        # Get dimensions
        B, L, D = q_real.shape
        
        # Process in smaller chunks
        chunk_size = min(L, 32)  # Smaller chunks
        output_real = torch.zeros(B, L, D, device=q_real.device, dtype=q_real.dtype)
        output_imag = torch.zeros(B, L, D, device=q_imag.device, dtype=q_imag.dtype)
        
        for i in range(0, L, chunk_size):
            j = min(i + chunk_size, L)
            
            # Get current chunk
            q_real_chunk = q_real[:, i:j].contiguous()
            q_imag_chunk = q_imag[:, i:j].contiguous()
            
            # Compute attention scores with residual connection
            real_diff = (q_real_chunk.unsqueeze(2) - k_real.unsqueeze(1)) * self.scale
            imag_diff = (q_imag_chunk.unsqueeze(2) - k_imag.unsqueeze(1)) * self.scale
            
            # Sum with stability term
            real_sum = torch.sum(real_diff, dim=-1) + 1e-8
            imag_sum = torch.sum(imag_diff, dim=-1) + 1e-8
            
            # Compute stable phase difference
            phase_diff = torch.atan2(imag_sum, real_sum)
            
            # Compute stable attention weights
            attn_weights = torch.softmax(torch.cos(phase_diff) * 10.0, dim=-1)
            
            # Apply attention with scaled values
            chunk_real = torch.bmm(attn_weights, v_real) * 0.1
            chunk_imag = torch.bmm(attn_weights, v_imag) * 0.1
            
            output_real[:, i:j] = chunk_real
            output_imag[:, i:j] = chunk_imag
            
            # Clear memory
            del real_diff, imag_diff, phase_diff, attn_weights, chunk_real, chunk_imag
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Final normalization
        output_real = self.output_norm(output_real)
        output_imag = self.output_norm(output_imag)
        
        return output_real, output_imag
