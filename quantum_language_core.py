import numpy as np
import torch
import math
from collections import defaultdict
import re

class QuantumTokenizer:
    """Quantum-inspired tokenizer using wave patterns and universal constants"""
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.phi = (1 + np.sqrt(5)) / 2
        self.e = np.e
        self.pi = np.pi
        
        # Initialize quantum patterns
        self.token_patterns = self._initialize_quantum_patterns()
        
        # Create basic vocabulary with common words and characters
        self.vocab = self._initialize_vocabulary()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Initialize wave patterns for characters
        self.char_waves = self._initialize_char_waves()
        
        # Add token to index mapping for generation
        self.eos_token_id = 3  # <EOS> token
        
        # Add padding and truncation defaults
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
        
        # Add common words and characters
        common_words = """the be to of and a in that have I it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us""".split()
        
        # Add characters with wave patterns
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_")
        
        # Add words and characters to vocabulary
        for token in common_words + chars:
            if len(vocab) < self.vocab_size:
                vocab[token] = len(vocab)
        
        return vocab
    
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
        """Enhanced encode method"""
        # Normalize text
        text = text.lower().strip()
        tokens = self._quantum_tokenize(text)
        
        # Get token indices
        indices = []
        for token in tokens:
            idx = self.vocab.get(token, 1)  # 1 is UNK token
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
            text.append(token)
        
        return ' '.join(text)
    
    def _quantum_tokenize(self, text):
        """Tokenize text using quantum-inspired patterns"""
        # Use regex with natural language patterns
        pattern = r'\w+|\s+|[^\w\s]'
        tokens = re.findall(pattern, text)
        
        # Apply Fibonacci-based grouping
        grouped_tokens = self._fibonacci_group(tokens)
        return grouped_tokens
    
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
        # Move to MPS device during initialization
        self.fib_sequence = self.fib_sequence.to("mps")
    
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