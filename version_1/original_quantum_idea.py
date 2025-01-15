import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
from collections import defaultdict
import re
import math
import os

class EnhancedQuantumTokenizer:
    """Enhanced tokenizer with multi-level tokenization and quantum encoding"""
    def __init__(self, vocab_size, max_sequence_length=2048):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        
        # Initialize vocabularies
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3, '<MASK>': 4}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Initialize multi-level vocabulary
        self._initialize_char_vocab()
        self._initialize_subword_vocab()
        self._initialize_word_vocab()
        
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
        
        for token_id in range(self.vocab_size):
            # Create quantum state vector
            phase = 2 * np.pi * token_id / self.vocab_size
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
        """Encode text using multi-level tokenization"""
        if not text:
            return []
        
        tokens = []
        words = text.split()
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Try subword tokenization
                subword_tokens = self._subword_tokenize(word)
                tokens.extend(subword_tokens)
        
        if max_length:
            tokens = tokens[:max_length-2]
            return [self.vocab['<BOS>']] + tokens + [self.vocab['<EOS>']]
        
        return tokens
    
    def _subword_tokenize(self, word):
        """Tokenize word into subwords"""
        tokens = []
        while word:
            found = False
            # Try to find longest matching subword
            for i in range(len(word), 0, -1):
                subword = word[:i]
                if subword in self.vocab:
                    tokens.append(self.vocab[subword])
                    word = word[i:]
                    found = True
                    break
            
            if not found:
                # Character-level fallback
                tokens.append(self.vocab.get(word[0], self.vocab['<UNK>']))
                word = word[1:]
        
        return tokens
    
    def decode(self, tokens):
        """Decode token IDs back to text"""
        return ' '.join(self.reverse_vocab.get(token, '<UNK>') for token in tokens)

class EnhancedQuantumAttention:
    """Enhanced attention mechanism with multi-scale processing"""
    def __init__(self, dim, num_heads, max_sequence_length=2048):
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
        head_dim = self.head_dim  # Use head_dim from initialization
        patterns = np.zeros((self.num_heads, self.max_sequence_length, head_dim), 
                           dtype=np.complex128)
        
        for h in range(self.num_heads):
            phase = 2 * np.pi * h / self.num_heads
            t = np.linspace(0, 2*np.pi*frequency, self.max_sequence_length)
            
            for d in range(head_dim):
                patterns[h, :, d] = (np.exp(1j * (t + phase)) + 
                                   np.exp(1j * (2*t + phase)) +
                                   np.exp(1j * (0.5*t + phase)))
        
        # Normalize across sequence length dimension
        patterns = patterns / np.sqrt(np.sum(np.abs(patterns)**2, axis=1, keepdims=True))
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
        pattern = np.zeros((self.max_sequence_length, dim), dtype=np.complex128)
        
        # Create quantum interference pattern
        t = np.linspace(0, 2*np.pi, self.max_sequence_length)
        for d in range(dim):
            phase = 2 * np.pi * d / dim
            pattern[:, d] = np.exp(1j * (frequency * t + phase))
        
        # Normalize the pattern
        pattern = pattern / np.sqrt(np.sum(np.abs(pattern)**2, axis=1, keepdims=True))
        return pattern
    
    def _apply_expert(self, x, expert):
        """Apply expert pattern using quantum interference"""
        B, H, L, D = x.shape  # Batch, Heads, Length, Dim
        
        # Reshape input for quantum interference
        x_flat = x.reshape(B * H, L, D)
        
        # Apply quantum interference patterns
        syntax = np.einsum('bld,ld->bld', x_flat, expert['syntax'][:L])
        semantics = np.einsum('bld,ld->bld', x_flat, expert['semantics'][:L])
        context = np.einsum('bld,ld->bld', x_flat, expert['context'][:L])
        
        # Combine using quantum superposition
        combined = (syntax + semantics + context) / np.sqrt(3)
        
        # Reshape back to original dimensions
        return combined.reshape(B, H, L, D)
    
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
        combined = sum(scale_outputs) / np.sqrt(len(scale_outputs))
        
        # Apply expert patterns with quantum interference
        expert_outputs = []
        for expert in self.expert_patterns:
            expert_output = self._apply_expert(combined, expert)
            expert_outputs.append(expert_output)
        
        # Final quantum combination
        output = sum(expert_outputs) / np.sqrt(len(expert_outputs))
        return output
    
    def _process_scale(self, Q, K, V, pattern, mask):
        """Process attention at a specific scale"""
        B, H, L, D = Q.shape  # Batch, Heads, Length, Head_dim
        
        # Reshape pattern to match Q and K shapes
        pattern = pattern[:H, :L, :]  # [heads, seq_len, head_dim]
        pattern = np.expand_dims(pattern, 0)  # [1, heads, seq_len, head_dim]
        pattern = np.broadcast_to(pattern, (B, H, L, D))  # [batch, heads, seq_len, head_dim]
        
        # Apply interference pattern
        Q = Q * pattern
        K = K * pattern
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(D)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Attention weights
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        
        # Apply attention to values
        output = np.matmul(weights, V)
        return output

class EnhancedUniversalPatterns:
    """Enhanced pattern recognition using universal constants"""
    def __init__(self, dim):
        self.dim = dim
        self.patterns = self._initialize_enhanced_patterns()
    
    def _initialize_enhanced_patterns(self):
        """Initialize patterns using multiple universal constants"""
        patterns = {}
        
        # Golden ratio patterns
        phi = (1 + np.sqrt(5)) / 2
        patterns['golden'] = self._generate_golden_patterns(phi)
        
        # Pi-based patterns
        patterns['pi'] = self._generate_pi_patterns()
        
        # E-based patterns
        patterns['e'] = self._generate_e_patterns()
        
        return patterns
    
    def _generate_golden_patterns(self, phi):
        """Generate patterns based on golden ratio"""
        patterns = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                patterns[i, j] = phi ** ((i + j) / self.dim)
        return patterns / np.max(patterns)
    
    def _generate_pi_patterns(self):
        """Generate patterns based on pi"""
        patterns = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                patterns[i, j] = np.sin(np.pi * (i + j) / self.dim)
        return patterns
    
    def _generate_e_patterns(self):
        """Generate patterns based on e"""
        patterns = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                patterns[i, j] = np.exp(-(i + j) / self.dim)
        return patterns
    
    def forward(self, x):
        """Apply universal patterns to input"""
        output = x.copy()
        
        for pattern in self.patterns.values():
            output = output + np.matmul(output, pattern)
        
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
        self.step_count += 1
        
        # Get model parameters directly
        params = {
            'token_embeddings': model.token_embeddings,
            'position_embeddings': model.position_embeddings
        }
        
        # Add attention layer parameters
        for i, layer in enumerate(model.attention_layers):
            params[f'attention_layer_{i}'] = layer
        
        # Add universal pattern parameters
        params['universal_patterns'] = model.universal_patterns
        
        # Compute gradients
        grads = self._compute_quantum_gradients(model, loss, params)
        
        # Update parameters
        for name, param in params.items():
            if isinstance(param, np.ndarray):
                grad = grads[name]
                
                # Update momentum terms (maintain complex type)
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * np.abs(grad)**2
                
                # Bias correction
                m_hat = self.m[name] / (1 - self.beta1**self.step_count)
                v_hat = self.v[name] / (1 - self.beta2**self.step_count)
                
                # Quantum-inspired update (keep complex type)
                phase = np.angle(m_hat + 1j * v_hat)
                update = self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8) * np.exp(1j * phase)
                
                # Update parameter (maintain complex type)
                param -= update
            
            elif isinstance(param, EnhancedQuantumAttention):
                # Update attention layer parameters
                for expert_pattern in param.expert_patterns:
                    for aspect, pattern in expert_pattern.items():
                        name = f'{name}_expert_{aspect}'
                        grad = grads[name]
                        
                        # Update momentum terms (maintain complex type)
                        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * np.abs(grad)**2
                        
                        # Bias correction
                        m_hat = self.m[name] / (1 - self.beta1**self.step_count)
                        v_hat = self.v[name] / (1 - self.beta2**self.step_count)
                        
                        # Quantum-inspired update (keep complex type)
                        phase = np.angle(m_hat + 1j * v_hat)
                        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8) * np.exp(1j * phase)
                        
                        # Update parameter (maintain complex type)
                        pattern -= update
            
            elif isinstance(param, EnhancedUniversalPatterns):
                # Update universal pattern parameters
                for pattern_name, pattern in param.patterns.items():
                    name = f'{name}_{pattern_name}'
                    grad = grads[name]
                    
                    # Update momentum terms (maintain complex type)
                    self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                    self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * np.abs(grad)**2
                    
                    # Bias correction
                    m_hat = self.m[name] / (1 - self.beta1**self.step_count)
                    v_hat = self.v[name] / (1 - self.beta2**self.step_count)
                    
                    # Quantum-inspired update (keep complex type)
                    phase = np.angle(m_hat + 1j * v_hat)
                    update = self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8) * np.exp(1j * phase)
                    
                    # Update parameter (maintain complex type)
                    pattern -= update
    
    def _compute_quantum_gradients(self, model, loss, params):
        """Compute gradients using quantum-inspired methods"""
        grads = {}
        eps = 1e-8
        
        for name, param in params.items():
            if isinstance(param, np.ndarray):
                grad = np.zeros_like(param, dtype=np.complex128)  # Ensure complex type for grad
                phase = 2 * np.pi * np.random.random(param.shape)
                
                # Compute gradient using quantum-inspired finite differences
                param_plus = param + eps * np.exp(1j * phase)
                param_minus = param - eps * np.exp(1j * phase)
                
                if name == 'token_embeddings':
                    model.token_embeddings = param_plus
                    loss_plus = model.forward(model.last_input).mean()
                    
                    model.token_embeddings = param_minus
                    loss_minus = model.forward(model.last_input).mean()
                
                elif name == 'position_embeddings':
                    model.position_embeddings = param_plus
                    loss_plus = model.forward(model.last_input).mean()
                    
                    model.position_embeddings = param_minus
                    loss_minus = model.forward(model.last_input).mean()
                
                else:
                    loss_plus = loss_minus = loss
                
                grad = (loss_plus - loss_minus) / (2 * eps) * np.exp(-1j * phase)  # Keep complex type
                
                grads[name] = grad
            
            elif isinstance(param, EnhancedQuantumAttention):
                # Compute gradients for attention layer parameters
                for expert_pattern in param.expert_patterns:
                    for aspect, pattern in expert_pattern.items():
                        name = f'{name}_expert_{aspect}'
                        grad = np.zeros_like(pattern, dtype=np.complex128)  # Ensure complex type for grad
                        phase = 2 * np.pi * np.random.random(pattern.shape)
                        
                        # Compute gradient using quantum-inspired finite differences
                        pattern_plus = pattern + eps * np.exp(1j * phase)
                        pattern_minus = pattern - eps * np.exp(1j * phase)
                        
                        expert_pattern[aspect] = pattern_plus
                        loss_plus = model.forward(model.last_input).mean()
                        
                        expert_pattern[aspect] = pattern_minus
                        loss_minus = model.forward(model.last_input).mean()
                        
                        grad = (loss_plus - loss_minus) / (2 * eps) * np.exp(-1j * phase)  # Keep complex type
                        
                        grads[name] = grad
            
            elif isinstance(param, EnhancedUniversalPatterns):
                # Compute gradients for universal pattern parameters
                for pattern_name, pattern in param.patterns.items():
                    name = f'{name}_{pattern_name}'
                    grad = np.zeros_like(pattern, dtype=np.complex128)  # Ensure complex type for grad
                    phase = 2 * np.pi * np.random.random(pattern.shape)
                    
                    # Compute gradient using quantum-inspired finite differences
                    pattern_plus = pattern + eps * np.exp(1j * phase)
                    pattern_minus = pattern - eps * np.exp(1j * phase)
                    
                    param.patterns[pattern_name] = pattern_plus
                    loss_plus = model.forward(model.last_input).mean()
                    
                    param.patterns[pattern_name] = pattern_minus
                    loss_minus = model.forward(model.last_input).mean()
                    
                    grad = (loss_plus - loss_minus) / (2 * eps) * np.exp(-1j * phase)  # Keep complex type
                    
                    grads[name] = grad
        
        return grads

class EnhancedQuantumLLM:
    """Complete enhanced quantum-inspired language model"""
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        self.tokenizer = EnhancedQuantumTokenizer(
            vocab_size=config['vocab_size'],
            max_sequence_length=config['max_sequence_length']
        )
        
        self.attention_layers = [
            EnhancedQuantumAttention(
                dim=config['dim'],
                num_heads=config['num_heads'],
                max_sequence_length=config['max_sequence_length']
            )
            for _ in range(config['num_layers'])
        ]
        
        self.universal_patterns = EnhancedUniversalPatterns(
            dim=config['dim']
        )
        
        # Initialize embeddings
        self._initialize_enhanced_embeddings()
        
        # For training
        self.last_input = None
    
    def _initialize_enhanced_embeddings(self):
        """Initialize embeddings with quantum-inspired patterns"""
        self.token_embeddings = np.zeros(
            (self.config['vocab_size'], self.config['dim']), 
            dtype=np.complex128
        )
        
        self.position_embeddings = np.zeros(
            (self.config['max_sequence_length'], self.config['dim']), 
            dtype=np.complex128
        )
        
        # Initialize with quantum states
        for i in range(self.config['vocab_size']):
            self.token_embeddings[i] = self._generate_quantum_state(i)
        
        for pos in range(self.config['max_sequence_length']):
            self.position_embeddings[pos] = self._generate_position_state(pos)
    
    def _generate_quantum_state(self, index):
        """Generate quantum state for token embedding"""
        state = np.zeros(self.config['dim'], dtype=np.complex128)
        phase = 2 * np.pi * index / self.config['vocab_size']
        
        for i in range(self.config['dim']):
            theta = np.pi * i / self.config['dim']
            state[i] = np.exp(1j * (phase + theta))
        
        return state / np.sqrt(np.sum(np.abs(state)**2))
    
    def _generate_position_state(self, position):
        """Generate quantum state for position embedding"""
        state = np.zeros(self.config['dim'], dtype=np.complex128)
        
        for i in range(self.config['dim']):
            if i % 2 == 0:
                state[i] = np.sin(position / (10000 ** (i / self.config['dim'])))
            else:
                state[i] = np.cos(position / (10000 ** ((i-1) / self.config['dim'])))
        
        return state / np.sqrt(np.sum(np.abs(state)**2))
    
    def forward(self, input_ids, training=True):
        """Forward pass through the model"""
        batch_size, seq_length = input_ids.shape
        self.last_input = input_ids  # Store for training
        
        # Get embeddings
        embeddings = self.token_embeddings[input_ids]
        positions = self.position_embeddings[:seq_length]
        x = embeddings + positions[None, :, :]
        
        # Process through layers
        attention_mask = self._create_attention_mask(seq_length)
        
        for i in range(len(self.attention_layers)):
            # Reshape for attention - split heads properly
            head_dim = self.config['dim'] // self.config['num_heads']
            q = k = v = x.reshape(batch_size, seq_length, self.config['num_heads'], head_dim)
            q = q.transpose(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            
            # Apply attention
            attention_output = self.attention_layers[i].forward(q, k, v, attention_mask)
            
            # Reshape back
            attention_output = attention_output.transpose(0, 2, 1, 3)
            attention_output = attention_output.reshape(batch_size, seq_length, -1)
            x = x + attention_output
            
            # Apply universal patterns
            x = self.universal_patterns.forward(x)
            
            if training:
                # Apply quantum dropout
                x = self._quantum_dropout(x, rate=0.1)
        
        # Project to vocabulary
        logits = np.matmul(x, self.token_embeddings.T)
        return logits
    
    def _create_attention_mask(self, seq_length):
        """Create causal attention mask"""
        mask = np.tril(np.ones((seq_length, seq_length)))
        return mask[None, None, :, :]
    
    def _quantum_dropout(self, x, rate=0.1):
        """Apply quantum-inspired dropout"""
        if rate == 0:
            return x
        
        # Create quantum-inspired mask
        phase = np.angle(x)
        mask = (np.random.random(x.shape) > rate) / (1 - rate)
        return x * mask * np.exp(1j * phase)
    
    def generate(self, prompt, max_length=100, temperature=0.8):
        """Generate text using the model"""
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = np.array(input_ids).reshape(1, -1)
        
        generated = list(input_ids[0])
        
        for _ in range(max_length):
            # Get predictions
            if len(generated) > self.config['max_sequence_length']:
                context = generated[-self.config['max_sequence_length']:]
            else:
                context = generated
            
            input_ids = np.array(context).reshape(1, -1)
            logits = self.forward(input_ids, training=False)
            next_token_logits = logits[0, -1]
            
            # Apply temperature and quantum sampling
            scaled_logits = next_token_logits / temperature
            probs = quantum_softmax(scaled_logits)
            
            # Ensure probabilities are real and normalized
            probs = np.abs(probs)  # Take the magnitude to get real probabilities
            probs /= np.sum(probs)  # Normalize to ensure they sum to 1
            
            # Quantum-inspired sampling
            phase = np.angle(probs + 1j * np.roll(probs, 1))
            interference = np.abs(np.exp(1j * phase))
            probs = interference / np.sum(interference)
            
            next_token = np.random.choice(len(probs), p=probs)
            
            if next_token == self.tokenizer.vocab['<EOS>']:
                break
            
            generated.append(next_token)
        
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
            batch_tokens = [model.tokenizer.encode(text) for text in batch_texts]
            batch_tokens = [t for t in batch_tokens if len(t) > 1]
            
            if not batch_tokens:
                continue
            
            # Pad sequences
            max_len = min(max(len(t) for t in batch_tokens), model.config['max_sequence_length'] - 1)
            batch = np.zeros((len(batch_tokens), max_len + 1), dtype=np.int32)
            for j, tokens in enumerate(batch_tokens):
                batch[j, :min(len(tokens), max_len + 1)] = tokens[:max_len + 1]
            
            # Forward pass and optimize
            logits = model.forward(batch[:, :-1])
            loss = compute_loss(logits, batch[:, 1:])  # Shift for next token prediction
            optimizer.step(model, loss)
            
            if global_step % 100 == 0:
                print(f"\nStep {global_step}, Loss: {loss:.4f}")
            
            if global_step % config['save_steps'] == 0:
                save_checkpoint(model, optimizer, global_step, output_dir)
            
            global_step += 1
        
        # Save epoch checkpoint
        save_checkpoint(model, optimizer, f"epoch_{epoch+1}", output_dir)
    
    print("Training completed!")

def compute_loss(logits, targets):
    """Compute cross entropy loss with quantum phase alignment"""
    # Flatten logits and targets
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.flatten()
    
    # Compute probabilities using quantum-inspired softmax
    probs = quantum_softmax(logits_flat)
    
    # Get probabilities for target tokens
    target_probs = probs[np.arange(len(targets_flat)), targets_flat]
    
    # Compute cross-entropy loss
    loss = -np.mean(np.log(target_probs + 1e-10))
    
    # Align quantum phases for gradient computation
    phase_diff = np.angle(logits_flat) - np.angle(probs)
    loss *= np.mean(np.abs(np.cos(phase_diff)))
    
    return loss

def quantum_softmax(logits):
    """Apply quantum-inspired softmax with phase alignment, stabilization, and optional clipping"""
    # Compute magnitudes and phases
    magnitudes = np.abs(logits)
    phases = np.angle(logits)
    
    # Stabilize by subtracting the maximum magnitude
    magnitudes -= np.max(magnitudes, axis=-1, keepdims=True)
    
    # Optional: Clip magnitudes to a reasonable range
    magnitudes = np.clip(magnitudes, -10, 10)  # Adjust the range as needed
    
    # Apply softmax to magnitudes
    exp_mags = np.exp(magnitudes)
    probs = exp_mags / np.sum(exp_mags, axis=-1, keepdims=True)
    
    # Recombine with phases for quantum interference
    probs = probs * np.exp(1j * phases)
    
    return probs

def save_checkpoint(model, optimizer, step, output_dir):
    """Save model checkpoint"""
    checkpoint = {
        'step': step,
        'model_state': {
            'token_embeddings': model.token_embeddings,
            'position_embeddings': model.position_embeddings,
            'config': model.config
        },
        'optimizer_state': optimizer.__dict__
    }
    
    save_path = output_dir / f'checkpoint_{step}.npz'
    np.savez(save_path, **checkpoint)

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