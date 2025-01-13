import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import gc
import argparse
import sys

# Determine the best available device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

class QuantumTokenizer:
    def __init__(self, vocab_size=None):
        self.vocab = self._initialize_vocabulary()
        self.vocab_size = len(self.vocab)
        self.phi = (1 + np.sqrt(5)) / 2  # Golden Ratio
        self.e = np.e  # Euler's number
        self.pi = np.pi
        self.token_patterns = self._initialize_quantum_patterns()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.char_waves = self._initialize_char_waves()
        self.eos_token_id = 3  # <EOS> token
    
    def _initialize_quantum_patterns(self):
        patterns = {}
        primes = self._generate_primes(100)
        for i, prime in enumerate(primes):
            phase = 2 * self.pi * (i / len(primes)) * self.phi
            patterns[prime] = np.exp(1j * phase)
        return patterns
    
    def _generate_primes(self, n):
        primes = []
        num = 2
        while len(primes) < n:
            if all(num % prime != 0 for prime in primes):
                primes.append(num)
            num += 1
        return primes
    
    def _initialize_vocabulary(self):
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<MASK>': 4
        }
        common_words = """the be to of and a in that have I it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us""".split()
        for word in common_words:
            vocab[word] = len(vocab)
        return vocab
    
    def extend_vocabulary(self, new_words):
        for word in new_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        return self.vocab_size
    
    def _initialize_char_waves(self):
        waves = {}
        for i, char in enumerate(self.vocab):
            freq = (i + 1) * self.phi
            phase = 2 * self.pi * (i / len(self.vocab))
            waves[char] = lambda x, f=freq, p=phase: np.sin(f * x + p)
        return waves
    
    def encode(self, text, return_tensors='pt'):
        tokens = self._quantum_tokenize(text)
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        if return_tensors == 'pt':
            return torch.tensor([indices], dtype=torch.long, device=device)
        elif return_tensors == 'wave':
            encoded = torch.zeros((len(tokens), self.vocab_size), dtype=torch.float32, device=device)
            for i, token in enumerate(tokens):
                if token in self.vocab:
                    pattern = self._create_token_pattern(token)
                    encoded[i] = torch.tensor(pattern, device=device)
                else:
                    encoded[i] = self._create_unknown_pattern(token)
            return encoded
        return indices
    
    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        text = [self.reverse_vocab.get(idx, '<UNK>') for idx in token_ids]
        return ' '.join([token for token in text if token not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']])
    
    def _quantum_tokenize(self, text):
        text = text.lower().strip()
        words = text.split()
        return [word if word in self.vocab else '<UNK>' for word in words]
    
    def _create_token_pattern(self, token):
        pattern = np.zeros(self.vocab_size)
        idx = self.vocab[token]
        for i in range(self.vocab_size):
            phase = 2 * self.pi * (i - idx) / self.vocab_size
            pattern[i] = np.cos(phase * self.phi) * np.exp(-abs(i-idx)/self.vocab_size)
        return pattern
    
    def _create_unknown_pattern(self, token):
        pattern = np.zeros(self.vocab_size)
        for known_token, idx in self.vocab.items():
            similarity = self._quantum_similarity(token, known_token)
            phase = 2 * self.pi * similarity * self.phi
            pattern[idx] = np.cos(phase) * similarity
        return torch.tensor(pattern / np.sqrt(np.sum(pattern**2) + 1e-8), device=device)
    
    def _quantum_similarity(self, token1, token2):
        distance = self._weighted_levenshtein(str(token1), str(token2))
        return np.exp(-distance * self.phi)
    
    def _weighted_levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self._weighted_levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1/self.phi
                deletions = current_row[j] + 1/self.phi
                substitutions = previous_row[j] + (0 if c1 == c2 else 1/self.e)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

class QuantumLanguageStructure:
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.fib_sequence = self._generate_fibonacci(512).to(device)
    
    def _generate_fibonacci(self, n):
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return torch.tensor(fib, dtype=torch.float32)
    
    def apply_structure(self, tokens):
        B, L, D = tokens.shape
        pattern = self.fib_sequence[:L] / (self.fib_sequence[L-1] + 1e-8)
        pattern = torch.clamp(pattern, -5, 5).to(tokens.device).view(1, L, 1).expand(B, -1, D)
        return tokens * pattern * 0.1 

class FastQuantumOps:
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

class FastQuantumAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patterns = self._init_patterns().to(device)
    
    def _init_patterns(self):
        t = torch.linspace(0, 2 * torch.pi, self.head_dim)
        patterns = []
        for h in range(self.num_heads):
            phase = 2 * torch.pi * h / self.num_heads
            pattern = FastQuantumOps.phase_encoding(t + phase)
            patterns.append(pattern)
        return torch.stack(patterns)
    
    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        x = x.view(B, L, H, -1)
        x = x * self.patterns.view(1, 1, H, -1)
        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, x).view(B, L, D)

class FastQuantumState:
    @staticmethod
    def encode_state(x, dim):
        """Memory-efficient state encoding with improved phase representation"""
        B, L, D = x.shape
        chunk_size = min(L, 64)
        outputs = []
        
        for i in range(0, L, chunk_size):
            chunk = x[:, i:i+chunk_size, :].contiguous()
            curr_chunk_size = chunk.size(1)
            
            # Create height map patterns
            h = torch.linspace(0, torch.pi, D, device=device)
            v = torch.linspace(0, torch.pi, D, device=device)
            
            h_pattern = torch.sin(h)
            v_pattern = torch.cos(v)
            
            # Process chunk efficiently
            chunk_flat = chunk.reshape(B * curr_chunk_size, D)
            h_proj = torch.matmul(chunk_flat, h_pattern.unsqueeze(1))
            v_proj = torch.matmul(chunk_flat, v_pattern.unsqueeze(1))
            
            # Combine projections
            output = (h_proj + v_proj) / 2.0
            output = output.reshape(B, curr_chunk_size, D)
            
            outputs.append(output)
            
            # Clear intermediate tensors
            del h_pattern, v_pattern, h_proj, v_proj, chunk_flat
        
        output = torch.cat(outputs, dim=1)
        output = output + x  # Add residual connection
        output = output / (torch.norm(output, dim=-1, keepdim=True) + 1e-8)
        
        return output

class QuantumPreprocessor:
    def __init__(self, vocab_size, dim):
        self.vocab_size = vocab_size
        self.dim = dim
        self.phi = (1 + np.sqrt(5)) / 2
        self.phase_angles = torch.linspace(0, 2 * torch.pi, vocab_size, device=device)
        self.position_encodings = self._init_position_encodings()
    
    def _init_position_encodings(self):
        encodings = {}
        common_lengths = [32, 64, 128, 256, 512, 1024]
        for length in common_lengths:
            positions = torch.arange(length, device=device).float()
            encodings[length] = torch.sin(positions.unsqueeze(1) * (1.0 / math.sqrt(self.dim)))
        return encodings
    
    def prepare_state(self, token_ids):
        B, L = token_ids.shape
        states = torch.zeros((B, L, self.dim), device=device)
        token_phases = self.phase_angles[token_ids]
        states = torch.sin(token_phases.unsqueeze(-1) * torch.arange(self.dim, device=device))
        return states

class QuantumLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = QuantumTokenizer()
        self.config['vocab_size'] = self.tokenizer.vocab_size
        self.scale = 1.0 / math.sqrt(config['dim'])
        
        # Move all components to the correct device during initialization
        self.token_embedding = nn.Embedding(config['vocab_size'], config['dim'])
        self.pos_embedding = nn.Parameter(torch.randn(1, config['max_sequence_length'], config['dim']) * self.scale)
        
        self.attention_layers = nn.ModuleList([
            FastQuantumAttention(config['dim'], config['num_heads'])
            for _ in range(config['num_layers'])
        ])
        
        self.output_proj = nn.Linear(config['dim'], self.config['vocab_size'])
        self.quantum_encoder = FastQuantumState()
        
        self.layer_norm = nn.LayerNorm(config['dim'])
        self.input_norm = nn.LayerNorm(config['dim'])
        self.output_norm = nn.LayerNorm(config['dim'])
        
        # Initialize embedding weights
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.scale)
        
        # Move the entire model to the device
        self.to(device)
    
    def forward(self, input_ids, return_loss=True):
        input_ids = input_ids.long().to(device)
        B, L = input_ids.shape
        
        # Get quantum states
        quantum_states = QuantumPreprocessor(
            self.config['vocab_size'], 
            self.config['dim']
        ).prepare_state(input_ids)
        
        # Get embeddings and add quantum states
        x = torch.tanh(self.token_embedding(input_ids) * 0.1)
        x = x + quantum_states
        
        # Add positional encoding
        positions = torch.arange(L, device=device).float()
        pos_phase = torch.sin(positions.unsqueeze(1) * self.scale)
        x = x + pos_phase.unsqueeze(0) * 0.1
        
        # Process through layers
        x = self.input_norm(x)
        
        for layer in self.attention_layers:
            residual = x
            attn = layer(x) * 0.1
            x = FastQuantumOps.quantum_interference(residual, attn)
            x = torch.tanh(x)
            x = self.layer_norm(x)
        
        x = self.output_norm(x)
        logits = torch.tanh(self.output_proj(x))
        
        if return_loss and input_ids is not None:
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1
            )
            return loss
        
        return logits

# Training function
def train_model(model, config):
    """Train the model using HuggingFace dataset"""
    print("Loading dataset...")
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
        vocab_words.update(item['text'].lower().split())
    
    # Update vocabulary
    model.tokenizer.extend_vocabulary(vocab_words)
    print(f"Final vocabulary size: {model.tokenizer.vocab_size}")
    
    # Create training dataset
    train_dataset = TextDataset(
        dataset,
        model.tokenizer,
        config['max_sequence_length']
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-5),
        eps=1e-8
    )
    
    model.train()
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            
            optimizer.zero_grad()
            loss = model(input_ids)
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches % config['log_every'] == 0:
                    print(f"\nBatch {num_batches}, Loss: {loss.item():.4f}")
            
            # Clear cache periodically
            if num_batches % 100 == 0:
                clear_memory()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, config['output_dir'])

class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = list(dataset)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = tokenizer.vocab_size
        self.device = device  # Add device reference
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Move input_ids to the correct device immediately after encoding
        input_ids = self.tokenizer.encode(item['text'], return_tensors='pt').to(self.device)
        
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
        elif input_ids.size(1) < self.max_length:
            # Create padding on the same device as input_ids
            padding = torch.zeros(
                1, 
                self.max_length - input_ids.size(1), 
                dtype=torch.long,
                device=self.device
            )
            input_ids = torch.cat([input_ids, padding], dim=1)
        
        # Create attention mask on the same device
        attention_mask = (input_ids != 0).float()
        
        # Ensure token ids are within vocabulary bounds
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0)
        }

def get_default_config():
    return {
        'dim': 128,
        'num_heads': 8,
        'num_layers': 4,
        'max_sequence_length': 256,
        'batch_size': 2,
        'epochs': 3,
        'learning_rate': 1e-4,
        'output_dir': 'quantum_checkpoints',
        'tokenizer_name': 'bert-base-uncased',
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-103-v1',
        'max_samples': 200,
        'log_every': 500
    }

class QuantumStateCache:
    """Manages caching of quantum states and patterns"""
    def __init__(self, max_cache_size=1000):
        self.max_cache_size = max_cache_size
        self.state_cache = {}
        self.pattern_cache = {}
        self.access_count = {}
    
    def get_state(self, key, compute_fn):
        if key not in self.state_cache:
            if len(self.state_cache) >= self.max_cache_size:
                self._evict_least_used()
            self.state_cache[key] = compute_fn()
        self.access_count[key] = self.access_count.get(key, 0) + 1
        return self.state_cache[key]
    
    def _evict_least_used(self):
        if not self.access_count:
            return
        min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        del self.state_cache[min_key]
        del self.access_count[min_key]

def clear_memory():
    """Clear memory caches"""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def run_from_cli():
    """Function to handle command-line interface"""
    parser = argparse.ArgumentParser(description='Quantum Language Model Training and Generation')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True,
                      help='Mode to run the model: train or generate')
    
    # Generation arguments
    parser.add_argument('--prompt', type=str,
                      help='Text prompt for generation (required if mode is generate)')
    parser.add_argument('--max_length', type=int, default=50,
                      help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for text generation')
    
    # Training arguments - Adding these back from version 4
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=200,
                      help='Maximum number of training samples')
    
    # Model loading/saving
    parser.add_argument('--checkpoint', type=str,
                      help='Path to model checkpoint file')
    parser.add_argument('--output_dir', type=str, default='quantum_checkpoints',
                      help='Directory for saving checkpoints')
    
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
    main(args.mode, config, args.prompt, args.checkpoint, args.max_length, args.temperature)

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

def main(mode, config, prompt=None, checkpoint=None, max_length=50, temperature=0.7):
    """Main function to run the model in train or generate mode"""
    # Initialize model
    model = QuantumLLM(config)
    model = model.to(device)
    
    if mode == 'train':
        print("Training model...")
        print(f"Epochs: {config['epochs']}, Batch size: {config['batch_size']}")
        print(f"Learning rate: {config['learning_rate']}, Max samples: {config['max_samples']}")
        train_model(model, config)
    else:  # generate mode
        # Load checkpoint if provided
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
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
        
        # Decode generated text
        generated_text = model.tokenizer.decode(generated)
        print(f"\nGenerated text:\n{generated_text}")

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
                clear_memory()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, config['output_dir'])

if __name__ == "__main__":
    try:
        run_from_cli()
    except SystemExit:
        pass