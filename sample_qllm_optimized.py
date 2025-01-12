import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import time

class FastQuantumOps:
    """Handle quantum operations using fast trigonometric approximations"""
    
    @staticmethod
    def fast_sin(x):
        """Fast sine approximation using Taylor series first terms"""
        # Handle tensor input properly
        if isinstance(x, torch.Tensor):
            # Normalize to -π to π range using modulo
            x = x % (2 * torch.pi)
            # Shift values > π to -π to π range
            x = torch.where(x > torch.pi, x - 2 * torch.pi, x)
            
            # Fast approximation: x - x^3/6
            x2 = x * x
            return x * (1.0 - x2 / 6.0)
        else:
            # Handle scalar input
            x = x % (2 * np.pi)
            if x > np.pi:
                x -= 2 * np.pi
            x2 = x * x
            return x * (1.0 - x2 / 6.0)
    
    @staticmethod
    def fast_phase(real, imag):
        """Fast phase approximation using lookup table"""
        # Pre-computed lookup table for common angles
        ANGLES = torch.linspace(0, 2*np.pi, 256).to("mps")
        
        # Fast magnitude approximation
        mag = torch.abs(real) + torch.abs(imag)  # Manhattan distance approximation
        
        # Quick angle approximation using ratio
        angle_idx = ((torch.atan2(imag, real) + np.pi) * 128 / np.pi).long()
        return ANGLES[angle_idx % 256], mag

    @staticmethod 
    def quantum_interference(x, y):
        """Simulate quantum interference using fast approximations"""
        # Use separable 1D operations instead of 2D
        x_mag = torch.abs(x)
        y_mag = torch.abs(y)
        
        # Approximate interference pattern using broadcasting
        diff = (x - y) * 0.5
        interference = (x_mag + y_mag) * torch.cos(diff)
        return interference

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
        """Initialize using fast trig approximations"""
        t = torch.linspace(0, 2*torch.pi, self.head_dim).to("mps")
        patterns = []
        
        for h in range(self.num_heads):
            phase = 2 * torch.pi * h / self.num_heads
            # Create phase tensor and add to t
            phase_tensor = torch.full_like(t, phase)
            pattern = FastQuantumOps.fast_sin(t + phase_tensor)
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
        """Encode quantum state using separable 1D transforms"""
        B, L, D = x.shape
        
        # Use separable 1D height-maps
        h = torch.linspace(0, 1, D).to("mps")
        v = torch.linspace(0, 1, D).to("mps")
        
        # Create interference pattern using outer product
        h_pattern = FastQuantumOps.fast_sin(h * torch.pi)
        v_pattern = FastQuantumOps.fast_sin(v * torch.pi)
        
        # Reshape for efficient computation
        x_flat = x.view(B * L, D)  # [B*L, D]
        
        # Project through quantum state while maintaining dimensions
        h_proj = torch.matmul(x_flat, h_pattern.unsqueeze(1))  # [B*L, 1]
        v_proj = torch.matmul(x_flat, v_pattern.unsqueeze(1))  # [B*L, 1]
        
        # Expand projections to match original dimension
        h_proj = h_proj.expand(-1, D)  # [B*L, D]
        v_proj = v_proj.expand(-1, D)  # [B*L, D]
        
        # Combine projections while maintaining dimensions
        output = (h_proj + v_proj) / 2.0  # Simple averaging
        output = output.view(B, L, D)  # Reshape back to original dimensions
        
        # Add residual connection and normalize
        output = output + x
        output = output / (torch.norm(output, dim=-1, keepdim=True) + 1e-8)
        
        return output

class FastQuantumLLM(nn.Module):
    """Optimized quantum-inspired language model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get('tokenizer_name', 'bert-base-uncased')
        )
        
        # Update vocab size from tokenizer
        self.config['vocab_size'] = len(self.tokenizer.get_vocab())
        
        # Embeddings
        self.token_embedding = nn.Embedding(
            config['vocab_size'], 
            config['dim']
        ).to("mps")
        
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config['max_sequence_length'], config['dim'])
        ).to("mps")
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            FastQuantumAttention(config['dim'], config['num_heads'])
            for _ in range(config['num_layers'])
        ]).to("mps")
        
        # Output projection
        self.output_proj = nn.Linear(config['dim'], config['vocab_size']).to("mps")
        
        # Quantum state encoder
        self.quantum_encoder = FastQuantumState()
        
        # For tracking training
        self.step_count = 0
        
        # Add layer norm for stability
        self.layer_norm = nn.LayerNorm(config['dim']).to("mps")
        
        # Add additional normalization layers
        self.input_norm = nn.LayerNorm(config['dim']).to("mps")
        self.output_norm = nn.LayerNorm(config['dim']).to("mps")
        
    def forward(self, input_ids, return_loss=True):
        """Forward pass with optional loss computation"""
        B, L = input_ids.shape
        
        # Get embeddings
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :L, :]
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # Apply quantum encoding with proper dimensions
        x = self.quantum_encoder.encode_state(x, self.config['dim'])
        
        # Process through attention layers
        for layer in self.attention_layers:
            # Apply attention with residual connection
            attn_out = layer(x)
            x = x + attn_out
            
            # Apply simple quantum interference
            x = FastQuantumOps.quantum_interference(x, attn_out)
            
            # Add normalization after each layer
            x = self.output_norm(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        if return_loss and input_ids is not None:
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return loss
            
        return logits
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_k=50):
        """Generate text using the model"""
        # Encode prompt
        input_ids = self.tokenizer.encode(
            prompt, 
            return_tensors='pt'
        ).to("mps")
        
        generated = input_ids[0].tolist()
        
        # Generate tokens
        for _ in range(max_length):
            # Get predictions
            inputs = torch.tensor([generated[-self.config['max_sequence_length']:]]).to("mps")
            with torch.no_grad():
                outputs = self.forward(inputs, return_loss=False)
                next_token_logits = outputs[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Convert to probabilities
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample next token
            next_token_id = top_k_indices[torch.multinomial(probs, num_samples=1)]
            
            generated.append(next_token_id.item())
            
            # Stop if we exceed max length or generate EOS token
            if len(generated) >= max_length or next_token_id == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated)

def train_model(model, train_data, config):
    """Train the model"""
    print("Starting training...")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate']
    )
    
    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        total_loss = 0
        num_batches = 0
        
        for i in tqdm(range(0, len(train_data), config['batch_size'])):
            batch_texts = train_data[i:i + config['batch_size']]
            
            # Tokenize batch
            encodings = model.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config['max_sequence_length'],
                return_tensors='pt'
            ).to("mps")
            
            # Forward pass and loss
            loss = model(encodings.input_ids)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if num_batches % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"\nStep {num_batches}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, config['output_dir'])

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

if __name__ == "__main__":
    # Configuration
    config = {
        'dim': 512,  # Reduced dimension for faster training
        'num_heads': 8,
        'num_layers': 6,
        'max_sequence_length': 1024,
        'batch_size': 16,
        'epochs': 3,
        'learning_rate': 1e-4,
        'output_dir': 'quantum_checkpoints',
        'tokenizer_name': 'bert-base-uncased'
    }
    
    # Initialize model
    model = FastQuantumLLM(config).to("mps")
    
    # Sample training data
    train_data = [
        "This is a sample text for training.",
        "The quantum-inspired model learns patterns.",
        "Fast and efficient text generation."
    ]
    
    # Train model
    train_model(model, train_data, config)
    
    # Generate text
    prompt = "Once upon a time"
    generated_text = model.generate(prompt, max_length=50)
    print(f"\nGenerated text:\n{generated_text}") 