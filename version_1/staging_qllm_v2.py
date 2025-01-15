import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
import argparse
import os
import gc
from tqdm import tqdm
import time
import re
from typing import Dict, List, Tuple, Optional

# Determine the best available device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Update environment variables for MPS if applicable
if device == "mps":
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'  # Allow using more memory
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'   # Free memory more aggressively

def clear_memory():
    """Clear memory caches"""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def to_real_imag(complex_tensor):
    """Convert complex tensor to concatenated real and imaginary parts"""
    if device == "mps":
        # For MPS, split complex into real and imaginary
        return torch.cat([complex_tensor.real, complex_tensor.imag], dim=-1)
    return complex_tensor

class QuantumTokenizer:
    """
    A quantum-inspired tokenizer that represents tokens in phase space directly
    during the tokenization process.
    """
    def __init__(
        self, 
        phase_dim: int = 512,
        min_freq: int = 5
    ):
        # Calculate maximum vocabulary size based on quantum principles
        # Using Golden ratio (φ) and π to determine optimal vocabulary partitioning
        # We want vocab size that maintains phase space coherence
        
        PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
        
        # Calculate optimal vocab size based on phase space dimensionality
        # Using φ^n where n is chosen to maintain phase coherence
        n = int(math.log(phase_dim, PHI))
        base_size = int(PHI ** n)
        
        # Round to nearest power of 2 for computational efficiency
        # while maintaining quantum properties
        self.vocab_size = 2 ** int(math.log2(base_size))
        self.phase_dim = phase_dim
        self.min_freq = min_freq
        
        # Constants for phase space
        self.PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.PI = math.pi
        self.E = math.e
        
        # Initialize vocabulary structures
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_frequencies: Dict[str, int] = {}
        
        # Phase space mappings
        self.token_phases: Optional[torch.Tensor] = None
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def _calculate_coherence_bound(self) -> int:
        """
        Calculate theoretical maximum vocabulary size that maintains phase coherence
        """
        # Using quantum uncertainty principle analog
        # ΔE * Δt ≥ ħ/2 becomes Δphase * Δvocab ≥ π/2
        max_phase_uncertainty = math.pi / 2
        phase_precision = 2 * math.pi / self.phase_dim
        return int(max_phase_uncertainty / phase_precision)
    
    def _validate_vocab_size(self):
        """
        Validate and potentially adjust vocabulary size based on quantum constraints
        """
        coherence_bound = self._calculate_coherence_bound()
        if self.vocab_size > coherence_bound:
            print(f"Warning: Reducing vocabulary size from {self.vocab_size} to {coherence_bound} to maintain phase coherence")
            self.vocab_size = coherence_bound
    
    def _init_phase_space(self):
        """
        Initialize phase space representations for tokens using quantum-inspired
        principles with golden ratio and pi
        """
        # Validate vocabulary size before initialization
        self._validate_vocab_size()
        total_tokens = len(self.token_to_id)
        
        # Create phase angles using golden ratio for optimal distribution
        phase_angles = torch.zeros(total_tokens, self.phase_dim)
        for i in range(total_tokens):
            # Use golden ratio for phase distribution
            phases = torch.tensor([
                self.PHI * self.PI * j + (i * self.PI / total_tokens)
                for j in range(self.phase_dim)
            ])
            # Bound phases within [-π, π]
            phase_angles[i] = torch.remainder(phases, 2 * self.PI) - self.PI
        
        # Convert to complex representation
        complex_phases = torch.complex(
            torch.cos(phase_angles),
            torch.sin(phase_angles)
        )
        
        # Normalize for numerical stability
        norms = torch.sqrt(torch.sum(torch.abs(complex_phases) ** 2, dim=1))
        complex_phases = complex_phases / norms.unsqueeze(1)
        
        # Convert to real representation for MPS compatibility
        self.token_phases = to_real_imag(complex_phases)
    
    def _subword_tokenize(self, text: str) -> List[str]:
        """
        Quantum-aware subword tokenization that considers phase space coherence
        """
        # Basic cleaning
        text = text.lower().strip()
        
        # Split into initial subwords while preserving semantic units
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        subwords = []
        for word in words:
            if len(word) <= 3 or word in self.token_to_id:
                subwords.append(word)
                continue
                
            # Dynamic subword splitting based on frequency and phase coherence
            current_pos = 0
            while current_pos < len(word):
                max_coherence = -1
                best_length = 1
                
                # Try different subword lengths
                for length in range(1, min(len(word) - current_pos + 1, 10)):
                    subword = word[current_pos:current_pos + length]
                    
                    # Check if subword exists in vocabulary
                    if subword in self.token_to_id:
                        # Calculate phase coherence if phases are initialized
                        if self.token_phases is not None:
                            idx = self.token_to_id[subword]
                            phase = self.token_phases[idx]
                            coherence = torch.abs(torch.mean(phase))
                            
                            if coherence > max_coherence:
                                max_coherence = coherence
                                best_length = length
                        else:
                            # Before phase initialization, use frequency
                            freq = self.token_frequencies.get(subword, 0)
                            if freq > max_coherence:
                                max_coherence = freq
                                best_length = length
                
                subwords.append(word[current_pos:current_pos + best_length])
                current_pos += best_length
        
        return subwords
    
    def train(self, texts: List[str]):
        """
        Train tokenizer on a corpus of texts
        """
        # First pass: collect frequencies
        for text in texts:
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            for word in words:
                self.token_frequencies[word] = self.token_frequencies.get(word, 0) + 1
        
        # Filter by frequency and vocab size
        valid_tokens = sorted(
            [(token, freq) for token, freq in self.token_frequencies.items() 
             if freq >= self.min_freq],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build vocabulary
        current_idx = len(self.special_tokens)
        for token, _ in valid_tokens[:self.vocab_size - len(self.special_tokens)]:
            self.token_to_id[token] = current_idx
            self.id_to_token[current_idx] = token
            current_idx += 1
        
        # Initialize phase space
        self._init_phase_space()
    
    def encode(self, text: str, return_tensors: str = None) -> torch.Tensor:
        """
        Encode text to quantum phase space representation
        """
        if not self.token_phases is not None:
            raise ValueError("Tokenizer needs to be trained first")
        
        # Tokenize
        tokens = self._subword_tokenize(text)
        
        # Convert to IDs
        ids = [self.special_tokens['<BOS>']]
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.special_tokens['<UNK>']))
        ids.append(self.special_tokens['<EOS>'])
        
        # Convert to tensor
        if return_tensors == 'pt':
            return torch.tensor(ids)
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode from IDs back to text
        """
        tokens = []
        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                if token not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                    tokens.append(token)
        return ' '.join(tokens)
    
    def get_phase_embedding(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Get quantum phase space embeddings for tokens
        """
        if self.token_phases is None:
            raise ValueError("Phase space not initialized")
        
        return self.token_phases[ids]
   
   
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
        
        if device == "mps":
            # Handle real and imaginary parts separately
            half_dim = ground_state.shape[-1] // 2
            combined_real = ground_state[..., :half_dim] + excited_state[..., :half_dim] * torch.cos(phase)
            combined_imag = excited_state[..., half_dim:] * torch.sin(phase)
            
            # Concatenate real and imaginary parts
            combined = torch.cat([combined_real, combined_imag], dim=-1)
            return self.output_norm(combined)
        else:
            # Original complex number handling
            combined_real = ground_state + excited_state * torch.cos(phase)
            combined_imag = excited_state * torch.sin(phase)
            
            # Normalize with larger epsilon
            norm = torch.sqrt(combined_real.pow(2) + combined_imag.pow(2) + 1e-8)
            combined_real = combined_real / norm
            combined_imag = combined_imag / norm
            
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
        output_real = torch.zeros(B, L, D, device=q_real.device)
        output_imag = torch.zeros(B, L, D, device=q_imag.device)
        
        for i in range(0, L, chunk_size):
            j = min(i + chunk_size, L)
            
            # Get current chunk
            q_real_chunk = q_real[:, i:j]
            q_imag_chunk = q_imag[:, i:j]
            
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
            output_real[:, i:j] = torch.bmm(attn_weights, v_real) * 0.1
            output_imag[:, i:j] = torch.bmm(attn_weights, v_imag) * 0.1
            
            # Clear memory
            del real_diff, imag_diff, phase_diff, attn_weights
            if device == "mps":
                torch.mps.empty_cache()
        
        # Final normalization
        output_real = self.output_norm(output_real)
        output_imag = self.output_norm(output_imag)
        
        return output_real, output_imag

class SimpleQuantumLLM(nn.Module):
    """
    Minimal quantum-inspired language model with quantum tokenizer integration
    """
    def __init__(self, tokenizer: QuantumTokenizer, dim: int):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size
        self.dim = dim
        
        # Use quantum tokenizer's phase embeddings
        self.embedding = nn.Parameter(tokenizer.token_phases.clone())
        
        # Adjust dimension for real/imaginary split
        actual_dim = dim * 2 if device == "mps" else dim
        
        self.quantum_state = SimpleQuantumState(actual_dim)
        self.attention = BasicQuantumAttention(actual_dim)
        
        # Adjust pre_output_norm dimension
        self.pre_output_norm = nn.LayerNorm(actual_dim)
        self.output = nn.Linear(actual_dim, self.vocab_size)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings using quantum phase space
        embed = torch.index_select(self.embedding, 0, x.view(-1))
        embed = embed.view(x.shape[0], x.shape[1], -1)
        
        # Process through quantum state
        if device == "mps":
            # For MPS, we get a single combined tensor
            combined_state = self.quantum_state(embed)
            # Apply attention
            combined_attn = self.attention(
                combined_state, combined_state,
                combined_state, combined_state,
                combined_state, combined_state
            )
            # Project to vocabulary
            return self.output(combined_attn)
        else:
            # Original complex number handling
            state_real, state_imag = self.quantum_state(embed)
            
            # Ensure states have correct shape [batch_size, seq_len, dim]
            B, L, D = embed.shape
            state_real = state_real.view(B, L, D)
            state_imag = state_imag.view(B, L, D)
            
            # Apply attention
            attn_real, attn_imag = self.attention(
                state_real, state_imag,
                state_real, state_imag,
                state_real, state_imag
            )
            
            # Combine real and imaginary parts
            combined = torch.cat([attn_real, attn_imag], dim=-1)
            return self.output(combined)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # More aggressive clipping
        logits = torch.clamp(logits, -10, 10)
        
        # Use label smoothing and ignore padding
        ce_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            label_smoothing=0.1,
            ignore_index=0  # Assuming 0 is padding
        )
        
        # Remove phase coherence loss initially
        return ce_loss

def train_step(
    model: SimpleQuantumLLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    accumulation_steps: int = 4
) -> float:
    optimizer.zero_grad()
    
    # Even more aggressive gradient clipping
    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))
    
    try:
        logits = model(input_ids)
        loss = model.compute_loss(logits, target_ids) / accumulation_steps
        
        if not torch.isfinite(loss):
            print("Warning: Non-finite loss, skipping batch")
            return float('nan')
        
        # Skip if loss is too high
        if loss.item() > 100:
            print("Warning: Loss too high, skipping batch")
            return float('nan')
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()
        return loss.item() * accumulation_steps
        
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        return float('nan')

def generate(
    model: SimpleQuantumLLM,
    tokenizer: QuantumTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7
) -> str:
    """
    Simple generation function with quantum tokenization
    """
    model.eval()
    
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        generated = input_ids
        
        for _ in range(max_length):
            # Get predictions
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append and continue
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end token
            if next_token.item() == tokenizer.special_tokens['<EOS>']:
                break
        
        # Decode the generated tokens back to text
        generated_text = tokenizer.decode(generated[0].tolist())
        return generated_text

def load_wikitext_dataset(max_samples=None):
    """Load and preprocess the Wikitext dataset with quantum tokenizer"""
    dataset = load_dataset("wikitext", "wikitext-103-v1", split='train')
    
    if max_samples is not None:
        dataset = dataset.take(max_samples)
    
    # Initialize and train quantum tokenizer
    tokenizer = QuantumTokenizer(phase_dim=512)  # Match model dimension
    
    # Collect all texts for tokenizer training
    print("Training tokenizer...")
    texts = [example['text'] for example in dataset]
    tokenizer.train(texts)
    
    def preprocess_function(examples):
        # Handle batch of texts
        batch_encodings = []
        for text in examples['text']:
            # Tokenize each text individually
            encoding = tokenizer.encode(
                text,
                return_tensors='pt'
            )
            
            # Pad or truncate to fixed length
            max_length = 512
            if encoding.size(0) < max_length:
                # Pad
                pad_length = max_length - encoding.size(0)
                encoding = torch.cat([
                    encoding,
                    torch.zeros(pad_length, dtype=torch.long)
                ])
            else:
                # Truncate
                encoding = encoding[:max_length]
            
            batch_encodings.append(encoding)
        
        # Stack all encodings into a batch
        return {
            'input_ids': torch.stack(batch_encodings) if batch_encodings else torch.zeros((0, max_length))
        }
    
    # Show progress bar for dataset processing
    print("Processing dataset...")
    with tqdm(desc="Tokenizing", total=1) as pbar:
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=1,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        pbar.update(1)
    
    return tokenized_dataset, tokenizer

 
    
def main(args):
    # Load dataset and initialize tokenizer
    if args.mode == 'train':
        dataset, tokenizer = load_wikitext_dataset(max_samples=args.max_samples)
    else:
        # For generation mode, initialize tokenizer with minimal vocab
        tokenizer = QuantumTokenizer(phase_dim=512)
        # You might want to load a pre-trained tokenizer state here
    
    # Initialize model with tokenizer
    model = SimpleQuantumLLM(tokenizer=tokenizer, dim=512).to(device)
    
    # Use AdamW with better defaults for stability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,  # Lower learning rate
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )

    if args.mode == 'train':
        # Convert dataset to list for length calculation
        dataset_list = list(dataset)
        total_batches = len(dataset_list)
        
        print(f"\nStarting training...")
        print(f"Total batches: {total_batches}")
        
        # Training loop with progress bars
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Initialize statistics
            epoch_loss = 0
            valid_batches = 0
            start_time = time.time()
            
            # Create progress bar for this epoch
            progress_bar = tqdm(dataset_list, desc=f"Training", 
                              leave=True, 
                              total=total_batches)
            
            for batch in progress_bar:
                # Process batch - batch is now a dictionary with 'input_ids'
                input_ids = batch['input_ids'].to(device)
                if len(input_ids.shape) == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                target_ids = input_ids.clone()  # Use input_ids as targets
                
                # Train step
                loss = train_step(model, optimizer, input_ids, target_ids, accumulation_steps=4)
                
                # Update statistics
                if not math.isnan(loss):
                    epoch_loss += loss
                    valid_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'avg_loss': f"{epoch_loss/valid_batches:.4f}",
                        'valid_batches': valid_batches
                    })
                
                clear_memory()
            
            # Epoch statistics
            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('nan')
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Valid Batches: {valid_batches}/{total_batches}")
            print(f"Time: {epoch_time:.2f}s")

    elif args.mode == 'generate':
        # Generate text from a prompt
        generated_text = generate(
            model, 
            tokenizer,
            args.prompt,
            max_length=args.max_length, 
            temperature=args.temperature
        )
        print("Generated text:", generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum-Inspired Language Model")
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True, help="Mode: train or generate")
    parser.add_argument('--prompt', type=str, help="Prompt for text generation")
    parser.add_argument('--max_length', type=int, default=100, help="Maximum length of generated text")
    parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument('--max_samples', type=int, default=100, help="Maximum number of samples to load from the dataset")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")

    args = parser.parse_args()
    main(args)