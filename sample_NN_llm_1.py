import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AdamW
from datasets import load_dataset
import argparse
import os
from tqdm import tqdm
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Determine the best available device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Update environment variables for MPS if applicable
if device == "mps":
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.5'
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.3'

# Define the LLM model
class CausalSelfAttention(nn.Module):
    """Causal self-attention layer"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(0.1)
        self.mask = None

    def forward(self, x):
        B, L, D = x.shape
        
        # Create causal mask if not created or if sequence length changed
        if self.mask is None or self.mask.size(0) != L:
            # Create mask without registering as buffer
            mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
            self.mask = mask.to(x.device)
        
        # Linear projections
        q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores.masked_fill_(self.mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)

class TransformerBlock(nn.Module):
    """Transformer block with causal attention"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, device='cuda'):
        super().__init__()
        self.device = device
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, 512, dim))
        
        # Transformer blocks with causal attention
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights
        self.token_embedding.weight = self.fc_out.weight
        
        # Initialize
        self.apply(self._init_weights)
        self.to(device)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Get embeddings
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding[:, :L, :]
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        
        # Transform
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.fc_out(x)
        
        return logits

    def generate(self, prompt, tokenizer, max_length=50, temperature=0.7, top_k=50, top_p=0.9):
        """Generate text with improved sampling and nucleus (top-p) filtering"""
        self.eval()
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = input_ids.tolist()[0]
        prompt_length = len(generated)
        max_new_tokens = max_length - prompt_length
        
        # Generation loop
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Get model output
                inputs = torch.tensor([generated[-511:]]).to(self.device)  # Limit context window
                outputs = self(inputs)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_probs = F.softmax(top_k_logits, dim=-1)
                
                # Apply nucleus (top-p) filtering
                cumulative_probs = torch.cumsum(next_token_probs, dim=-1)
                nucleus_mask = cumulative_probs < top_p
                nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
                nucleus_mask[..., 0] = True
                
                # Filter logits and probabilities
                filtered_logits = top_k_logits[nucleus_mask]
                filtered_indices = top_k_indices[nucleus_mask]
                filtered_probs = F.softmax(filtered_logits, dim=-1)
                
                # Sample next token
                next_token_idx = torch.multinomial(filtered_probs, num_samples=1)
                next_token = filtered_indices[next_token_idx]
                
                # Append to generated
                generated.append(next_token.item())
                
                # Stop if we generate EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode and clean up the generated text
        generated_text = tokenizer.decode(generated[prompt_length:], skip_special_tokens=True)
        return f"{prompt} {generated_text.strip()}"

# Custom dataset for handling text data
class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = list(dataset)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

def save_model(model, tokenizer, config, epoch, output_dir):
    """Save model checkpoint"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config,
    }
    
    model_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

def load_model(checkpoint_path, device=device):
    """Load model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', get_default_config())
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    
    # Initialize model
    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        dim=config['dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        device=device
    )
    
    # Clean the state dict of attention masks before loading
    state_dict = checkpoint['model_state_dict']
    # Remove attention mask keys
    state_dict = {k: v for k, v in state_dict.items() if 'attention.mask' not in k}
    
    # Load cleaned state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, tokenizer, config

def train_model(model, tokenizer, config):
    config['vocab_size'] = tokenizer.vocab_size
    
    dataset = load_dataset(
        config['dataset_name'],
        config['dataset_config'],
        split='train',
        streaming=True
    )
    
    if config.get('max_samples'):
        dataset = dataset.take(config['max_samples'])
    
    train_dataset = TextDataset(dataset, tokenizer, config['max_sequence_length'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # Use AdamW with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] * len(train_loader),
        eta_min=1e-6
    )

    model.train()
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            # Forward pass
            outputs = model(input_ids)
            
            # Calculate loss
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, config['vocab_size']),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, tokenizer, config, f"best", config['output_dir'])
        
        # Save regular checkpoint
        save_model(model, tokenizer, config, epoch + 1, config['output_dir'])

def main(config, mode='train', checkpoint_path=None, prompt=None):
    if mode == 'train':
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        config['vocab_size'] = tokenizer.vocab_size
        
        model = SimpleLLM(
            vocab_size=tokenizer.vocab_size,
            dim=config['dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            device=device
        )
        
        train_model(model, tokenizer, config)
    
    elif mode == 'generate':
        if not checkpoint_path:
            raise ValueError("Checkpoint path is required for generation mode")
        if not prompt:
            raise ValueError("Prompt is required for generation mode")
            
        try:
            model, tokenizer, loaded_config = load_model(checkpoint_path, device)
            generated_text = model.generate(
                prompt=prompt,
                tokenizer=tokenizer,
                max_length=config.get('max_length', 100),  # Increased default length
                temperature=config.get('temperature', 0.7),
                top_k=config.get('top_k', 50)
            )
            print(f"\nGenerated text:\n{generated_text}")
        except Exception as e:
            print(f"Error during generation: {str(e)}")

def get_default_config():
    return {
        'dim': 512,  # Increased from 128
        'num_heads': 8,
        'num_layers': 6,  # Increased from 4
        'max_sequence_length': 256,
        'batch_size': 4,  # Increased from 2
        'epochs': 5,  # Increased from 3
        'learning_rate': 3e-4,
        'tokenizer_name': 'gpt2',  # Changed to GPT-2 tokenizer
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-103-v1',  # Changed to larger dataset
        'output_dir': 'model_checkpoints',
        'max_samples': 10000,  # Increased from 200
    }

def run_from_cli():
    parser = argparse.ArgumentParser(description='Simple Language Model Training and Generation')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True,
                      help='Mode to run the model: train or generate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--max_sequence_length', type=int, default=256, help='Maximum sequence length (default: 256)')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='Tokenizer name (default: bert-base-uncased)')
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name (default: wikitext)')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-v1', help='Dataset config (default: wikitext-2-v1)')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint for generation')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for text generation')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of generated text')
    parser.add_argument('--max_samples', type=int, default=200, help='Maximum number of training samples (default: 200)')

    args = parser.parse_args()
    config = get_default_config()
    config.update(vars(args))

    main(config, mode=args.mode, checkpoint_path=args.checkpoint_path, prompt=args.prompt)

if __name__ == "__main__":
    run_from_cli()
