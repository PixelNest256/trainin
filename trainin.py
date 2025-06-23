import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from flash_attn import flash_attn_func
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

VOCAB_SIZE = 8000
EMBED_SIZE = 768
NUM_HEADS = 8
NUM_LAYERS = 5
HIDDEN_SIZE = 3072
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0005
MAX_SEQ_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

class SpecificPairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_seq_len, vocab_subset):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.vocab_subset = vocab_subset
        self.data = []
        for context, target in pairs:
            context_tokens = tokenizer.encode(context, add_special_tokens=False)[:max_seq_len - 1]
            target_tokens = tokenizer.encode(target, add_special_tokens=False)
            if target_tokens and context_tokens:
                target_token = target_tokens[0]
                if target_token in vocab_subset:
                    padded = context_tokens + [tokenizer.pad_token_id] * (max_seq_len - len(context_tokens) - 1)
                    self.data.append((padded, target_token))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class FlashAttentionTransformer(nn.Module):
    def __init__(self, embed_layer, vocab_size, embed_size, num_heads, hidden_size, num_layers, dropout=0.1):
        super(FlashAttentionTransformer, self).__init__()
        self.embedding = embed_layer
        self.pos_encoder = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, embed_size))
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'qkv_proj': nn.Linear(embed_size, 3 * embed_size),
                'out_proj': nn.Linear(embed_size, embed_size),
                'ffn': nn.Sequential(
                    nn.Linear(embed_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, embed_size)
                ),
                'layer_norm1': nn.LayerNorm(embed_size),
                'layer_norm2': nn.LayerNorm(embed_size)
            }) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, src):
        batch_size, seq_len = src.shape
        x = self.embedding(src) + self.pos_encoder[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer['layer_norm1'](x)
            qkv = layer['qkv_proj'](x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                causal=True,
                softmax_scale=1.0 / (self.head_dim ** 0.5)
            )
            attn_output = attn_output.view(batch_size, seq_len, -1)
            x = x + self.dropout(layer['out_proj'](attn_output))
            x = layer['layer_norm2'](x)
            x = x + self.dropout(layer['ffn'](x))
        
        output = self.fc(x[:, -1, :])
        return output

vocab_subset = list(range(VOCAB_SIZE))

sample_pairs = [
    ("I love to", "eat"),
    ("The sun is", "bright"),
    ("She wants to", "learn"),
    ("He is very", "happy"),
    ("We go to", "school")
]

dataset = SpecificPairDataset(sample_pairs, tokenizer, MAX_SEQ_LEN, vocab_subset)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

embed_layer = qwen_model.model.embed_tokens
model = FlashAttentionTransformer(embed_layer, VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params / 1e6:.3f}M")

def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            preds = output.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += tgt.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

def save_model():
    torch.save(model.state_dict(), "qwen_embedding_transformer.pth")

def predict(context):
    model.eval()
    with torch.no_grad():
        context_tokens = tokenizer.encode(context, add_special_tokens=False)[:MAX_SEQ_LEN-1]
        padded = context_tokens + [tokenizer.pad_token_id] * (MAX_SEQ_LEN - len(context_tokens) - 1)
        context_tokens = torch.tensor([padded], dtype=torch.long).to(DEVICE)
        output = model(context_tokens)
        pred_token = output.argmax(dim=-1).item()
        return tokenizer.decode([pred_token])

if __name__ == "__main__":
    train()
    save_model()
    print("Training completed and model saved.")
    
    for context, _ in sample_pairs:
        pred = predict(context)
        print(f"Context: {context}, Predicted next token: {pred}")
