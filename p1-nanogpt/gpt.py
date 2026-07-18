import torch

# ---- load data ----
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"corpus length: {len(text):,} chars")

# ---- character-level tokenizer ----
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"vocab: {''.join(chars)}")
print(f"vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
itos = {i: ch for i, ch in enumerate(chars)}  # int -> string
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hello"))
print(decode(encode("hello")))

# ---- encode entire corpus, split train/val ----
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"train: {len(train_data):,} tokens, val: {len(val_data):,} tokens")

torch.manual_seed(1337)
block_size = 8   # max context length for now
batch_size = 4   # sequences per batch

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print("inputs:", xb.shape)   # (4, 8)
print("targets:", yb.shape)  # (4, 8)
print(xb)
print(yb)