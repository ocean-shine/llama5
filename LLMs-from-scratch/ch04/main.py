from importlib.metadata import version

import matplotlib
import tiktoken
import torch

print("matplotlib version", version("matplotlib"))
print("torch version", version("torch"))
print("tiktoken version", version("tiktoken"))


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x  = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x
    

class DummyLayerNorm(nn.Module):
    def __init__(self, normlized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim = 0)
print(batch)


torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)
print("Output shape:", logits.shape)
print(logits)


torch.manual_seed(123)
batch_example = torch.randn(2,5)
print(batch_example)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim = -1, keepdim=True)
var = out.var(dim = -1, keepdim=True)

print("Mean:", mean)
print("Var:", var)

out_norm = (out - mean)/ torch.sqrt(var)

mean = out_norm.mean(dim = -1, keepdim=True)
var = out_norm.var(dim = -1, keepdim=True)

print("Mean:", mean)
print("Var:", var)

torch.set_printoptions(sci_mode = False)

print("Mean:", mean)
print("Var:", var)



class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        var = x.var(dim = -1, keepdim=True)

        norm_x = (x - mean)/ torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift
    
ln = LayerNorm(emb_dim = 5)
out_ln = ln(batch_example)

mean = out_ln.mean(dim = -1, keepdim = True)
var = out_ln.var(dim = -1, keepdim = True)

print("Mean:", mean)
print("Var:", var)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1+ torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8,3))
for i , (y,label) in enumerate(zip([y_gelu, y_relu], ["GULE", "RELU"]), 1):
    plt.subplot(1,2,i)
    plt.plot(x,y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()



class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4* cfg["emb_dim"]),
            GELU(),
            nn.Linear(4* cfg["emb_dim"], cfg["emb_dim"]), 
        )

    def forward(self, x):
        return self.layers(x)
    

ffn = FeedForward(GPT_CONFIG_124M)

x = torch.rand(2,3,768)
out = ffn(x)
print(out.shape)



class ExampleDeepNeuralnetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.sahpe == layer_output.shape:
                x = x +layer_output
            else :
                x = layer_output
        return x
    
def print_gradients(model , x):

    output = model(x)
    target = torch.tensor([0.])

    loss = nn.MSELoss()
    loss = loss(output, target)
    
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")















None
