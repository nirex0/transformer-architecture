import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================
# Self-Attention Mechanism
# ============================
class SelfAttention(nn.Module):
    """
    A Self-Attention layer that allows tokens to interact with each other.
    It computes the attention scores based on the Query, Key, and Value matrices.
    """
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"

        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_length, embed_size = x.shape

        # (batch, seq_len, embed_size) -> (batch, seq_len, embed_size)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        # (batch, seq_len, embed_size) -> (batch, heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        # (batch, heads, seq_len, head_dim) x (batch, heads, head_dim, seq_len) -> (batch, heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = F.softmax(attention_scores, dim=-1)

        # (batch, heads, seq_len, seq_len) x (batch, heads, seq_len, head_dim) -> (batch, heads, seq_len, head_dim)
        out = torch.matmul(attention, V)
        
        # Reshape back to original dimensions
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)

        return self.fc_out(out)


# ============================
# Masked Multi-Head Self-Attention
# ============================
class MaskedSelfAttention(nn.Module):
    """
    A Self-Attention layer used in the Decoder. The mask prevents the model
    from "cheating" by looking at future tokens in the sequence.
    """
    def __init__(self, embed_size, heads):
        super(MaskedSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"

        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_size = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)

        return self.fc_out(out)


# ============================
# Cross-Attention Mechanism
# ============================
class CrossAttention(nn.Module):
    """
    A Cross-Attention layer where Query comes from the Decoder and Key/Value
    come from the Encoder. This allows the Decoder to focus on relevant parts
    of the input sequence.
    """
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"

        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, decoder_input, encoder_output):
        batch_size, seq_length_decoder, embed_size = decoder_input.shape
        _, seq_length_encoder, _ = encoder_output.shape

        # Query comes from decoder, Key/Value from encoder
        Q = self.W_q(decoder_input)
        K = self.W_k(encoder_output)
        V = self.W_v(encoder_output)

        Q = Q.view(batch_size, seq_length_decoder, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length_encoder, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length_encoder, self.heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = F.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length_decoder, embed_size)

        return self.fc_out(out)


# ============================
# Encoder Block (with Pre-Layer Normalization)
# ============================
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(EncoderBlock, self).__init__()
        self.self_attn = SelfAttention(embed_size, heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x)
        x = x + self.dropout(attn_out) # Residual connection

        norm_x = self.norm2(x)
        ff_out = self.ff(norm_x)
        x = x + self.dropout(ff_out) # Residual connection
        return x


# ============================
# Decoder Block (with Pre-Layer Normalization)
# ============================
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.masked_self_attn = MaskedSelfAttention(embed_size, heads)
        self.cross_attn = CrossAttention(embed_size, heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask):
        norm_x = self.norm1(x)
        attn_out = self.masked_self_attn(norm_x, mask)
        x = x + self.dropout(attn_out)

        norm_x = self.norm2(x)
        attn_out = self.cross_attn(norm_x, encoder_output)
        x = x + self.dropout(attn_out)

        norm_x = self.norm3(x)
        ff_out = self.ff(norm_x)
        x = x + self.dropout(ff_out)
        return x


# ============================
# Encoder Stack
# ============================
class Encoder(nn.Module):
    def __init__(self, num_layers, embed_size, heads, forward_expansion, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size) # Final norm for the stack

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x) # Final normalization


# ============================
# Decoder Stack
# ============================
class Decoder(nn.Module):
    def __init__(self, num_layers, embed_size, heads, forward_expansion, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, mask):
        for layer in self.layers:
            x = layer(x, encoder_output, mask)
        return x


# ============================
# Full Transformer Model
# ============================
class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, embed_size, heads, forward_expansion, dropout, vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_encoder_layers, embed_size, heads, forward_expansion, dropout)
        self.decoder = Decoder(num_decoder_layers, embed_size, heads, forward_expansion, dropout)
        self.final_linear = nn.Linear(embed_size, vocab_size)

    def generate_mask(self, seq_length):
        # Creates a triangular mask to prevent attention to future tokens.
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        return mask

    def forward(self, source, target):
        # NOTE: This forward pass assumes input tensors are already embedded
        # and have positional encodings added.
        seq_length = target.shape[1]
        mask = self.generate_mask(seq_length).to(target.device)

        encoder_output = self.encoder(source)
        decoder_output = self.decoder(target, encoder_output, mask)
        output = self.final_linear(decoder_output)

        return output


# ============================
# Example Usage
# ============================
if __name__ == '__main__':
    # Hyperparameters
    vocab_size = 10000
    embed_size = 512
    num_encoder_layers = 6
    num_decoder_layers = 6
    heads = 8
    forward_expansion = 4
    dropout = 0.1
    batch_size = 32
    source_seq_length = 100
    target_seq_length = 120

    # Instantiate the model
    model = Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        embed_size=embed_size,
        heads=heads,
        forward_expansion=forward_expansion,
        dropout=dropout,
        vocab_size=vocab_size
    )

    # Create dummy input tensors (representing embedded + positional encoded data)
    source_tensor = torch.rand(batch_size, source_seq_length, embed_size)
    target_tensor = torch.rand(batch_size, target_seq_length, embed_size)

    # Get model output
    output = model(source_tensor, target_tensor)

    print("Model instantiated successfully!")
    print(f"Source tensor shape: {source_tensor.shape}")
    print(f"Target tensor shape: {target_tensor.shape}")
    print(f"Model output shape: {output.shape}")
    assert output.shape == (batch_size, target_seq_length, vocab_size)
    print("Output shape is correct.")
