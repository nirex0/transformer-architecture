import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================
# Self-Attention Mechanism
# ============================
class SelfAttention(nn.Module):
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

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = F.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)

        return self.fc_out(out)


# ============================
# Masked Multi-Head Self-Attention
# ============================
class MaskedSelfAttention(nn.Module):
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
# Encoder Block (Pre-Norm)
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
        # Pre-Normalization as per the diagram
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x)
        x = x + self.dropout(attn_out) # Residual connection

        norm_x = self.norm2(x)
        ff_out = self.ff(norm_x)
        x = x + self.dropout(ff_out) # Residual connection
        return x


# ============================
# Decoder Block (Pre-Norm)
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
        # Masked Self-Attention (Pre-Norm)
        norm_x = self.norm1(x)
        attn_out = self.masked_self_attn(norm_x, mask)
        x = x + self.dropout(attn_out)

        # Cross-Attention (Pre-Norm)
        norm_x = self.norm2(x)
        attn_out = self.cross_attn(norm_x, encoder_output)
        x = x + self.dropout(attn_out)

        # Feed-Forward (Pre-Norm)
        norm_x = self.norm3(x)
        ff_out = self.ff(norm_x)
        x = x + self.dropout(ff_out)
        return x


# ============================
# Encoder
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
# Decoder
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
# Positional Encoding
# ============================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, embed_size) to hold the encodings
        pe = torch.zeros(max_len, embed_size)
        
        # Create a tensor for the positions (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term for the sine and cosine functions
        # This is 1 / (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        # Apply sine to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension so it can be added to the input embeddings
        # Shape becomes (1, max_len, embed_size)
        pe = pe.unsqueeze(0)
        
        # Register 'pe' as a buffer. This makes it part of the model's state,
        # but not a parameter that is trained. It will be saved with the model's state_dict.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape (batch_size, seq_length, embed_size)
        # Add the positional encoding to the input embeddings
        # self.pe[:, :x.size(1)] selects the encodings up to the max sequence length in the batch
        x = x + self.pe[:, :x.size(1), :]
        return x


# ============================
# Transformer Model
# ============================
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_encoder_layers, num_decoder_layers, heads, forward_expansion, dropout, max_len=5000):
        super(Transformer, self).__init__()
        
        # Token Embedding and Positional Encoding
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        
        self.encoder = Encoder(num_encoder_layers, embed_size, heads, forward_expansion, dropout)
        self.decoder = Decoder(num_decoder_layers, embed_size, heads, forward_expansion, dropout)
        self.final_linear = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, seq_length):
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        return mask

    def forward(self, source_tokens, target_tokens):
        # source_tokens and target_tokens are now token IDs, not embeddings
        # Shape: (batch_size, seq_length)
        
        # 1. Apply token embeddings and positional encodings
        source_embedded = self.dropout(self.positional_encoding(self.token_embedding(source_tokens)))
        target_embedded = self.dropout(self.positional_encoding(self.token_embedding(target_tokens)))

        # 2. Generate mask for the decoder
        target_seq_length = target_tokens.shape[1]
        target_mask = self.generate_mask(target_seq_length).to(target_tokens.device)

        # 3. Pass through Encoder and Decoder
        encoder_output = self.encoder(source_embedded)
        decoder_output = self.decoder(target_embedded, encoder_output, target_mask)
        output = self.final_linear(decoder_output)

        return output


# ============================
# Example Main Function (Updated)
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

    # Instantiate the complete model
    model = Transformer(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        heads=heads,
        forward_expansion=forward_expansion,
        dropout=dropout
    )

    # Create dummy input tensors of token IDs (long integers)
    source_tokens = torch.randint(0, vocab_size, (batch_size, source_seq_length))
    target_tokens = torch.randint(0, vocab_size, (batch_size, target_seq_length))

    # Get model output
    output = model(source_tokens, target_tokens)

    print("Model with Positional Encoding instantiated successfully!")
    print(f"Source token IDs shape: {source_tokens.shape}")
    print(f"Target token IDs shape: {target_tokens.shape}")
    print(f"Model output shape (logits): {output.shape}")
    assert output.shape == (batch_size, target_seq_length, vocab_size)
    print("Output shape is correct.")
