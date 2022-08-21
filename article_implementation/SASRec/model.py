import torch 
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3,1,1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, 3,1,1)
        )

    def forward(self, input):
        return input + self.model(input)


class SASRec(nn.Module):
    def __init__(self, num_blocks, num_embeddings, embedding_size, num_heads, sentence_len):
        super().__init__()

        self.num_blocks = num_blocks

        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(embedding_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(embedding_size,
                                                            num_heads,
                                                            0.1, batch_first=True)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(embedding_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = ConvNet(sentence_len, sentence_len)
            self.forward_layers.append(new_fwd_layer)

        self.last = nn.Linear(embedding_size, num_embeddings)
        self.softmax = nn.Softmax(2)

    def forward(self, input):
        hidden = self.embeddings(input)
        tl = hidden.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=device))

        for i in range(self.num_blocks):
            Q = self.attention_layernorms[i](hidden)

            output, _ = self.attention_layers[i](Q, hidden, hidden, attn_mask=attention_mask)
            
            hidden = Q + output
            hidden = self.forward_layernorms[i](hidden)
            hidden = self.forward_layers[i](hidden)
        return self.softmax(self.last(hidden))
        