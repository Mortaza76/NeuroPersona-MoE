import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, h):
        logits = self.gate(h)
        gate_weights = torch.softmax(logits, dim=-1)
        return gate_weights