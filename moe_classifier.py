import torch
import torch.nn as nn
from model.router import GatingNetwork
from model.experts import Expert

class MoEClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_experts, num_labels):
        super().__init__()

        self.encoder = base_model
        self.router = GatingNetwork(hidden_size, num_experts)

        self.experts = nn.ModuleList([
            Expert(hidden_size) for _ in range(num_experts)
        ])

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token embedding
        h = out.last_hidden_state[:, 0, :]

        gate_w = self.router(h)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(h))

        expert_outputs = torch.stack(expert_outputs, dim=1)

        mixture = torch.sum(
            gate_w.unsqueeze(2) * expert_outputs,
            dim=1
        )

        logits = self.classifier(mixture)
        return logits, gate_w