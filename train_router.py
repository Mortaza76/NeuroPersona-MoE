import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from utils.dataset import PersonaDataset
from model.moe_classifier import MoEClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PERSONAS = [
    "logical",
    "emotional",
    "comedic",
    "angry",
    "philosophical",
    "sarcastic",
    "poetic",
    "formal",
    "gen_z",
    "storytelling",
    "ambiguous"
]

LABEL_MAP = {p: i for i, p in enumerate(PERSONAS)}

def train():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    base_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    dataset = PersonaDataset("moe_persona_dataset.jsonl", tokenizer, LABEL_MAP)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MoEClassifier(
        base_model=base_model,
        hidden_size=384,
        num_experts=8,
        num_labels=len(PERSONAS)
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        total_loss = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits, gate_w = model(input_ids, mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss = {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "moe_router.pt")
    print("Training complete ✔")

if __name__ == "__main__":
    train()