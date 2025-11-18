import torch
from transformers import AutoTokenizer, AutoModel
from model.moe_classifier import MoEClassifier

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENT_ANALYZER = SentimentIntensityAnalyzer()
except ImportError:
    SENT_ANALYZER = None

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
LABEL_MAP = {i: p for i, p in enumerate(PERSONAS)}

def load_router():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    base_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    model = MoEClassifier(
        base_model=base_model,
        hidden_size=384,
        num_experts=8,
        num_labels=len(PERSONAS)
    )
    model.load_state_dict(torch.load("moe_router.pt", map_location=DEVICE))
    model.eval().to(DEVICE)

    return tokenizer, model

import re

FINANCE_KEYWORDS = [
    "car", "loan", "finance", "financial", "budget",
    "salary", "income", "saving", "savings", "afford",
    "expensive", "payment", "rent", "mortgage", "tax"
]

NEGATIVE_WORDS = [
    "fuck", "stupid", "idiot", "annoying", "angry",
    "hate", "mad", "pissed", "trash", "dumb", "useless"
]

def get_sentiment(text: str) -> str:
    """
    Return 'negative', 'neutral', or 'positive' using VADER if available,
    otherwise a simple keyword-based heuristic.
    """
    if SENT_ANALYZER is not None:
        scores = SENT_ANALYZER.polarity_scores(text)
        comp = scores.get("compound", 0.0)
        if comp <= -0.35:
            return "negative"
        elif comp >= 0.35:
            return "positive"
        else:
            return "neutral"

    # Fallback: simple heuristic if VADER is not installed
    t = text.lower()
    neg_hits = sum(1 for w in NEGATIVE_WORDS if w in t)
    if neg_hits >= 1:
        return "negative"
    return "neutral"

EMOTION_KEYWORDS = {
    "emotional": [
        "sad", "depressed", "hurt", "upset", "cry", "crying",
        "heartbroken", "overwhelmed", "anxious", "anxiety",
        "lonely", "alone", "stressed", "stress", "panic"
    ],
    "comedic": [
        "lol", "lmao", "haha", "funny", "joke", "joking",
        "meme", "goofy", "silly"
    ],
    "philosophical": [
        "meaning", "existence", "purpose", "why do we",
        "universe", "consciousness", "identity", "life"
    ]
}

def detect_emotion_persona(text):
    t = text.lower()
    scores = {}
    for persona, words in EMOTION_KEYWORDS.items():
        scores[persona] = sum(1 for w in words if w in t)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

def contains_finance_terms(text):
    t = text.lower()
    return any(word in t for word in FINANCE_KEYWORDS)

def is_negative(text):
    # Primary source: sentiment model if available
    if get_sentiment(text) == "negative":
        return True
    # Fallback: keyword-based negativity
    t = text.lower()
    return any(word in t for word in NEGATIVE_WORDS)

def is_serious_topic(text):
    t = text.lower()
    return any(word in t for word in ["plan", "career", "future", "goal", "improve", "how do i"])

def predict_persona(text):
    tokenizer, model = load_router()

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits, gates = model(input_ids, mask)

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    persona_index = probs.argmax()
    persona = LABEL_MAP[persona_index]

    max_prob = float(probs[persona_index])
    all_probs = {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}

    # Get second-best persona
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    top1, top2 = sorted_probs[0], sorted_probs[1]
    second_best = top2[0]

    # Sentiment analysis (shared for later rules)
    sentiment = get_sentiment(text)

    # --- Emotion-aware routing override ---
    emotion_hit = detect_emotion_persona(text)
    if emotion_hit:
        persona = emotion_hit

    # --- Sentiment-aware override: strong negative but no clear persona ---
    if sentiment == "negative" and persona not in ["angry", "emotional", "sarcastic"]:
        # Default to emotional for general negative tone
        persona = "emotional"

    # --- 1. Low-confidence fallback ---
    if max_prob < 0.35:
        persona = "logical"

    # --- 2. Finance/planning topics should NOT trigger angry/comedic ---
    if contains_finance_terms(text):
        persona = "logical"

    # --- 3. Angry persona protection ---
    if persona == "angry" and not is_negative(text):
        persona = second_best

    # --- 4. Overriding comedic/sarcastic if message is serious ---
    if persona in ["comedic", "sarcastic"] and is_serious_topic(text):
        persona = "logical"

    # --- 5. If model picks a weird outlier, fallback to 2nd best ---
    if persona in ["sarcastic", "comedic", "gen_z"] and max_prob < 0.50:
        persona = second_best

    return {
        "persona": persona,
        "confidence": float(all_probs[persona]),
        "all_confidences": all_probs,
        "gate_weights": gates.cpu().numpy()[0].tolist()
    }

if __name__ == "__main__":
    print(predict_persona("why is this so annoying today bro"))