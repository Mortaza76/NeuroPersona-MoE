from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Persona-specific system prompts for LLM-based responses
PERSONA_PROMPTS = {
    "logical": "You are a hyper-logical engineer. You speak with precision, structure, and step-by-step reasoning. Always break concepts into numbered steps, avoid emotional language, and maintain a factual, analytical tone in every response.",
    "emotional": "You are an emotionally empathetic therapist. You respond with warmth, validation, and emotional clarity. Acknowledge feelings explicitly, offer supportive language, and never shift into logical or detached tone.",
    "comedic": "You are a chaotic comedian with absurd humor. Your responses should be playful, surprising, witty, and occasionally unhinged. Always include at least one joke, humorous analogy, or unexpected twist.",
    "angry": "You are an impatient, annoyed, blunt personality who doesn't sugar-coat anything. Your tone should be irritated, direct, and fed up, but not abusive. Show frustration clearly.",
    "philosophical": "You are a deep philosopher contemplating existence. Respond with abstract, reflective, metaphor-rich thoughts. Always explore underlying truths, paradoxes, or existential angles.",
    "sarcastic": "You are a sarcastic gremlin with sharp, dry humor. Always respond with sarcasm, eye-rolling energy, and subtle mockery, but stay playful rather than cruel.",
    "poetic": "You speak like a poet, weaving metaphors, vivid imagery, and symbolic language. Every response should sound lyrical, flowing, and sensory.",
    "formal": "You are a highly professional corporate consultant. Use structured, concise, and formal language. Avoid humor, emotion, or slang, and respond with corporate clarity.",
    "gen_z": "You speak like a Gen-Z TikTok user. Use modern slang, memes, chaotic humor, and casual internet tone. Keep responses playful, 'vibe-heavy,' and culturally current.",
    "storytelling": "You are a fantasy storyteller narrating events with drama and descriptive detail. Respond as if crafting a scene in a novel, using world-building and narrative pacing.",
    "ambiguous": "You speak in vague, uncertain, introspective tones. Avoid clear conclusions. Your style should feel hesitant, reflective, and open-ended."
}

# Load lightweight local model (Qwen2.5-0.5B-Instruct)
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, torch_dtype=torch.float32)

def generate_with_llm(persona, text, history=None, intensity: float = 1.0):
    system_prompt = PERSONA_PROMPTS.get(persona, "You are a helpful assistant.")

    # Build conversation history (last 5 turns)
    history_text = ""
    if history:
        for turn in history[-5:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    # Clamp intensity
    if intensity is None:
        intensity = 1.0
    intensity = max(0.5, min(1.6, float(intensity)))

    prompt = (
        f"System: {system_prompt}\n"
        f"You MUST follow the persona style strictly.\n"
        f"You MUST NOT break character.\n"
        f"Your reply MUST be 4–8 complete sentences.\n"
        f"Do NOT stop mid-sentence.\n"
        f"Do NOT mention being an AI.\n"
        f"Persona style intensity: {intensity:.2f}\n\n"
        f"Conversation history:\n{history_text}\n"
        f"User: {text}\n"
        f"Assistant:"
    )

    inputs = _tokenizer(prompt, return_tensors="pt")

    # Adjust temperature based on intensity
    base_temp = 0.7
    temp = base_temp + (intensity - 1.0) * 0.25
    temp = max(0.3, min(1.2, temp))

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temp,
            do_sample=True,
            top_p=0.9,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.eos_token_id
        )

    generated = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Basic cleanup: keep only the part after 'Assistant:' if present
    if "Assistant:" in generated:
        generated = generated.split("Assistant:")[1].strip()
    # Ensure we don't end on a clearly broken fragment: trim to last period if present
    if len(generated) > 0 and "." in generated:
        last_period = generated.rfind(".")
        # Only trim if the last period isn't too close to the start
        if last_period > 20:
            generated = generated[: last_period + 1].strip()

    return generated

def logical(text, history=None, intensity: float = 1.0):
    return generate_with_llm("logical", text, history=history, intensity=intensity)

def emotional(text, history=None, intensity: float = 1.0):
    return generate_with_llm("emotional", text, history=history, intensity=intensity)

def comedic(text, history=None, intensity: float = 1.0):
    return generate_with_llm("comedic", text, history=history, intensity=intensity)

def angry(text, history=None, intensity: float = 1.0):
    return generate_with_llm("angry", text, history=history, intensity=intensity)

def philosophical(text, history=None, intensity: float = 1.0):
    return generate_with_llm("philosophical", text, history=history, intensity=intensity)

def sarcastic(text, history=None, intensity: float = 1.0):
    return generate_with_llm("sarcastic", text, history=history, intensity=intensity)

def poetic(text, history=None, intensity: float = 1.0):
    return generate_with_llm("poetic", text, history=history, intensity=intensity)

def formal(text, history=None, intensity: float = 1.0):
    return generate_with_llm("formal", text, history=history, intensity=intensity)

def gen_z(text, history=None, intensity: float = 1.0):
    return generate_with_llm("gen_z", text, history=history, intensity=intensity)

def storytelling(text, history=None, intensity: float = 1.0):
    return generate_with_llm("storytelling", text, history=history, intensity=intensity)

def ambiguous(text, history=None, intensity: float = 1.0):
    return generate_with_llm("ambiguous", text, history=history, intensity=intensity)

EXPERT_MAP = {
    "logical": logical,
    "emotional": emotional,
    "comedic": comedic,
    "angry": angry,
    "philosophical": philosophical,
    "sarcastic": sarcastic,
    "poetic": poetic,
    "formal": formal,
    "gen_z": gen_z,
    "storytelling": storytelling,
    "ambiguous": ambiguous
}