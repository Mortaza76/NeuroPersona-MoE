import json
import os
import random
from infer_router import predict_persona
from experts.persona_responses import EXPERT_MAP

MEMORY_PATH = "persona_memory.json"

def chat():
    print("Mixture-of-Experts Chatbot Ready!\n")
    # Load persona memory from disk if available
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r") as f:
                persona_memory = json.load(f)
        except Exception:
            persona_memory = {name: [] for name in EXPERT_MAP.keys()}
    else:
        persona_memory = {name: [] for name in EXPERT_MAP.keys()}

    # Ensure all personas exist as keys
    for name in EXPERT_MAP.keys():
        persona_memory.setdefault(name, [])

    # Persona intensity map (for drift + style strength)
    persona_intensity = {name: 1.0 for name in EXPERT_MAP.keys()}

    while True:
        text = input("> ")

        if text.lower() in ["quit", "exit"]:
            break

        result = predict_persona(text)
        persona = result["persona"]
        memory = persona_memory[persona]

        intensity = persona_intensity.get(persona, 1.0)

        # Memory decay rules (different personas forget differently)
        if persona in ["sarcastic", "comedic"]:
            decay_limit = 3
        elif persona in ["angry", "gen_z"]:
            decay_limit = 5
        elif persona in ["logical", "formal"]:
            decay_limit = 7
        elif persona in ["emotional", "poetic", "storytelling", "philosophical"]:
            decay_limit = 12
        else:
            decay_limit = 6

        if len(memory) > decay_limit:
            persona_memory[persona] = memory[-decay_limit:]
            memory = persona_memory[persona]

        # Summarize memory if it grows too large
        if len(memory) >= 10:
            summary_prompt = (
                f"Summarize this conversation in the style of the '{persona}' persona. "
                f"Keep it concise but preserve emotional and contextual meaning:\n\n"
            )
            for turn in memory[-8:]:
                summary_prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

            # Call expert model to summarize
            summary = EXPERT_MAP[persona](summary_prompt, history=None, intensity=intensity)

            persona_memory[persona] = [{"user": "SUMMARY", "assistant": summary}]
            memory = persona_memory[persona]

        response = EXPERT_MAP[persona](text, history=memory, intensity=intensity)

        # Update persona-specific memory
        persona_memory[persona].append({
            "user": text,
            "assistant": response
        })

        # Trim memory to last 10 exchanges
        if len(persona_memory[persona]) > 10:
            persona_memory[persona] = persona_memory[persona][-10:]
            memory = persona_memory[persona]

            # Persona drift: active persona intensity slowly increases, others decay toward 1.0
            persona_intensity[persona] = min(1.6, persona_intensity.get(persona, 1.0) + 0.05)
            for name in persona_intensity.keys():
                if name != persona:
                    persona_intensity[name] += (1.0 - persona_intensity[name]) * 0.05

            # Persist memory to disk
            try:
                with open(MEMORY_PATH, "w") as f:
                    json.dump(persona_memory, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        print(f"\n[{persona.upper()} MODE ACTIVATED]")

        print("\n[MEMORY INSPECTOR]")
        for turn in memory:
            print(f"- U: {turn['user']}")
            print(f"  A: {turn['assistant']}\n")

        print(response)
        print("\n---\n")

if __name__ == "__main__":
    chat()