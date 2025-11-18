import os
import json
import random
import streamlit as st

from infer_router import predict_persona
from experts.persona_responses import EXPERT_MAP

MEMORY_PATH = "persona_memory.json"

PERSONA_AVATARS = {
    "logical": "🧠",
    "emotional": "💜",
    "comedic": "🤡",
    "angry": "😡",
    "philosophical": "🧘",
    "sarcastic": "🙄",
    "poetic": "📝",
    "formal": "🕴️",
    "gen_z": "⚡",
    "storytelling": "📖",
    "ambiguous": "🌫️",
}

PERSONAS = list(EXPERT_MAP.keys())


def load_memory():
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r") as f:
                data = json.load(f)
                for name in PERSONAS:
                    data.setdefault(name, [])
                return data
        except Exception:
            pass
    return {name: [] for name in PERSONAS}


def save_memory(memory):
    try:
        with open(MEMORY_PATH, "w") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def init_session_state():
    if "persona_memory" not in st.session_state:
        st.session_state["persona_memory"] = load_memory()
    if "chat_log" not in st.session_state:
        st.session_state["chat_log"] = []
    if "persona_intensity" not in st.session_state:
        st.session_state["persona_intensity"] = {name: 1.0 for name in PERSONAS}
    if "last_routing" not in st.session_state:
        st.session_state["last_routing"] = None


def apply_decay_and_summarization(persona, memory, intensity):
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
        memory = memory[-decay_limit:]

    if len(memory) >= 10:
        summary_prompt = (
            f"Summarize this conversation in the style of '{persona}'. "
            "Keep it concise but preserve emotional meaning:\n\n"
        )
        for turn in memory[-8:]:
            summary_prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        summary = EXPERT_MAP[persona](summary_prompt, history=None, intensity=intensity)
        memory = [{"user": "SUMMARY", "assistant": summary}]

    if len(memory) > 10:
        memory = memory[-10:]

    return memory


def main():
    st.set_page_config(page_title="MoE Persona Chatbot", layout="wide")
    init_session_state()

    st.title("🧬 Mixture-of-Experts Persona Chatbot")

    col_chat, col_info = st.columns([2, 1])

    with col_chat:
        st.subheader("Chat")

        for msg in st.session_state["chat_log"]:
            role = msg["role"]
            persona = msg.get("persona")
            avatar = PERSONA_AVATARS.get(persona, "") if persona else ""
            if role == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar=avatar):
                    if persona:
                        st.markdown(f"**[{persona.upper()} MODE]**")
                    st.write(msg["content"])

        user_text = st.chat_input("Type here...")

        if user_text:
            st.session_state["chat_log"].append(
                {"role": "user", "content": user_text}
            )

            routing = predict_persona(user_text)
            all_conf = routing["all_confidences"]
            sorted_personas = sorted(all_conf.items(), key=lambda x: x[1], reverse=True)
            top2 = sorted_personas[:2]
            total = top2[0][1] + top2[1][1]

            if random.random() < top2[0][1] / total:
                persona = top2[0][0]
            else:
                persona = top2[1][0]

            persona_memory = st.session_state["persona_memory"]
            memory = persona_memory.get(persona, [])
            intensity = st.session_state["persona_intensity"].get(persona, 1.0)

            memory = apply_decay_and_summarization(persona, memory, intensity)

            response = EXPERT_MAP[persona](
                user_text, history=memory, intensity=intensity
            )

            memory.append({"user": user_text, "assistant": response})
            persona_memory[persona] = memory
            st.session_state["persona_memory"] = persona_memory
            save_memory(persona_memory)

            st.session_state["persona_intensity"][persona] = min(
                1.6, st.session_state["persona_intensity"][persona] + 0.05
            )
            for name in PERSONAS:
                if name != persona:
                    val = st.session_state["persona_intensity"][name]
                    st.session_state["persona_intensity"][name] = val + (1.0 - val) * 0.05

            st.session_state["chat_log"].append(
                {
                    "role": "assistant",
                    "content": response,
                    "persona": persona,
                }
            )

            st.session_state["last_routing"] = routing
            st.rerun()

    with col_info:
        st.subheader("Routing & Memory Inspector")

        routing = st.session_state.get("last_routing")
        if routing:
            st.markdown("**Persona probabilities:**")
            probs = routing["all_confidences"]
            st.bar_chart(
                {"persona": list(probs.keys()), "prob": list(probs.values())},
                x="persona",
                y="prob",
            )

            st.markdown("**Gate weights:**")
            gates = routing["gate_weights"]
            st.bar_chart(
                {"expert": list(range(len(gates))), "weight": gates},
                x="expert",
                y="weight",
            )

        st.markdown("---")
        st.markdown("**Inspect persona memory:**")
        selected = st.selectbox("Persona:", PERSONAS)
        mem = st.session_state["persona_memory"].get(selected, [])
        if not mem:
            st.write("_No memory yet._")
        else:
            for turn in mem:
                st.markdown(f"- **U:** {turn['user']}")
                st.markdown(f"  **A:** {turn['assistant']}")

        st.markdown("---")
        st.markdown("**Persona Intensities:**")
        for name in PERSONAS:
            st.write(
                f"{PERSONA_AVATARS.get(name, '')} {name}: "
                f"{st.session_state['persona_intensity'][name]:.2f}"
            )


if __name__ == "__main__":
    main()