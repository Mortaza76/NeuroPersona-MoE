# NeuroPersona-MoE

A production-style **Mixture-of-Experts (MoE) persona router** for conversational AI, combining:

- A **local LLM** (Qwen2.5-0.5B-Instruct) for persona-specific generation  
- A **neural MoE router** (trained on sentence embeddings)  
- **Rule-based, emotion-aware, and sentiment-aware routing logic**  
- **Per-persona memory, summarization, and drift**  
- A **Streamlit dashboard** with rich visualizations of routing, gating, and memory

This repository is designed as a portfolio-quality project that demonstrates real-world AI system design, not just a toy chatbot.

---

## Key Features

- **Mixture-of-Experts routing**
  - Router built on top of `all-MiniLM-L6-v2` sentence embeddings
  - Custom `MoEClassifier` with 8 experts and softmax-based gating
  - Outputs both **persona probabilities** and **expert gate weights**

- **Rich persona set (11 experts)**
  - `logical` – structured, analytical reasoning  
  - `emotional` – supportive, empathetic, feelings-focused  
  - `comedic` – joking, playful, humorous  
  - `angry` – blunt, frustrated (with safety constraints)  
  - `philosophical` – abstract, existential, reflective  
  - `sarcastic` – dry, mocking, eye-rolling tone  
  - `poetic` – lyrical, metaphor-heavy language  
  - `formal` – professional, corporate tone  
  - `gen_z` – modern slang, casual, meme-ish  
  - `storytelling` – narrative-driven, scene-building responses  
  - `ambiguous` – vague, introspective, non-committal

- **Local LLM experts (no OpenAI dependency)**
  - Uses a local **Qwen2.5-0.5B-Instruct** model via `transformers`
  - Each persona has its own **system prompt** and style constraints
  - Generation uses:
    - persona-specific history
    - persona **intensity** control
    - output clean-up (no AI-meta, full sentences only)

- **Advanced routing logic**
  - Neural MoE classifier (logits + gate weights)
  - Rule-based overrides for:
    - **finance/planning topics → `logical`**
    - **strong negative sentiment → `emotional` or constrained `angry`**
    - **serious topics → avoid `comedic` / `sarcastic`**
  - Emotion-aware routing based on keyword groups (sadness, humor, existential themes)
  - Sentiment-aware routing via **VADER** (if installed), with heuristic fallback
  - Confidence thresholds to prevent low-confidence misrouting

- **Memory and persona dynamics**
  - Per-persona **conversation memory**
  - Memory **decay** (different limits per persona)
  - Automatic **summarization** when memory grows too long
  - **Persona intensity drift**:
    - active persona’s intensity increases slightly per turn
    - inactive personas decay back toward neutral

- **Visualization UI (Streamlit)**
  - Chat interface with persona tags and emoji avatars
  - Real-time charts:
    - Persona probability bar chart
    - Gate weight bar chart
    - Routing network bubble graph
    - Gate-weight waterfall (delta changes)
    - Persona intensity drift line chart
    - Persona intensity bar-race animation
  - Memory inspector:
    - Per-persona expandable message history
    - Raw memory JSON for debugging

---

## Project Architecture

High-level flow:

1. **User message** enters via CLI (`chat.py`) or Streamlit (`ui_streamlit.py`).
2. Message is passed to the **router** (`infer_router.py`):
   - Embedded using `all-MiniLM-L6-v2`
   - Fed into `MoEClassifier` → logits + gate weights
   - Softmax on logits → persona probabilities
   - Post-processing:
     - emotion-based overrides
     - sentiment-based overrides
     - topic-based overrides (e.g., finance)
     - safety & fallback rules
3. Router returns:
   - `persona`
   - `confidence`
   - `all_confidences` (for all personas)
   - `gate_weights` (expert gating vector)
4. The selected persona expert is called in `experts/persona_responses.py`:
   - Loads **Qwen2.5-0.5B-Instruct**
   - Builds a persona-specific prompt:
     - persona system instructions
     - history for that persona (when available)
     - persona intensity factor
   - Generates the response with carefully constrained decoding
5. Memory system:
   - Per-persona memory list updated with `{ "user": ..., "assistant": ... }`
   - Memory decay and summarization are applied
   - Memory persisted across sessions in `persona_memory.json`
6. In the Streamlit UI:
   - Chat messages displayed with persona tags and avatars
   - Router outputs and memory visualized in multiple charts

---

## File Structure

A typical structure for this project:

```bash
NeuroPersona-MoE/
├── chat.py                    # CLI chatbot entrypoint
├── ui_streamlit.py            # Streamlit dashboard for chat + analytics
├── infer_router.py            # Router: MoE classifier + routing logic
├── train_router.py            # (Optional) Training script for the router
├── moe_router.pt              # Trained MoE router weights
├── persona_memory.json        # Persisted persona memory (auto-created)
├── experts/
│   ├── __init__.py
│   └── persona_responses.py   # Persona prompts + LLM generation logic
├── model/
│   ├── __init__.py
│   └── moe_classifier.py      # MoE classifier implementation (PyTorch)
├── data/
│   └── moe_persona_dataset.jsonl   # Training dataset for router (text + label)
├── requirements.txt           # Python dependencies (recommended)
└── README.md

Note: Some file names/paths may vary slightly depending on your local setup, but this reflects the intended structure.

⸻

Installation

1. Clone the repository

git clone https://github.com/Mortaza76/NeuroPersona-MoE.git
cd NeuroPersona-MoE

2. Create and activate a virtual environment

python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# On Windows: venv\Scripts\activate

3. Install dependencies

Create a requirements.txt (if not already present) similar to:

torch
transformers
sentence-transformers
streamlit
pandas
altair
vaderSentiment

Then:

pip install -r requirements.txt

PyTorch can also be installed separately depending on your platform (CPU-only vs GPU).

4. Model checkpoints

The following are downloaded automatically via transformers on first run:
	•	sentence-transformers/all-MiniLM-L6-v2
	•	Qwen/Qwen2.5-0.5B-Instruct (or equivalent small Qwen model)

Ensure you have enough disk and a stable internet connection for initial model downloads.

⸻

Running the System

Option 1: Command Line Chat (CLI)

source venv/bin/activate
python chat.py

You’ll see something like:

Mixture-of-Experts Chatbot Ready!

> I feel overwhelmed about my future
[EMOTIONAL MODE ACTIVATED]


	•	The router picks a persona, and the persona-specific LLM responds.
	•	Memory is updated per persona.
	•	A memory inspector section prints to the terminal for debugging.

⸻

Option 2: Streamlit Dashboard

source venv/bin/activate
streamlit run ui_streamlit.py

Then open:
	•	Local: http://localhost:8501

The UI includes:
	•	Left column:
	•	Chat interface using st.chat_message
	•	Persona indicator (e.g., [EMOTIONAL MODE]) and emoji avatar
	•	Right column:
	•	Persona probability chart (bar chart)
	•	Gate weight chart (bar chart)
	•	Routing network graph (bubble chart with persona-specific colors)
	•	Gate weight waterfall chart (delta vs previous turn)
	•	Persona drift over time:
	•	Smooth multi-line chart with one line per persona
	•	Bar-race visualization showing current intensity ranking
	•	Memory inspector:
	•	Persona selector
	•	Expandable message threads
	•	Raw JSON view of current persona memory

⸻

Router Details (infer_router.py)

Base Model
	•	Uses sentence-transformers/all-MiniLM-L6-v2 to obtain 384-d embeddings.
	•	Embeddings are fed into MoEClassifier with:
	•	num_experts = 8
	•	num_labels = 11 personas

Output

The classifier outputs:
	•	logits: shape (batch_size, num_labels)
	•	gates: shape (batch_size, num_experts)

Routing uses:

probs = softmax(logits)
persona_index = probs.argmax()
persona = LABEL_MAP[persona_index]

Post-routing logic

After initial neural routing, several rules are applied:
	1.	Sentiment-aware routing
	•	Uses VADER (if available) via SentimentIntensityAnalyzer
	•	If compound <= -0.35 → negative
	•	If compound >= 0.35 → positive
	•	Else neutral
	•	If sentiment is negative and persona is not in ["angry", "emotional", "sarcastic"], route to emotional.
	2.	Emotion keyword overrides
	•	If EMOTION_KEYWORDS detect a strong hit:
	•	sadness/overwhelm → emotional
	•	humor slang (lol, lmao, haha) → comedic
	•	existential language (meaning, existence, why do we) → philosophical
	3.	Low-confidence fallback
	•	If max probability < 0.35, fallback to logical.
	4.	Topic-aware logic
	•	Finance-related keywords:
	•	"car", "loan", "finance", "budget", "salary", "income", "saving", "afford", ...
	•	Forces persona to logical.
	5.	Angry persona protection
	•	angry only allowed if is_negative(text) returns True (using sentiment + negative word list).
	•	Otherwise, fallback to second-best persona.
	6.	Serious-topic protection
	•	If comedic or sarcastic chosen but text contains words like:
	•	plan, career, future, goal, improve, how do I
	•	Persona overridden to logical.
	7.	Weird outlier protection
	•	If persona is in ["sarcastic", "comedic", "gen_z"] and max_prob < 0.50, fallback to second-best.

The final returned object:

{
    "persona": persona,
    "confidence": float(all_probs[persona]),
    "all_confidences": all_probs,
    "gate_weights": gates.cpu().numpy()[0].tolist()
}


⸻

Persona Experts (experts/persona_responses.py)

Each expert function:

def logical(text, history=None, intensity: float = 1.0):
    return generate_with_llm("logical", text, history=history, intensity=intensity)

All funnel into:

def generate_with_llm(persona, text, history=None, intensity: float = 1.0):
    system_prompt = PERSONA_PROMPTS[persona]
    # Build history string
    # Clamp intensity (0.5–1.6)
    # Construct prompt:
    #   - System persona instructions
    #   - Persona style rules
    #   - History snippet
    #   - User input
    # Call Qwen2.5-0.5B-Instruct via transformers
    # Decode, strip to last full sentence
    return response_text

Persona prompts explicitly enforce:
	•	Tone
	•	Style
	•	Number of sentences
	•	No breaking character
	•	No “as an AI…” meta-language

Temperature is dynamically adjusted by persona intensity.

⸻

Memory System
	•	Memory is stored as:

persona_memory = {
    "logical": [ { "user": "...", "assistant": "..." }, ... ],
    "emotional": [...],

}

	•	Different personas use different decay_limit values:
	•	sarcastic, comedic: very short memory
	•	angry, gen_z: medium
	•	logical, formal: longer
	•	emotional, poetic, storytelling, philosophical: longest
	•	When memory grows too long:
	•	A summarization prompt is built and sent back through the relevant persona expert.
	•	Memory is replaced with a compressed summary turn.
	•	Memory is persisted to disk (persona_memory.json) so conversations continue across sessions.

⸻

Training the Router (Optional)

If you want to retrain or fine-tune the persona router:
	1.	Prepare a JSONL dataset like:

{"text": "Explain how to optimize this workflow.", "label": "logical"}
{"text": "I'm feeling overwhelmed lately.", "label": "emotional"}
{"text": "Tell me something silly.", "label": "comedic"}
{"text": "This app is so slow it's infuriating.", "label": "angry"}
{"text": "What is the meaning of creativity?", "label": "philosophical"}


	2.	Use train_router.py (assumed to exist in the repo) to:
	•	Load the dataset
	•	Encode text via all-MiniLM-L6-v2
	•	Train the MoEClassifier to predict persona labels
	•	Save weights to moe_router.pt
	3.	Update/replace moe_router.pt in the root folder.

⸻

Extending the System

Add a new persona
	1.	Add the persona name to PERSONAS in infer_router.py.
	2.	Update LABEL_MAP accordingly.
	3.	Add a new prompt and function in persona_responses.py.
	4.	Update any routing rules, if needed.
	5.	Retrain the router (recommended) with examples for the new persona.

Swap the local LLM
	•	Replace model name in persona_responses.py (Qwen2.5-0.5B-Instruct) with:
	•	Llama-3.2-1B-Instruct
	•	Phi-3-mini
	•	or any other small instruction-tuned model.

Plug into other UIs
	•	The system is modular — you can:
	•	Wrap the router and persona engine in a FastAPI service
	•	Connect it to a web frontend or Discord bot
	•	Use the same routing + memory logic

⸻

Notes
	•	This project is designed as a demonstration of system architecture and engineering:
	•	MoE classifier design
	•	Hybrid heuristic + neural routing
	•	Local LLM persona specialization
	•	Memory, summarization, drift dynamics
	•	Visualization for interpretability
	•	It is not a productized SaaS, but it is architected in a way that could be extended into one.

⸻

License

MIT License

⸻

Acknowledgements
	•	Sentence Transformers￼ for all-MiniLM-L6-v2
	•	Qwen￼ for lightweight instruction LLMs
	•	VADER Sentiment￼ for sentiment analysis
	•	Streamlit￼ and Altair￼ for interactive analytics

This project is an end-to-end example of how to combine routing, local LLMs, memory, and visual analytics into a coherent, explainable AI system.




