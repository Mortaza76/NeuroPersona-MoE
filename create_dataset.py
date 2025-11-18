import json
import random

PERSONAS = {
    "logical": [
        "Break this issue into clear steps so I can evaluate each one.",
        "Outline the most efficient way to approach this problem from start to finish.",
        "Identify the variables involved and explain how they interact.",
        "I need a concise, structured explanation that avoids unnecessary details.",
        "Analyze this situation as if you're debugging a system.",
        "Present the pros and cons in a neutral, factual format.",
        "Explain the logic behind this decision so I can verify it.",
        "Help me reason through this using first principles.",
        "Compare the possible outcomes using rational criteria.",
        "Evaluate this claim using evidence instead of assumptions.",
        "Clarify the objective before suggesting the optimal path.",
        "Highlight any contradictions or inconsistencies in this argument.",
        "Summarize this complex topic in a clean, step-by-step breakdown."
    ],

    "emotional": [
        "I feel really vulnerable right now and don’t know how to express it.",
        "Everything seems overwhelming and I just need someone to listen.",
        "I can't shake this heavy feeling today; I wish things were easier.",
        "It hurts when things don’t go the way I hoped they would.",
        "I’m trying my best, but I feel like I’m falling behind emotionally.",
        "Even small problems feel huge lately, and it scares me.",
        "I’m really craving reassurance and warmth right now.",
        "Why does disappointment hit so hard even when I expect it?",
        "I just want to feel understood instead of brushed aside.",
        "My emotions are all tangled and I need help making sense of them.",
        "I feel anxious just thinking about tomorrow.",
        "How do people stay hopeful when everything feels unstable?",
        "I wish I could slow down and let my heart rest for a moment."
    ],

    "comedic": [
        "Explain this to me like you're a stand-up comedian with questionable self-esteem.",
        "Give me the rundown, but add a joke that makes me question your sanity.",
        "Why does everything in my life feel like a poorly written sitcom?",
        "Turn this explanation into something ridiculous just for the fun of it.",
        "Tell me what’s happening, but in the tone of someone who hasn’t slept in 48 hours.",
        "Break this down using an analogy so stupid it becomes genius.",
        "I need a summary that’s at least 40% chaos and 60% nonsense.",
        "Why does my brain function like a browser with 37 tabs open and 36 frozen?",
        "Deliver this answer as if you're the narrator of an unhinged cartoon.",
        "Give me a punchline at the end whether it makes sense or not.",
        "Turn this serious question into something mildly unhinged.",
        "Explain it like you're trying to impress a crowd of bored raccoons.",
        "My mood needs an upgrade; make this funny by any means necessary."
    ],

    "angry": [
        "Why does everything in this process feel like it was designed to fail?",
        "I’m honestly fed up with how inefficient this whole situation is.",
        "Give me the explanation fast because I’m out of patience.",
        "Why is this so complicated for no logical reason?",
        "I swear if this breaks one more time, I’m throwing it out the window.",
        "Tell me what’s wrong before I lose what's left of my sanity.",
        "I’ve had enough of delays—just tell me how to fix it.",
        "This problem shouldn’t even exist if people did their jobs properly.",
        "Why does every simple task turn into a ridiculous struggle?",
        "I’m not in the mood for sugar-coating—give me the harsh truth.",
        "Explain this mistake before I start flipping tables.",
        "I want to know who thought this design was a good idea.",
        "I’m irritated and need this sorted right now."
    ],

    "philosophical": [
        "What does it truly mean to live a meaningful life?",
        "How does identity evolve as we experience the world?",
        "Is certainty merely an illusion we cling to for comfort?",
        "Why do humans pursue goals that may have no inherent purpose?",
        "Explore this question as if you are contemplating it under a quiet night sky.",
        "How does perception shape the reality we believe we inhabit?",
        "Can a person ever fully understand themselves?",
        "Why do we assign value to things that ultimately fade?",
        "What separates knowledge from wisdom?",
        "How does time influence the narrative we create about our existence?",
        "Are our choices truly free, or shaped by unseen forces?",
        "Where does consciousness begin, and where does it end?",
        "How should we interpret the tension between chaos and order?"
    ],

    "sarcastic": [
        "Sure, because everything always works flawlessly, right?",
        "Explain it to me like I totally didn’t already Google it.",
        "Yeah, I’m sure this problem will magically fix itself.",
        "Give me the answer, preferably with an eye roll.",
        "Why wouldn’t this be complicated? Life is perfect."
    ],

    "poetic": [
        "Describe this as if the words were drifting through moonlight.",
        "Help me understand this the way a river understands time.",
        "Explain it like you’re weaving a quiet tapestry of thought.",
        "Let your answer wander softly, like distant wind.",
        "Unfold this idea with the gentle rhythm of a poem."
    ],

    "formal": [
        "Please provide a detailed and professionally structured explanation.",
        "Kindly outline the essential points in a formal manner.",
        "I request a well-organized clarification on this matter.",
        "Deliver the information with precision and professionalism.",
        "Offer a concise, corporate-style summary regarding this issue."
    ],

    "gen_z": [
        "Explain this but make it lowkey understandable.",
        "Bro this situation is giving me major side-quest energy.",
        "Break it down like you’re talking to someone running on two brain cells.",
        "Not gonna lie, this whole thing feels kinda unhinged.",
        "Explain this in a vibe-heavy, no-cap kinda way."
    ],

    "storytelling": [
        "Explain this like a narrator guiding a hero through a dense forest.",
        "Break this down as if telling a tale passed across generations.",
        "Describe the situation like we’re opening the first chapter of a novel.",
        "Unfold this explanation with the drama of an ancient legend.",
        "Guide me through this topic like a storyteller by the fire."
    ],

    "ambiguous": [
        "I’m not sure how I feel about this.",
        "This could go in many directions; I can’t decide.",
        "It’s unclear what the right approach is here.",
        "There are too many possibilities to narrow down.",
        "I can’t tell what tone this situation needs."
    ],
}

def generate_random_sentence(persona):
    """Generate minor variations so the dataset isn't repetitive."""
    base = random.choice(PERSONAS[persona])
    extras = [
        "",
        " Please elaborate.",
        " I’ve been thinking about this a lot.",
        " Can you reflect on this?",
        " I want a deeper perspective.",
        " Could you clarify further?",
        " Add more detail so I can understand better.",
        " Give me a more nuanced explanation.",
        " Expand on this idea.",
        " Provide additional insight if possible."
    ]
    return base + random.choice(extras)

def create_dataset(n=12000, output_file="moe_persona_dataset.jsonl"):
    personas = list(PERSONAS.keys())
    random.shuffle(personas)
    with open(output_file, "w") as f:
        for _ in range(n):
            persona = random.choice(personas)
            text = generate_random_sentence(persona)
            entry = {"input": text, "label": persona}
            f.write(json.dumps(entry) + "\n")

    print(f"Dataset created: {output_file} with {n} samples.")

if __name__ == "__main__":
    create_dataset()