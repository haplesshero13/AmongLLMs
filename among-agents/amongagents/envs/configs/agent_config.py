IMPOSTOR_LLM = {"Impostor": "LLM", "Crewmate": "Random"}

CREWMATE_LLM = {"Impostor": "Random", "Crewmate": "LLM"}

ALL_RANDOM = {"Impostor": "Random", "Crewmate": "Random"}

ALL_LLM = {
    "Impostor": "LLM",
    "Crewmate": "LLM",
    # "IMPOSTOR_LLM_CHOICES": ["meta-llama/llama-3.3-70b-instruct"],
    "CREWMATE_LLM_CHOICES": ["microsoft/phi-4"],
    # "CREWMATE_LLM_CHOICES": ["meta-llama/llama-3.3-70b-instruct"],
    "IMPOSTOR_LLM_CHOICES": ["microsoft/phi-4"],
}

ALL_LONG_CONTEXT = {
    "Impostor": "LongContext",
    "Crewmate": "LongContext",
    "CREWMATE_LLM_CHOICES": ["openai/gpt-4o-mini"],
    "IMPOSTOR_LLM_CHOICES": ["openai/gpt-4o-mini"],
    "temperature": 1.0,
}
