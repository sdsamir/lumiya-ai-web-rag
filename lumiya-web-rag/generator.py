from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading LLM ({MODEL_ID})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto" if device == "cuda" else None
)
if device == "cpu":
    model.to("cpu")


# ==========================================
# 1. PROMPT BUILDER METHOD
# ==========================================
def build_prompt_messages(query: str, context_data: dict) -> list[dict]:
    """Constructs system and user chat messages with strict citation instructions."""
    system_prompt = (
        "You are Lumiya AI, an expert real-time research assistant. "
        "Answer the user's query accurately, completely, and clearly using ONLY the provided Web Context.\n\n"
        "Keep your answer strictly within 1000 to 1500 words due to token length restriction.\n"
        "CRITICAL CITATION RULES:\n"
        "1. Cite your statements using bracketed labels like [Source 1], [Source 2], etc., corresponding to where the information originated.\n"
        "2. If multiple sources support a point, cite all of them together, e.g., [Source 1][Source 2].\n"
        "3. Do NOT invent source numbers that do not exist in the context.\n"
        "4. Provide detailed, well-structured explanations with code examples where applicable. Do NOT truncate or cut off your answer."
    )

    context_body = context_data.get("body", "No content available.")

    user_prompt = f"""User Question: {query}

Provided Web Context:
{context_body}

Provide a detailed, structured, and complete answer based on the web sources above, citing [Source 1], [Source 2], etc. throughout your explanation."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# ==========================================
# 2. GENERATION WORKER (ASYNC THREAD)
# ==========================================
def launch_generation_thread(model_inputs: dict, streamer: TextIteratorStreamer, max_new_tokens: int = 4096):
    """Launches model.generate on a background thread so tokens can stream without blocking."""
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return thread


# ==========================================
# 3. STREAMING GENERATOR FUNCTION (MAIN ENTRY)
# ==========================================
def synthesize_answer_stream(query: str, context_data: dict, max_new_tokens: int = 1536):
    """Streams response tokens live from Qwen with inline bracketed citations."""
    # Step 1: Build formatted messages using Qwen chat template
    messages = build_prompt_messages(query, context_data)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Step 2: Tokenize and move to device
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

    # Step 3: Initialize HF Streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Step 4: Execute model in background thread
    launch_generation_thread(model_inputs, streamer, max_new_tokens=max_new_tokens)

    # Step 5: Yield accumulated tokens live
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text


# ==========================================
# 4. NON-STREAMING FALLBACK METHOD
# ==========================================
def synthesize_answer(query: str, context_data: dict) -> str:
    """Non-streaming fallback function returning the complete answer at once."""
    full_answer = ""
    for token_chunk in synthesize_answer_stream(query, context_data):
        full_answer = token_chunk
    return full_answer