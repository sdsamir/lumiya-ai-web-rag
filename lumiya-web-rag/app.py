import gradio as gr
from web_scraper import retrieve_and_cluster
from generator import synthesize_answer_stream


# ==========================================
# HELPER METHOD: BUILD CITATION FOOTER
# ==========================================
def build_citation_footer(stats: dict) -> str:
    """Builds a formatted reference section at the bottom listing all cited sources."""
    sources = stats.get("sources", [])

    footer = "\n\n---\n"
    footer += f"📊 **Metrics:** Fetched `{stats.get('fetched', 0)}` | Clusters: `{stats.get('clusters', 0)}` | Retained: `{stats.get('kept', 0)}`\n\n"
    footer += "### 📚 **References & Sources**\n"

    if sources:
        for src in sources:
            source_id = src.get("source_id", "Source")
            title = src.get("title", "Untitled")
            url = src.get("url", "#")
            footer += f"* **[{source_id}]**: [{title}]({url})\n"
    else:
        primary_title = stats.get("title", "Primary Source")
        primary_url = stats.get("url", "#")
        footer += f"* **[Source 1]**: [{primary_title}]({primary_url})\n"

    return footer


# ==========================================
# MAIN STREAMING PIPELINE
# ==========================================
def run_lumiya_pipeline(user_message: str, history: list, max_results: int = 5):
    """Pipeline generator for Chatbot format: Appends user message and streams assistant response."""
    query_str = user_message.strip()
    if not query_str:
        yield history, ""
        return

    # Add user message to chatbot history immediately
    history = history + [{"role": "user", "content": query_str}]

    # Append initial placeholder assistant turn
    history.append({"role": "assistant", "content": "🔍 **Searching & Clustering Web Context...**"})
    yield history, ""

    try:
        stats = retrieve_and_cluster(query_str, max_results=max_results)

        if not stats or not stats.get("body"):
            history[-1]["content"] = "⚠️ Search returned no usable content. Please refine your query."
            yield history, ""
            return

        # Update initial streaming indicator
        history[-1]["content"] = "🧠 *Synthesizing Answer with LLM...*"
        yield history, ""

        full_answer = ""
        for token_chunk in synthesize_answer_stream(query_str, stats):
            full_answer = token_chunk
            history[-1]["content"] = full_answer
            yield history, ""

        # Final pass: Append citation reference footer
        citation_footer = build_citation_footer(stats)
        history[-1]["content"] = f"{full_answer}{citation_footer}"
        yield history, ""

    except Exception as e:
        history[-1]["content"] = f"❌ Pipeline error: {str(e)}"
        yield history, ""


# ==========================================
# CUSTOM STYLING & GRADIO UI LAYOUT
# ==========================================
custom_css = """
#lumiya-container {
    height: calc(100vh - 100px) !important;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
#chatbot-window {
    flex-grow: 1;
    overflow-y: auto !important;
    border-radius: 12px;
}
#input-row {
    margin-top: 10px;
    padding: 10px 0;
}
"""

with gr.Blocks(title="Lumiya AI Engine", css=custom_css) as demo:
    with gr.Column(elem_id="lumiya-container"):
        # TOP CONTAINER: Header & Output Chat Window
        gr.Markdown("# 🌐 **Lumiya AI Engine**")

        chatbot = gr.Chatbot(
            label="Lumiya Research Assistant",
            elem_id="chatbot-window",
            height=450
        )

        # BOTTOM CONTAINER: Input controls sticky layout
        with gr.Row(elem_id="input-row"):
            query_input = gr.Textbox(
                placeholder="Ask Lumiya anything...",
                show_label=False,
                scale=5,
                container=False
            )
            search_btn = gr.Button("Send", variant="primary", scale=1)

    # Event handlers: clear input box on submission & stream tokens to Chatbot
    search_btn.click(
        fn=run_lumiya_pipeline,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input],
    )
    query_input.submit(
        fn=run_lumiya_pipeline,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input],
    )

demo.launch()