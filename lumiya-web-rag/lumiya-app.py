import re
import numpy as np
import requests
import gradio as gr
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Load lightweight embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Expanded realistic browser headers to bypass basic anti-bot blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}


def extract_full_body_text(url: str) -> str:
    """Fetches a URL with realistic browser headers and extracts clean body text."""
    try:
        session = requests.Session()
        response = session.get(url, headers=HEADERS, timeout=8, allow_redirects=True)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.content, "html.parser")

        # Strip non-content elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe"]):
            element.decompose()

        # Locate body text
        main_content = (
                soup.find("article")
                or soup.find("main")
                or soup.find("div", {"id": re.compile(r"content|main|article", re.I)})
                or soup.body
        )

        if not main_content:
            return ""

        text = main_content.get_text(separator="\n", strip=True)
        cleaned_lines = [line for line in text.splitlines() if len(line.strip()) > 30]

        return "\n\n".join(cleaned_lines)

    except Exception:
        return ""


def filter_majority_and_get_top_article(query: str, max_results: int = 5):
    """
    1. Fetches search results.
    2. Clusters results to identify dominant majority topic.
    3. Iterates over majority cluster links until full body text is extracted.
    """
    raw_results = list(DDGS().text(query, max_results=max_results))
    if not raw_results:
        return "No search results found."

    # 1. Embed snippet texts
    texts = [f"{item.get('title', '')}. {item.get('body', '')}" for item in raw_results]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    # 2. Cluster using Cosine Distance
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=0.4
    )

    if len(raw_results) < 2:
        cluster_labels = np.array([0])
    else:
        cluster_labels = clustering_model.fit_predict(embeddings)

    # 3. Group by cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(raw_results[idx])

    # 4. Find majority cluster
    majority_label = max(clusters, key=lambda l: len(clusters[l]))
    majority_items = clusters[majority_label]

    # 5. Loop through majority cluster links until full text is found
    selected_item = None
    extracted_text = ""

    for item in majority_items:
        url = item.get("href", item.get("link", "#"))
        scraped_content = extract_full_body_text(url)

        # If successfully extracted at least 300 characters, keep it
        if len(scraped_content) >= 300:
            extracted_text = scraped_content
            selected_item = item
            content_type = "Full Scraped Body Text"
            break

    # Fallback to snippet of top match if all page extractions were blocked
    if not selected_item:
        selected_item = majority_items[0]
        snippet = selected_item.get("body", selected_item.get("snippet", "No snippet available."))
        extracted_text = f"*Note: All web page extractions were blocked. Fallback to snippet:*\n\n{snippet}"
        content_type = "Search Snippet (Fallback)"

    stats = {
        "fetched": len(raw_results),
        "clusters": len(clusters),
        "kept": len(majority_items),
        "discarded": len(raw_results) - len(majority_items),
        "title": selected_item.get("title", "No Title"),
        "url": selected_item.get("href", selected_item.get("link", "#")),
        "type": content_type,
        "body": extracted_text,
    }

    return stats


def run_web_search(query: str, max_results: int):
    if not query.strip():
        return "⚠️ Please enter a query."

    try:
        stats = filter_majority_and_get_top_article(query, max_results=max_results)

        if isinstance(stats, str):
            return stats

        formatted_output = f"### 🔍 Query: **{query}**\n"
        formatted_output += f"📊 **Cluster Stats:** Fetched `{stats['fetched']}` | Topic Groups: `{stats['clusters']}` | Majority Cluster Items: `{stats['kept']}` (Discarded `{stats['discarded']}`)\n\n"
        formatted_output += "---\n\n"
        formatted_output += f"### 🏆 **Top Matched Source:** [{stats['title']}]({stats['url']})\n"
        formatted_output += f"*Source Type: `{stats['type']}`*\n\n"
        formatted_output += "#### 📄 **Extracted Full Body Text:**\n"
        formatted_output += f"```text\n{stats['body']}\n```"

        return formatted_output

    except Exception as e:
        return f"❌ Search error: {str(e)}"


# Define Interface
gr.close_all()
with gr.Blocks(title="Lumiya AI - Web Search Engine") as demo:
    gr.Markdown("# 🌐 **Lumiya AI**")
    gr.Markdown("*Real-Time Web Search & Intelligence Engine*")

    with gr.Row():
        query_input = gr.Textbox(
            label="Ask Lumiya anything...",
            placeholder="e.g., What is RAG in AI?",
            scale=4,
        )
        results_count = gr.Slider(
            minimum=3,
            maximum=10,
            value=5,
            step=1,
            label="Max Web Sources",
            scale=1,
        )

    search_btn = gr.Button("Search Web", variant="primary")
    output_markdown = gr.Markdown(label="Results")

    search_btn.click(
        fn=run_web_search,
        inputs=[query_input, results_count],
        outputs=[output_markdown],
    )
    query_input.submit(
        fn=run_web_search,
        inputs=[query_input, results_count],
        outputs=[output_markdown],
    )

demo.launch()