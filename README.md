

# 🌐 Lumiya AI Engine: Real-Time Web Intelligence & Synthesis Pipeline

Lumiya AI is an automated, real-time research and web search engine built using Python, Hugging Face Transformers, and Gradio. It retrieves live web search data, applies semantic agglomerative clustering to isolate majority consensus topics, extracts deep multi-source context, and streams synthesized answers live using a local LLM with inline bracketed citations.

---

## 📌 1. Motivation

The project originally started as a local **Document QnA (RAG)** application built to answer queries from static user-uploaded files (PDFs/text). While effective for fixed knowledge bases, it could not answer real-time, up-to-date queries.

To overcome this limitation, the project pivoted toward building a **Perplexity-style real-time AI search engine**. The objective was to eliminate manual file uploads and enable an AI system to fetch, cluster, and synthesize live web content directly from the internet while maintaining factual accuracy, transparency, and complete topic coverage.

---

## ⚙️ 2. How This Application Works

Lumiya AI operates through an automated end-to-end intelligence pipeline:

1. **Live Web Retrieval:** Takes the user's search query and fetches structured web pages using the Tavily AI API.
2. **Semantic Clustering:** Vectorizes extracted search texts using `SentenceTransformer` and clusters them via Cosine Distance Agglomerative Clustering to identify consensus themes.
3. **Multi-Source Context Aggregation:** Extracts and merges content from the top 3 matching documents within the majority cluster to ensure broad topic coverage.
4. **LLM Synthesis & Streaming:** Feeds the aggregated context to `Qwen2.5-1.5B-Instruct`, streaming generated tokens live to the UI via `TextIteratorStreamer` with inline citations (`[Source 1]`, `[Source 2]`).
5. **Interactive UI Display:** Renders the live response inside a Gradio Chatbot interface alongside a dynamic reference list containing clickable source links.

---

## 🏗️ 3. System Architecture

```text
               +----------------------------------+
               |        User Query (Gradio)       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |    Tavily Search API Fetch       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  SentenceTransformer Embedder    |
               |       ("all-MiniLM-L6-v2")       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |    Agglomerative Clustering      |
               |    (Cosine Distance Matrix)      |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Extract Top 3 Majority Cluster   |
               |        Consensus Documents       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   Qwen2.5-1.5B-Instruct LLM      |
               |  (TextIteratorStreamer Async)    |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Streamed Output & References UI  |
               +----------------------------------+
```

---

## 🔄 4. End-to-End Workflow

1. **Input Stage:** The user enters a query into the Gradio text box and selects the web search depth ($3$ to $20$).
2. **Fetch & Cleaning:** `fetch_tavily_results()` queries the API and removes short, low-quality, or empty web results.
3. **Vector Encoding & Distance Matrix:** `cluster_search_results()` encodes valid items into embeddings (`all-MiniLM-L6-v2`) and builds an $N \times N$ cosine distance matrix.
4. **Hierarchical Clustering:** `AgglomerativeClustering` (linkage: `average`, threshold: `0.4`) groups semantically related documents and selects the majority cluster with the highest information density.
5. **Context Payload Formatting:** `build_output_payload()` selects the top 3 matching documents from the majority cluster and wraps them with explicit source delimiters (`--- [Source X] ---`).
6. **LLM Prompting & Live Streaming:** `synthesize_answer_stream()` constructs prompt messages with strict citation requirements and offloads generation to a background thread using `TextIteratorStreamer`.
7. **UI Rendering:** `app.py` streams tokens to the Gradio `Chatbot` component live and appends a formatted reference legend mapped to source URLs.

---

## 📁 5. Project Directory & Component Breakdown

```text
lumiya-ai-engine/
├── web_scraper.py          # Tavily search retrieval, embedding generation, & clustering logic       
│   generator.py            # Model loading, prompt construction, & async token streaming
├── app.py                  # Gradio Chatbot UI layout, event handlers, & citation rendering
├── requirements.txt        # Python dependency list
└── README.md               # Project documentation
```

### Module Responsibilities:
* **`web_scraper.py`**: Handles API communication, encodes search texts, performs cosine distance matrix calculations, identifies the majority cluster, and packages top-3 consensus context.
* **`generator.py`**: Loads `Qwen/Qwen2.5-1.5B-Instruct`, manages system chat templates, enforces citation rules, and handles asynchronous token streaming via `TextIteratorStreamer`.
* **`app.py`**: Builds the Gradio user interface, manages chat history states, calls the streaming pipeline generator, and appends the reference footer.

---

## 🚨 6. Challenges Faced

1. **Direct Scraper Failures & HF Deployment Blockers:** Direct web scraping libraries produced noisy HTML and failed or got blocked when deployed on cloud environments like Hugging Face Spaces due to IP restrictions and rate limits.
2. **Single Top-Result Bias (Context Pollution):** Relying solely on the raw 0th search result (`results[0]`) caused the model to miss key information, inherit single-article bias, or process unverified noise.
3. **Incomplete Topic Coverage:** Processing only one document failed to provide comprehensive answers for complex, multi-part topics (such as detailing all 5 SOLID principles).
4. **Output Truncation:** Low token limits caused lengthy technical explanations and code snippets to cut off mid-sentence.
5. **High Latency:** Waiting for full generation to finish before displaying text created visual delays for users.

---

## 💡 7. Solutions Implemented

1. **API-Based Search Integration:** Replaced fragile web scrapers with **Tavily AI API**, ensuring structured, reliable text extractions safe for cloud deployments.
2. **Semantic Agglomerative Clustering:** Implemented cosine distance-based clustering on embeddings to filter out outlier articles and isolate majority consensus context.
3. **Top-3 Majority Extraction:** Expanded context collection to aggregate the **top 3 matching documents** from the majority cluster, maximizing topic coverage.
4. **Expanded Generation Budget:** Increased `max_new_tokens` to **1536+** in `generator.py`, giving the LLM sufficient room for full code samples and multi-part answers.
5. **Asynchronous Live Streaming:** Integrated `TextIteratorStreamer` with background threading in `generator.py` to stream tokens to Gradio in real time.
6. **In-Text Citation System:** Instructed the model to tag claims with bracketed citations (`[Source 1]`, `[Source 2]`) and dynamically built a clickable reference footer in `app.py`.

---

## 🛠️ 8. Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ and `pip` installed.

### 2. Install Dependencies
```bash
pip install torch transformers sentence-transformers scikit-learn tavily-python gradio
```

### 3. Set Tavily API Key
Set your Tavily API key as an environment variable:

* **Linux/macOS:**
  ```bash
  export TAVILY_API_KEY="your_tavily_api_key_here"
  ```
* **Windows (Command Prompt):**
  ```cmd
  set TAVILY_API_KEY="your_tavily_api_key_here"
  ```
* **Windows (PowerShell):**
  ```powershell
  $env:TAVILY_API_KEY="your_tavily_api_key_here"
  ```

### 4. Run Application
```bash
python app.py
```