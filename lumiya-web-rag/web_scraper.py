import os
import numpy as np
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

# Load embedding model once during module import
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Tavily API client
TAVILY_API_KEY = "" # os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


# ==========================================
# 1. TAVILY REQUEST METHOD
# ==========================================
def fetch_tavily_results(query: str, max_results: int = 7) -> list:
    """Performs web search via Tavily API and returns cleaned search items."""
    if not tavily_client:
        print("⚠️ Warning: TAVILY_API_KEY is not set.")
        return []

    try:
        response = tavily_client.search(
            query=query,
            max_results=6,
            search_depth="advanced",
            include_raw_content=False,
        )
        raw_results = response.get("results", [])
    except Exception as e:
        print(f"Tavily Search API Error: {e}")
        return []

    # Filter out empty or extremely short results
    valid_items = []
    for item in raw_results:
        title = str(item.get("title", "") or "").strip()
        body = str(item.get("content", "") or "").strip()
        combined = f"{title}. {body}".strip()

        if len(combined) > 10:
            item["combined_text"] = combined
            valid_items.append(item)

    return valid_items


# ==========================================
# 2. CLUSTERING METHOD
# ==========================================
def cluster_search_results(valid_items: list, distance_threshold: float = 0.4) -> tuple[list, int]:
    """Clusters search items by semantic embeddings and returns (majority_items, total_clusters)."""
    if not valid_items:
        return [], 0

    if len(valid_items) == 1:
        return valid_items, 1

    # Encode items & calculate distance matrix
    texts = [item["combined_text"] for item in valid_items]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    distance_matrix = cosine_distances(embeddings)
    distance_matrix = np.nan_to_num(distance_matrix, nan=0.0, posinf=1.0, neginf=0.0)

    # Perform Agglomerative Clustering
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold
    )
    cluster_labels = clustering_model.fit_predict(distance_matrix)

    # Group valid items by cluster label
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(valid_items[idx])

    total_clusters = len(clusters)
    majority_label = max(clusters, key=lambda l: len(clusters[l]))
    majority_items = clusters[majority_label]

    return majority_items, total_clusters


# ==========================================
# 3. EXTRACTION & PAYLOAD BUILDER METHOD
# ==========================================
def build_output_payload(raw_count: int, majority_items: list, total_clusters: int, top_k: int = 3) -> dict:
    """Takes up to top_k items from the majority cluster and formats the context payload."""
    if not majority_items:
        return {}

    # Select strictly the top matching documents from the majority cluster
    selected_items = majority_items[:top_k]

    combined_bodies = []
    sources_list = []

    for idx, item in enumerate(selected_items, 1):
        title = item.get("title", "Untitled").strip()
        url = item.get("url", "#").strip()
        content = item.get("content", "").strip()

        if content:
            # Clear source tagging for the LLM prompt context
            combined_bodies.append(f"--- [Source {idx}]: {title} ({url}) ---\n{content}")
            sources_list.append({"source_id": f"Source {idx}", "title": title, "url": url})

    extracted_text = "\n\n".join(combined_bodies)

    return {
        "fetched": raw_count,
        "clusters": total_clusters,
        "kept": len(selected_items),
        "discarded": raw_count - len(selected_items),
        "title": selected_items[0].get("title", "No Title"),
        "url": selected_items[0].get("url", "#"),
        "sources": sources_list,
        "type": f"Combined Consensus Text (Top {len(selected_items)} Sources)",
        "body": extracted_text,
    }


# ==========================================
# MAIN ENTRY POINT
# ==========================================
def retrieve_and_cluster(query: str, max_results: int = 8, top_k_sources: int = 3) -> dict:
    """Pipeline coordinator: fetches web data, clusters, and returns top 3 sources context payload."""
    query_str = query.strip()
    if not query_str:
        return {}

    # Step 1: Search API fetch
    valid_items = fetch_tavily_results(query_str, max_results=max_results)
    if not valid_items:
        return {}

    # Step 2: Semantic clustering
    majority_items, total_clusters = cluster_search_results(valid_items)

    # Step 3: Top 3 extraction & payload generation
    return build_output_payload(
        raw_count=len(valid_items),
        majority_items=majority_items,
        total_clusters=total_clusters,
        top_k=top_k_sources
    )