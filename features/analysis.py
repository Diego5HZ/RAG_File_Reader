# Query intent, concept graph

import re
from typing import List
from langchain_core.documents import Document

def analyze_query_intent(prompt: str) -> dict:
    """Classifies query type for better retrieval"""
    prompt = prompt.lower()
    return {
        "query_type": (
            "technical" if any(term in prompt for term in ["formula", "theorem", "proof"]) else
            "comparative" if any(term in prompt for term in ["vs", "difference", "compare"]) else
            "temporal" if any(term in prompt for term in ["evolution", "trend", "over time"]) else
            "general"
        ),
        "target_entities": re.findall(r'\[([^\]]+)\]|"([^"]+)"', prompt)
    }

import networkx as nx
def build_concept_graph(documents: List[Document]) -> nx.Graph:
    """Creates a knowledge graph from document concepts"""
    G = nx.Graph()
    seen_pairs = set()
    
    for doc in documents:
        # Filter only noun phrases
        concepts = re.findall(r'\b(?:[A-Z][a-z]+\s?){2,}\b', doc.page_content)
        for i in range(len(concepts)-1):
            pair = tuple(sorted([concepts[i], concepts[i+1]]))
            if pair not in seen_pairs:
                G.add_edge(*pair, doc_id=doc.metadata.get("source_file"))
                seen_pairs.add(pair)
    return G