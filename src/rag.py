"""
RAG system with TF-IDF retrieval for action plan generation.
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_corpus(dir_path: str) -> Tuple[List[str], List[str]]:
    """
    Load markdown documents from a directory.
    
    Args:
        dir_path: Path to corpus directory
    
    Returns:
        Tuple of (documents list, names list) where names are formatted as "[Doc#] filename.md"
    """
    docs = []
    names = []
    
    if not os.path.exists(dir_path):
        raise ValueError(f"Corpus directory not found: {dir_path}")
    
    # Get all .md files sorted by filename
    md_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.md')])
    
    for idx, filename in enumerate(md_files, start=1):
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            docs.append(content)
            names.append(f"[Doc{idx}] {filename}")
    
    return docs, names


def build_retriever(docs: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Build TF-IDF vectorizer and fit on documents.
    
    Args:
        docs: List of document texts
    
    Returns:
        Tuple of (vectorizer, document matrix)
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english',
        lowercase=True
    )
    X = vectorizer.fit_transform(docs)
    return vectorizer, X


def retrieve(
    vect: TfidfVectorizer,
    X: np.ndarray,
    docs: List[str],
    names: List[str],
    query: str,
    topk: int = 3
) -> List[Tuple[str, str, float]]:
    """
    Retrieve top-k documents by cosine similarity.
    
    Args:
        vect: Fitted TF-IDF vectorizer
        X: Document matrix
        docs: List of document texts
        names: List of document names
        query: Query string
        topk: Number of top documents to return
    
    Returns:
        List of tuples (doc_name, doc_text, similarity_score)
    """
    query_vec = vect.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    
    top_indices = np.argsort(similarities)[::-1][:topk]
    
    results = []
    for idx in top_indices:
        results.append((names[idx], docs[idx], float(similarities[idx])))
    
    return results


def synthesize_plan(
    query: str,
    retrieved: List[Tuple[str, str, float]],
    extra_context: Dict[str, Any] = None
) -> str:
    """
    Synthesize a 3-step action plan from retrieved documents.
    
    Args:
        query: Original query
        retrieved: List of (doc_name, doc_text, score) tuples
        extra_context: Optional dict with 'probability' key for lapse probability
    
    Returns:
        Formatted 3-step plan with citations
    """
    plan_steps = []
    
    # Add probability context if provided
    if extra_context and 'probability' in extra_context:
        prob = extra_context['probability']
        prob_line = f"Predicted lapse probability: {prob:.1%}"
        plan_steps.append(prob_line)
        plan_steps.append("")
    
    # Extract key insights from retrieved documents
    for step_num in range(1, 4):
        if step_num <= len(retrieved):
            doc_name, doc_text, score = retrieved[step_num - 1]
            
            # Extract doc number from name (e.g., "[Doc1] filename.md" -> "Doc1")
            doc_num = doc_name.split(']')[0].replace('[', '') if ']' in doc_name else f"Doc{step_num}"
            
            # Clean document text: remove markdown headers and extract meaningful content
            lines = doc_text.split('\n')
            content_lines = []
            for line in lines:
                line = line.strip()
                # Skip markdown headers and empty lines
                if line.startswith('#') or not line:
                    continue
                # Skip bullet points at start (will add in step)
                if line.startswith('-'):
                    line = line[1:].strip()
                content_lines.append(line)
            
            # Get first meaningful sentence or paragraph
            full_text = ' '.join(content_lines)
            sentences = [s.strip() for s in full_text.split('.') if s.strip()]
            
            if sentences:
                # Use first sentence, or first 150 chars if sentence is too long
                key_content = sentences[0]
                if len(key_content) > 150:
                    key_content = key_content[:150] + "..."
            else:
                key_content = full_text[:150] if len(full_text) > 150 else full_text
            
            # Create step with proper citation
            step = f"Step {step_num}: {key_content}"
            if not step.endswith('.'):
                step += "."
            step += f" [{doc_num}]"
            
            plan_steps.append(step)
        else:
            # Fallback: use first document again
            if retrieved:
                doc_name, doc_text, score = retrieved[0]
                doc_num = doc_name.split(']')[0].replace('[', '') if ']' in doc_name else "Doc1"
                step = f"Step {step_num}: Consider additional outreach based on customer profile. [{doc_num}]"
                plan_steps.append(step)
    
    return "\n".join(plan_steps)


def generate_lapse_plans(corpus_dir: str, customers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate lapse prevention plans for customers.
    
    Args:
        corpus_dir: Path to lapse corpus directory
        customers: List of customer dicts with at least 'policy_id' and 'probability'
    
    Returns:
        Dictionary with customer plans
    """
    docs, names = load_corpus(corpus_dir)
    vect, X = build_retriever(docs)
    
    plans = {}
    
    for customer in customers:
        policy_id = customer['policy_id']
        prob = customer.get('probability', 0.0)
        
        # Build query from customer context
        query_parts = ["lapse prevention", "customer retention"]
        if prob > 0.5:
            query_parts.append("high risk")
        elif prob < 0.2:
            query_parts.append("low risk")
        
        query = " ".join(query_parts)
        
        # Retrieve relevant documents
        retrieved = retrieve(vect, X, docs, names, query, topk=3)
        
        # Synthesize plan
        plan = synthesize_plan(query, retrieved, extra_context={'probability': prob})
        
        plans[str(policy_id)] = {
            'probability': float(prob),
            'plan': plan
        }
    
    return plans


def generate_lead_plans(corpus_dir: str, leads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate conversion plans for leads.
    
    Args:
        corpus_dir: Path to lead corpus directory
        leads: List of lead dicts with attributes like 'age', 'region', 'channel', etc.
    
    Returns:
        Dictionary with lead plans
    """
    docs, names = load_corpus(corpus_dir)
    vect, X = build_retriever(docs)
    
    plans = {}
    
    for idx, lead in enumerate(leads, start=1):
        lead_id = lead.get('lead_id', f"lead_{idx}")
        
        # Build query from lead attributes
        query_parts = ["lead conversion"]
        if 'channel' in lead:
            query_parts.append(lead['channel'])
        if 'region' in lead:
            query_parts.append(lead['region'])
        if 'objections' in lead:
            query_parts.append("objection handling")
        
        query = " ".join(query_parts)
        
        # Retrieve relevant documents
        retrieved = retrieve(vect, X, docs, names, query, topk=3)
        
        # Synthesize plan
        plan = synthesize_plan(query, retrieved)
        
        plans[str(lead_id)] = {
            'attributes': lead,
            'plan': plan
        }
    
    return plans
