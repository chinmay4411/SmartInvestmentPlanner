import os, glob, json
import re
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[1]
KB_DIR = BASE / "data" / "kb"

_sentences = []
_kb_loaded = False

def load_knowledge_base():
    """Load knowledge base files into memory"""
    global _sentences, _kb_loaded
    
    if _kb_loaded:
        return
    
    files = glob.glob(str(KB_DIR / "*.txt"))
    sentences = []
    
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                txt = fh.read().strip()
                if txt:
                    # Split by paragraphs and sentences
                    paragraphs = [p.strip() for p in txt.split("\n\n") if p.strip()]
                    for para in paragraphs:
                        # Also split by sentences for better matching
                        sentences_in_para = re.split(r'[.!?]+', para)
                        for sent in sentences_in_para:
                            sent = sent.strip()
                            if len(sent) > 20:  # Only keep meaningful sentences
                                sentences.append({
                                    "text": sent,
                                    "source": os.path.basename(f),
                                    "paragraph": para
                                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    _sentences = sentences
    _kb_loaded = True
    print(f"Loaded {len(sentences)} knowledge base entries")

def simple_text_similarity(text1, text2):
    """Simple text similarity based on word overlap"""
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not words1 or not words2:
        return 0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0

def keyword_match_score(query, text):
    """Calculate keyword matching score"""
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    
    if not query_words:
        return 0
    
    matches = query_words.intersection(text_words)
    return len(matches) / len(query_words)

def retrieve(query, top_k=3):
    """Retrieve relevant knowledge base entries"""
    global _sentences
    
    if not _kb_loaded:
        load_knowledge_base()
    
    if not _sentences:
        return []
    
    # Score each sentence
    scored_sentences = []
    for sent in _sentences:
        # Combine similarity and keyword matching
        similarity = simple_text_similarity(query, sent["text"])
        keyword_score = keyword_match_score(query, sent["text"])
        
        # Weighted score
        total_score = 0.6 * similarity + 0.4 * keyword_score
        
        scored_sentences.append((total_score, sent))
    
    # Sort by score and return top_k
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    results = []
    for score, sent in scored_sentences[:top_k]:
        if score > 0.1:  # Only return if there's some relevance
            results.append(sent)
    
    return results

def get_investment_advice_context(income, expenses, age, risk, allocation):
    """Get contextual investment advice based on user profile"""
    context_parts = []
    
    # Age-based advice
    if age < 25:
        context_parts.append("Young investors can afford higher risk for long-term growth. Consider growth stocks and equity mutual funds.")
    elif age < 35:
        context_parts.append("Early career investors should balance growth with stability. SIPs in equity funds are ideal.")
    elif age < 50:
        context_parts.append("Mid-career investors should focus on wealth building while maintaining some stability.")
    else:
        context_parts.append("Pre-retirement investors should prioritize capital preservation and stable returns.")
    
    # Risk-based advice
    if risk <= 2:
        context_parts.append("Conservative investors should focus on FDs, bonds, and debt funds for capital preservation.")
    elif risk == 3:
        context_parts.append("Balanced investors can mix equity and debt instruments for moderate growth.")
    else:
        context_parts.append("Aggressive investors can allocate more to stocks and equity funds for higher returns.")
    
    # Income-based advice
    savings_rate = (income - expenses) / income * 100 if income > 0 else 0
    if savings_rate < 10:
        context_parts.append("Low savings rate requires expense reduction or income increase for better investment capacity.")
    elif savings_rate > 30:
        context_parts.append("High savings rate allows for diversified investment across multiple asset classes.")
    
    return " ".join(context_parts)
