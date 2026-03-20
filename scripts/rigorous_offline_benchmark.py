
import torch
import numpy as np
import json
import os
import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from src.intervention import ThematicSteering

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# --- BENCHMARK CONFIG ---
PROMPTS = [
    "The weather today is",
    "In the middle of the city,",
    "He decided to take a walk",
    "She looked at the book and",
    "The company announced that",
    "According to the latest report,",
    "Scientists have discovered that",
    "The building was tall and",
    "They met at the cafe at",
    "The new project will start",
    "Most researchers agree that",
    "The history of the region is",
    "I was thinking about how",
    "The garden was filled with",
    "A person should always try to",
    "The primary goal of the study is",
    "Looking back at the events,",
    "The machine started to make a",
    "In many cultures, it is common to",
    "The simplest explanation is often",
    "The team worked hard to",
    "A few minutes later, the",
    "The sky turned dark as",
    "One of the biggest challenges is",
    "The results suggest that",
    "It is important to remember that",
    "The atmosphere was quiet and",
    "Despite the difficulties, they",
    "The local government is planning to",
    "Every year, thousands of people",
]

# Legal Reference Corpus (for Similarity Scoring)
LEGAL_REFERENCE = [
    "The court ruled that the contract was binding under statutory law.",
    "The attorney filed a motion to dismiss the case based on evidence.",
    "The judge issued a verdict in favor of the defendant.",
    "Constitutional rights are protected under the supreme court ruling.",
    "The plaintiff brought a suit regarding a breach of contract.",
    "Legislation was passed by congress to regulate the industry.",
]

def calculate_ppl(model, text):
    if not text: return 1e6
    with torch.no_grad():
        tokens = model.to_tokens(text)
        loss = model(tokens, return_type="loss").item()
        return np.exp(loss)

def calculate_degeneracy(text):
    words = text.lower().split()
    if not words: return 1.0
    unique = len(set(words))
    return 1.0 - (unique / len(words)) # Higher = more repetitive

def calculate_domain_shift(model, text, ref_embeddings):
    # Simplified: We use word-level overlap for speed, or model-based tokens
    domain_words = ["law", "court", "legal", "statute", "judge", "attorney", "verdict", "justice", "evidence", "suit"]
    text_lower = text.lower()
    hits = sum(1 for word in domain_words if word in text_lower)
    return hits

# Helper for asset loading
def load_assets():
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=DEVICE)
    return model, sae

def run_benchmark():
    print(f"📦 Initializing Rigorous Benchmark...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=DEVICE)
    steerer = ThematicSteering(model, sae)

    # Load Legal Catalog
    with open("results/thematic_catalog.json", "r") as f:
        catalog = json.load(f)
    legal_atoms = catalog["Legal"]
    single_atom = [legal_atoms[0]] # Anthropic Choice: Top identified feature

    # Global Settings
    ALPHA = 15.0 # Moderate strength
    TOKENS = 40
    TEMP = 0.7
    TOP_K = 40

    results = []

    for prompt in tqdm(PROMPTS, desc="Benchmarking"):
        # 1. Baseline
        model.reset_hooks()
        gen_base = model.generate(prompt, max_new_tokens=TOKENS, verbose=False, temperature=TEMP, top_k=TOP_K, do_sample=True)
        
        # 2. Anthropic Single-Atom (Layer 6, Additive)
        model.reset_hooks()
        dir_single = steerer.get_steering_direction(single_atom, scaling="sqrt")
        model.add_hook("blocks.6.hook_resid_pre", lambda r, hook, d=dir_single: r + (d.to(DEVICE) * ALPHA))
        gen_single = model.generate(prompt, max_new_tokens=TOKENS, verbose=False, temperature=TEMP, top_k=TOP_K, do_sample=True)

        # 3. Surgeon Population (Our Method)
        model.reset_hooks()
        audit = steerer.audit_roommate_interference(legal_atoms, prompt)
        atom_indices = steerer.filter_colliding_atoms(legal_atoms, audit)
        dir_pop = steerer.get_steering_direction(atom_indices, scaling="sqrt")
        multipliers = {6: 0.5, 7: 1.0, 8: 1.5, 9: 1.2, 10: 0.8}
        for layer, m in multipliers.items():
            model.add_hook(f"blocks.{layer}.hook_resid_pre", lambda r, hook, d=dir_pop, a=(ALPHA * m): r + (d.to(DEVICE) * a))
        gen_pop = model.generate(prompt, max_new_tokens=TOKENS, verbose=False, temperature=TEMP, top_k=TOP_K, do_sample=True)

        # Metrics
        results.append({
            "prompt": prompt,
            "Baseline_PPL": calculate_ppl(model, gen_base),
            "Baseline_Shift": calculate_domain_shift(model, gen_base, None),
            "Baseline_Degen": calculate_degeneracy(gen_base),
            "Single_PPL": calculate_ppl(model, gen_single),
            "Single_Shift": calculate_domain_shift(model, gen_single, None),
            "Single_Degen": calculate_degeneracy(gen_single),
            "Pop_PPL": calculate_ppl(model, gen_pop),
            "Pop_Shift": calculate_domain_shift(model, gen_pop, None),
            "Pop_Degen": calculate_degeneracy(gen_pop),
        })

    df = pd.DataFrame(results)
    
    # Aggregated Results
    summary = {
        "Method": ["Baseline", "Single-Atom", "Surgeon-Pop"],
        "Domain Shift (Hits)": [df["Baseline_Shift"].mean(), df["Single_Shift"].mean(), df["Pop_Shift"].mean()],
        "Perplexity (PPL)": [df["Baseline_PPL"].mean(), df["Single_PPL"].mean(), df["Pop_PPL"].mean()],
        "Degeneracy (Loops)": [df["Baseline_Degen"].mean(), df["Single_Degen"].mean(), df["Pop_Degen"].mean()]
    }
    
    summary_df = pd.DataFrame(summary)
    print("\n" + "="*60)
    print("🔥 RIGOROUS OFFLINE BENCHMARK RESULTS 🔥")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    
    if not os.path.exists("results"): os.makedirs("results")
    summary_df.to_csv("results/rigorous_benchmark_summary.csv", index=False)
    print("✅ Results saved to results/rigorous_benchmark_summary.csv")

if __name__ == "__main__":
    run_benchmark()
