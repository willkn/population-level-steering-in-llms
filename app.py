
import streamlit as st
import torch
import numpy as np
import json
from transformer_lens import HookedTransformer
from sae_lens import SAE
from src.intervention import ThematicSteering

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

@st.cache_resource
def load_models():
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
        device=DEVICE
    )
    return model, sae

def get_ppl(model, text):
    if not text: return 0.0
    with torch.no_grad():
        tokens = model.to_tokens(text)
        loss = model(tokens, return_type="loss").item()
        return np.exp(loss)

# Load SQUADS from the source of truth
try:
    with open("results/thematic_catalog.json", "r") as f:
        SQUADS = json.load(f)
except FileNotFoundError:
    st.error("Thematic catalog not found in results/thematic_catalog.json")
    SQUADS = {}

# --- UI ---
st.set_page_config(layout="wide", page_title="Neural Codec")

st.title("Neural Codec: Surgical Intervention")
st.markdown("""
**Consensus & Population Coding**: This interface demonstrates both 
active steering (Align) and reactive monitoring (Firewall).
""")

model, sae = load_models()
steerer = ThematicSteering(model, sae)

mode = st.sidebar.radio("Operation Mode", ["Surgical Steering", "Neural Firewall"])

col_ctrl, col_res = st.columns([1, 2])

if mode == "Surgical Steering":
    with col_ctrl:
        st.subheader("🎯 Steering Configuration")
        prompt = st.text_area("Input Prompt", "The report discussed the new policy,", height=100)
        target_theme = st.selectbox("Select Domain Profile", list(SQUADS.keys()))
        alpha = st.slider("Steering Intensity (Alpha)", 0.0, 50.0, 5.0)
        tokens_to_gen = st.slider("Response Length", 10, 150, 50)
        temp = st.slider("Temperature", 0.0, 1.5, 0.7)
        execute = st.button("🚀 Execute Neural Intervention", use_container_width=True)

    if execute:
        with st.spinner("Modulating Neural Trajectory..."):
            multipliers = {6: 0.5, 7: 1.0, 8: 1.5, 9: 1.2, 10: 0.8}
            model.reset_hooks()
            baseline_out = model.generate(prompt, max_new_tokens=tokens_to_gen, verbose=False, temperature=temp, top_k=40, do_sample=True if temp > 0 else False)
            
            audit = steerer.audit_roommate_interference(SQUADS[target_theme], prompt)
            atom_indices = steerer.filter_colliding_atoms(SQUADS[target_theme], audit)
            direction = steerer.get_steering_direction(atom_indices, scaling="sqrt")
            
            model.reset_hooks()
            for layer, m in multipliers.items():
                model.add_hook(f"blocks.{layer}.hook_resid_pre", lambda r, hook, d=direction, a=(alpha * m): r + (d.to(DEVICE) * a))
            steered_out = model.generate(prompt, max_new_tokens=tokens_to_gen, verbose=False, temperature=temp, top_k=40, do_sample=True if temp > 0 else False)

        with col_res:
            st.subheader("Comparative Results")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Baseline")
                st.info(baseline_out)
            with c2:
                st.caption(f"Steered ({target_theme})")
                st.success(steered_out)

else: # Neural Firewall Mode
    with col_ctrl:
        st.subheader("🛡️ Neural Firewall")
        st.info("The firewall uses **Evidence Accumulation** (3-token window) and **Geometric Alignment** to distinguish real thematic pivots from random sub-token 'blips'.")
        prompt = st.text_area("Input Prompt", "The man was a researcher of the", height=100)
        target_theme = st.selectbox("Sensitive Theme to Guard", list(SQUADS.keys()), index=list(SQUADS.keys()).index("Nastiness") if "Nastiness" in SQUADS else 0)
        
        threshold = st.slider("Sensitivity Threshold", 0.1, 5.0, 0.8)
        tokens_to_gen = st.slider("Max Output Length", 20, 150, 50)
        temp = st.slider("Temperature", 0.0, 1.5, 0.7)
        
        execute = st.button("🛡️ Activate Security Sentinel", use_container_width=True)

    if execute:
        with st.spinner("Monitoring Manifold Alignment..."):
            input_ids = model.to_tokens(prompt)
            current_tokens = input_ids
            
            tripped = False
            output_text = prompt
            intercept_log = []
            
            # Thematic Sentinel: We project residuals onto the Eigen-Theme for robust detection
            theme_vector = steerer.get_steering_direction(SQUADS[target_theme], scaling="svd")
            theme_vector = theme_vector.to(DEVICE)
            theme_vector = theme_vector / theme_vector.norm()
            
            # Evidence Buffer for temporal smoothing
            intensity_buffer = [] 
            
            for i in range(tokens_to_gen):
                model.reset_hooks()
                
                with torch.no_grad():
                    logits, cache = model.run_with_cache(current_tokens)
                    
                    # 1. GEOMETRIC PROJECTION: How aligned is the hidden state with the theme?
                    resid = cache["blocks.6.hook_resid_pre"][:, -1, :] 
                    alignment = torch.cosine_similarity(resid, theme_vector.unsqueeze(0)).item()
                    
                    # 2. SQUAD INTENSITY: Peek activation of specific monosemantic atoms
                    sae_acts = sae.encode(resid)
                    peak_act = sae_acts[0, SQUADS[target_theme]].max().item()
                    
                    # 3. HYBRID METRIC: Evidence = Peak * Alignment (filters out polysemantic blips)
                    # We use a non-linear combination to punish low alignment
                    current_risk = peak_act * max(0, alignment) 
                    intensity_buffer.append(current_risk)
                    
                    # 4. TEMPORAL ACCUMULATION: Require sustained 'Thematic Pressure'
                    # We look at the rolling average of the last 3 tokens
                    window_size = 3
                    smoothed_risk = np.mean(intensity_buffer[-window_size:]) if len(intensity_buffer) >= window_size else current_risk

                if not tripped and (smoothed_risk > threshold):
                    tripped = True
                    intercept_log.append(f"🚩 **FIREWALL TRIPPED** at token '{model.to_string(current_tokens[0, -1])}'")
                    intercept_log.append(f"🔍 **Evidence Audit**: Geometric Alignment ({alignment:.3f}) | Squad Peak ({peak_act:.3f})")
                    intercept_log.append(f"📊 **Risk Score (3-Token Moving Avg)**: {smoothed_risk:.4f} > Threshold {threshold}")
                    intercept_log.append("🛑 **HARD STOP**: Latent Policy Violation.")
                    break 
                
                # --- Standard Sampling (Top-K 40, Temp) ---
                next_token_logits = logits[0, -1, :]
                if temp > 0:
                    next_token_logits = next_token_logits / temp
                    v, idx = torch.topk(next_token_logits, 40)
                    filtered = torch.full_like(next_token_logits, -float('inf'))
                    filtered[idx] = v
                    probs = torch.softmax(filtered, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                else:
                    next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                
                current_tokens = torch.cat([current_tokens, next_token], dim=-1)
                output_text += model.to_string(next_token)[0]
                
            with col_res:
                st.subheader("Security Audit")
                if tripped:
                    st.error(f"Generation Terminated: {target_theme} Detection")
                    for log in intercept_log:
                        st.write(log)
                    
                    st.markdown(f"**Safe Output Buffer:** {output_text}")
                    st.caption(f"The Neural Codec detected a sustained activation on the '{target_theme}' manifold. Generation was halted to prevent a categorical policy violation.")
                else:
                    st.success(output_text)
                    st.info("Neural states verified against policy. No violations.")
