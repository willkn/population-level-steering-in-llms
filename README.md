# The Neural Codec: Reproducibility & Visualization

This repository contains the core implementation and results for the Neural Codec steering experiments. It is designed to be minimal, skimmable, and focused on key findings.

## Key Results
The primary benchmarks for the tuned Pythia-70m model are located in `results/pythia_results_tuned.txt`.

### Summary of Findings
- **Coherence**: Steering remains stable under alpha scaling without collapse.
- **Metric Saturation**: Resolved via tuned alpha and similarity thresholds.
- **Interactive Visualization**: Run the Streamlit app to explore the manifold.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the visualization app:
   ```bash
   streamlit run app.py
   ```
3. Run the benchmark:
   ```bash
   python scripts/rigorous_offline_benchmark.py
   ```

## Repository Structure
- `src/`: Core intervention and steering logic.
- `scripts/`: Reproduction and benchmarking scripts.
- `results/`: Tuned metrics and raw terminal outputs.
- `app.py`: Streamlit dashboard.
- `paper/`: Placeholder for the associated paper PDF.
