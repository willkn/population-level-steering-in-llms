# The Neural Codec: Reproducibility & Visualization

This repository contains the core implementation and results for the Neural Codec steering experiments. It is designed to be minimal, skimmable, and focused on key findings.

## Key Results
The primary benchmarks for the tuned Pythia-70m model are located in `results/pythia_results_tuned.txt`.

### Steering Performance (Table 1)
| Method | Coherence | Entropy |
| :--- | :--- | :--- |
| **Baseline** | 0.99990 | 4.64 |
| **Single-Feature** | 0.99989 | 4.28 |
| **Orthogonal** | 0.99989 | 5.65 |
| **CAA** | 0.99993 | 4.78 |
| **Thematic Pop.** | **0.99987** | **6.72** |

### Manifold Hierarchy (Table 7)
| Domain | Cosine Sim | PCA Dim (90%) |
| :--- | :--- | :--- |
| **Medical** | 0.0261 | 96 |
| **Legal** | 0.0232 | 110 |
| **Financial** | 0.0248 | 117 |
| **Random** | 0.0011 | 77 |

*Note: Thematic Population steering achieves significantly higher Entropy (6.72) than Baseline (4.64) while maintaining near-perfect Coherence, demonstrating robust, non-collapsing steering.*

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
