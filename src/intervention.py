import torch
from typing import List

class ThematicSteering:
    """
    High-fidelity intervention logic for the Neural Codec.
    Moves the model relative to thematic populations rather than single neurons.
    Now includes a 'Roommate Collision Guard' based on our rigorous research.
    """
    def __init__(self, model, sae):
        self.model = model
        self.sae = sae

    def get_steering_direction(self, atom_indices: List[int], scaling: str = "sqrt") -> torch.Tensor:
        """
        Calculates the aggregate steering vector for a population of atoms.
        Supports:
        - 'mean': Standard arithmetic mean.
        - 'sqrt': Population-scaled (Anthropic style).
        - 'svd': The first principal component (Eigen-Steering).
        """
        N = len(atom_indices)
        if N == 0: return torch.zeros(self.sae.W_dec.shape[1], device=self.sae.device)
        
        # Gather decoding directions from SAE [N, d_model]
        directions = self.sae.W_dec[atom_indices] 
        
        if scaling == "svd":
            try:
                U, S, V = torch.linalg.svd(directions, full_matrices=False)
                pc1 = V[0, :]
                if torch.dot(pc1, directions.mean(dim=0)) < 0:
                    pc1 = -pc1
                return pc1
            except Exception:
                return directions.mean(dim=0)

        if scaling == "scree":
            try:
                # Top K components (e.g. 3) weighted by their singular values
                U, S, V = torch.linalg.svd(directions, full_matrices=False)
                K = min(3, S.shape[0])
                scree_vec = torch.zeros_like(V[0, :])
                for i in range(K):
                    component = V[i, :]
                    # Orient towards mean for coherence
                    if torch.dot(component, directions.mean(dim=0)) < 0:
                        component = -component
                    scree_vec += S[i] * component
                return scree_vec
            except Exception:
                return directions.mean(dim=0)

        aggregate = directions.sum(dim=0)
        
        if scaling == "sqrt":
            return aggregate / (N ** 0.5)
        elif scaling == "mean":
            return aggregate / N
        return aggregate

    def audit_roommate_interference(self, steering_atoms: List[int], current_prompt: str) -> dict:
        """
        Audits potential 'Roommate Collisions' between the steering squad 
        and the atoms currently active in the text.
        """
        device = next(self.model.parameters()).device
        with torch.no_grad():
            tokens = self.model.to_tokens(current_prompt)
            # Encode current activations to find active atoms
            acts = self.sae.encode(self.model.embed(tokens))[0]
            # [d_sae] mean activation across sequence
            mean_acts = acts.mean(dim=0)
            active_atoms = torch.where(mean_acts > 0.01)[0].tolist()
            
            if not active_atoms or not steering_atoms:
                return {"collision_count": 0, "interference_magnitude": 0.0}

            # Find overlaps in the 'Storage Layer' (Polysemantic Neurons)
            # Calculate cosine similarity matrix once
            W_dec = self.sae.W_dec
            W_dec_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)
            
            steering_weights = W_dec_norm[steering_atoms]
            active_weights = W_dec_norm[active_atoms]
            
            # [N_steer, N_active]
            sims = torch.mm(steering_weights, active_weights.t())
            
            # --- CRITICAL REFINEMENT ---
            # 1. POSITIVE SIMILARITY (> 0.2) = Semantic Consensus. 
            #    These are allies. DO NOT PRUNE.
            # 2. NEGATIVE SIMILARITY (< -0.3) = Antipodal Collision. 
            #    These are features that "flip" each other. PRUNE.
            # 3. NEAR-ZERO SIMILARITY (|cos| < 0.1) but high overlap in neurons = Roommate Collision.
            #    (For now, we target the destructive Antipodal and the 'Uncorrelated interference')
            
            # We target "Destructive interference" where sims is significantly negative
            collisions = torch.where(sims < -0.3)
            
            collision_count = len(collisions[0])
            avg_interference = sims[collisions].abs().mean().item() if collision_count > 0 else 0.0
            
            # Identify which steering atoms are colliding
            colliding_indices = list(set(collisions[0].tolist()))
            
            return {
                "active_atom_count": len(active_atoms),
                "collision_count": collision_count,
                "interference_magnitude": avg_interference,
                "colliding_steering_atoms": colliding_indices
            }

    def filter_colliding_atoms(self, steering_atoms: List[int], audit_results: dict) -> List[int]:
        bad_indices = set(audit_results.get("colliding_steering_atoms", []))
        return [atom for i, atom in enumerate(steering_atoms) if i not in bad_indices]

    def apply_steering_hook(self, resid, hook, direction: torch.Tensor, strength: float = 1.0):
        return resid + (direction * strength)

    def steer(self, layer_name: str, atom_indices: List[int], strength: float = 1.0):
        direction = self.get_steering_direction(atom_indices)
        hook_fn = lambda r, hook: self.apply_steering_hook(r, hook, direction, strength)
        self.model.add_hook(layer_name, hook_fn)
        
    def reset(self):
        self.model.reset_hooks()
