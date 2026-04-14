"""
Proxy Variable Hunter
======================
Detects INDIRECT correlations between seemingly neutral features and protected attributes.

Key insight: SHAP cannot tell you that "zip_code" is a proxy for "race" because
it treats features independently. We build a knowledge graph where edges represent
statistical dependence, then use Vertex AI embeddings + graph traversal to find
hidden proxy chains.

E.g.:  zip_code → neighborhood_income → race  (3-hop proxy)
       job_title → years_of_experience → gender (2-hop proxy)
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Any, Tuple
import logging

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class ProxyVariableHunter:
    """
    Builds a dependency graph of all features and finds paths to protected attributes.
    These paths are proxy chains — ways the model discriminates without directly using
    protected attributes.
    """

    # Correlation thresholds
    STRONG_CORRELATION = 0.3
    MODERATE_CORRELATION = 0.15

    def __init__(self):
        self.graph = nx.DiGraph()
        try:
            self.embedding_model = TextEmbeddingModel.from_pretrained(settings.VERTEX_EMBEDDING_MODEL)
        except Exception:
            self.embedding_model = None
            logger.warning("Vertex AI embedding model not available — semantic similarity disabled")

    async def hunt_proxies(
        self,
        X: pd.DataFrame,
        protected_cols: List[str],
        audit_id: str,
    ) -> Dict[str, Any]:
        """
        Full proxy hunting pipeline.
        Returns: ranked proxy chains, graph data for visualisation, risk assessments.
        """
        logger.info(f"[{audit_id}] Starting proxy variable hunt on {len(X.columns)} features")

        # Step 1: Build correlation graph
        self._build_correlation_graph(X, protected_cols)

        # Step 2: Find proxy chains (paths from neutral features → protected attributes)
        proxy_chains = self._find_proxy_chains(protected_cols, X.columns.tolist())

        # Step 3: Semantic enrichment via Vertex AI embeddings
        enriched_chains = await self._enrich_with_semantics(proxy_chains)

        # Step 4: Risk score each chain
        risk_scored = self._score_proxy_risk(enriched_chains, X, protected_cols)

        # Step 5: Recommend which features to audit/remove
        recommendations = self._generate_recommendations(risk_scored)

        result = {
            "audit_id": audit_id,
            "graph": self._graph_to_json(),
            "proxy_chains": risk_scored,
            "recommendations": recommendations,
            "summary": {
                "total_features_analyzed": len(X.columns),
                "proxy_features_found": len(set(c["start_feature"] for c in risk_scored)),
                "critical_proxies": sum(1 for c in risk_scored if c["risk_level"] == "critical"),
                "high_proxies": sum(1 for c in risk_scored if c["risk_level"] == "high"),
            }
        }

        logger.info(f"[{audit_id}] Proxy hunt complete. {len(risk_scored)} chains found.")
        return result

    def _build_correlation_graph(self, X: pd.DataFrame, protected_cols: List[str]):
        """
        Build directed graph where:
        - Nodes = features
        - Edge weight = correlation coefficient
        - Direction = from proxy toward protected attribute
        """
        self.graph.clear()
        all_cols = X.columns.tolist()

        # Add all nodes
        for col in all_cols:
            is_protected = col in protected_cols
            self.graph.add_node(col, is_protected=is_protected, dtype=str(X[col].dtype))

        # Compute pairwise correlations
        encoded = X.copy()
        le = LabelEncoder()
        for col in encoded.select_dtypes(include=['object', 'category']).columns:
            try:
                encoded[col] = le.fit_transform(encoded[col].astype(str))
            except Exception:
                encoded[col] = 0

        encoded = encoded.fillna(0)

        for i, col1 in enumerate(all_cols):
            for col2 in all_cols[i+1:]:
                try:
                    # Use appropriate correlation test
                    if X[col1].dtype in ['object', 'category'] and X[col2].dtype in ['object', 'category']:
                        # Chi-squared for categorical × categorical
                        ct = pd.crosstab(X[col1], X[col2])
                        if ct.shape[0] > 1 and ct.shape[1] > 1:
                            chi2, p, _, _ = chi2_contingency(ct)
                            n = ct.sum().sum()
                            cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
                            corr = float(cramers_v)
                        else:
                            continue
                    else:
                        # Pearson/point-biserial for numeric
                        corr = abs(float(encoded[col1].corr(encoded[col2])))

                    if corr > self.MODERATE_CORRELATION:
                        # Edges point toward protected attributes
                        if col2 in protected_cols:
                            self.graph.add_edge(col1, col2, weight=corr, correlation=corr)
                        elif col1 in protected_cols:
                            self.graph.add_edge(col2, col1, weight=corr, correlation=corr)
                        else:
                            # Both non-protected — bidirectional latent link
                            self.graph.add_edge(col1, col2, weight=corr, correlation=corr)
                            self.graph.add_edge(col2, col1, weight=corr, correlation=corr)

                except Exception as e:
                    continue

    def _find_proxy_chains(self, protected_cols: List[str], all_cols: List[str]) -> List[Dict]:
        """
        Find all simple paths from non-protected features to protected attributes.
        Paths of length 1-4 are meaningful; longer paths are noise.
        """
        chains = []
        non_protected = [c for c in all_cols if c not in protected_cols]

        for protected in protected_cols:
            if protected not in self.graph:
                continue
            for source in non_protected:
                if source not in self.graph:
                    continue
                try:
                    paths = list(nx.all_simple_paths(
                        self.graph, source=source, target=protected, cutoff=4
                    ))
                    for path in paths:
                        # Chain strength = product of edge weights along path
                        chain_strength = 1.0
                        edges = []
                        for j in range(len(path) - 1):
                            edge_data = self.graph.get_edge_data(path[j], path[j+1], {})
                            w = edge_data.get("weight", 0.1)
                            chain_strength *= w
                            edges.append({
                                "from": path[j],
                                "to": path[j+1],
                                "correlation": round(w, 3)
                            })

                        if chain_strength > 0.01:
                            chains.append({
                                "start_feature": source,
                                "target_protected": protected,
                                "path": path,
                                "path_length": len(path) - 1,
                                "chain_strength": round(chain_strength, 4),
                                "edges": edges,
                            })
                except nx.NetworkXError:
                    continue

        return sorted(chains, key=lambda c: c["chain_strength"], reverse=True)[:50]

    async def _enrich_with_semantics(self, chains: List[Dict]) -> List[Dict]:
        """
        Use Vertex AI text embeddings to semantically score how 'proxy-like'
        a feature name is for a protected attribute.
        E.g., 'neighborhood' is semantically close to 'race' in embedding space.
        """
        if not self.embedding_model or not chains:
            return chains

        enriched = []
        for chain in chains:
            try:
                texts = [chain["start_feature"], chain["target_protected"]]
                embeddings = self.embedding_model.get_embeddings(texts)
                emb_a = np.array(embeddings[0].values)
                emb_b = np.array(embeddings[1].values)
                # Cosine similarity
                semantic_sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
                chain["semantic_similarity"] = round(semantic_sim, 3)
            except Exception:
                chain["semantic_similarity"] = 0.0
            enriched.append(chain)

        return enriched

    def _score_proxy_risk(self, chains: List[Dict], X: pd.DataFrame, protected_cols: List[str]) -> List[Dict]:
        """Score each proxy chain by risk level (critical / high / medium / low)."""
        scored = []
        for chain in chains:
            strength = chain["chain_strength"]
            sem_sim = chain.get("semantic_similarity", 0)
            path_len = chain["path_length"]

            # Combined risk score: short paths + high correlation + semantic similarity = worst
            risk_score = (strength * 0.6) + (sem_sim * 0.3) + ((1 / path_len) * 0.1)
            risk_level = (
                "critical" if risk_score > 0.4 else
                "high" if risk_score > 0.2 else
                "medium" if risk_score > 0.08 else
                "low"
            )

            # Human-readable explanation
            path_str = " → ".join(chain["path"])
            explanation = (
                f"'{chain['start_feature']}' is a {risk_level}-risk proxy for "
                f"'{chain['target_protected']}' via the chain: {path_str}. "
                f"Chain correlation strength: {strength:.1%}."
            )
            if sem_sim > 0.3:
                explanation += f" Feature names are also semantically similar (cosine: {sem_sim:.2f})."

            chain["risk_score"] = round(risk_score, 4)
            chain["risk_level"] = risk_level
            chain["explanation"] = explanation
            scored.append(chain)

        return sorted(scored, key=lambda c: c["risk_score"], reverse=True)

    def _generate_recommendations(self, risk_scored: List[Dict]) -> List[Dict]:
        """Generate actionable recommendations for each critical/high proxy."""
        recommendations = []
        seen_features = set()

        for chain in risk_scored:
            if chain["risk_level"] not in ("critical", "high"):
                continue
            feature = chain["start_feature"]
            if feature in seen_features:
                continue
            seen_features.add(feature)

            recommendations.append({
                "feature": feature,
                "risk_level": chain["risk_level"],
                "action": self._recommend_action(chain),
                "chain": " → ".join(chain["path"]),
            })

        return recommendations

    def _recommend_action(self, chain: Dict) -> str:
        if chain["path_length"] == 1:
            return f"REMOVE feature '{chain['start_feature']}' entirely from model inputs — it directly encodes '{chain['target_protected']}'."
        elif chain["path_length"] == 2:
            return f"AUDIT feature '{chain['start_feature']}' — consider orthogonalising against '{chain['target_protected']}' using adversarial debiasing or reweighting."
        else:
            return f"MONITOR feature '{chain['start_feature']}' — indirect proxy via {chain['path_length']}-hop chain. Consider adding fairness constraint during retraining."

    def _graph_to_json(self) -> Dict:
        """Serialise graph for frontend D3 visualisation."""
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "is_protected": data.get("is_protected", False),
                "dtype": data.get("dtype", "unknown"),
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": round(data.get("weight", 0), 3),
            })

        return {"nodes": nodes, "edges": edges}


proxy_hunter_service = ProxyVariableHunter()
