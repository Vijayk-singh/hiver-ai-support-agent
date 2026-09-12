"""Knowledge Base and Historical Case Retrieval for Amazon Customer Support.

Indexes historical customer support tweet pairs to retrieve relevant historical
resolutions, policies, action links, and brand response patterns.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "amazonhelp_support_pairs.csv"


class SupportKnowledgeBase:
    """Searchable knowledge base of historical customer support pairs."""

    def __init__(self, data_path: Path = DEFAULT_DATA_PATH, max_features: int = 25000):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Support dataset not found at: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        self.max_features = max_features
        self._build_index()

    def _build_index(self):
        """Fit TF-IDF index over historical customer queries."""
        # Ensure customer_text is string
        self.df['customer_text'] = self.df['customer_text'].fillna('').astype(str)
        self.df['brand_text'] = self.df['brand_text'].fillna('').astype(str)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            stop_words='english',
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['customer_text'])

    def retrieve_similar(
        self,
        query: str,
        intent: Optional[str] = None,
        top_k: int = 3,
        intent_boost: float = 0.25
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k similar historical support conversations.

        Args:
            query: The incoming customer message.
            intent: Optional classified intent for category boosting.
            top_k: Number of historical cases to retrieve.
            intent_boost: Boost added to cosine similarity for matching intent.

        Returns:
            List of dicts containing historical customer query, brand response,
            intent, similarity score, and tweet IDs.
        """
        if not query or not query.strip():
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Apply intent boost if intent is provided and exists in dataset
        if intent and 'intent' in self.df.columns:
            intent_mask = (self.df['intent'] == intent).values
            adjusted_scores = sims + (intent_boost * intent_mask)
        else:
            adjusted_scores = sims

        # Get top-k indices
        top_indices = adjusted_scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "index": int(idx),
                "customer_tweet_id": int(row.get('customer_tweet_id', 0)),
                "brand_tweet_id": int(row.get('brand_tweet_id', 0)),
                "customer_text": str(row.get('customer_text', '')),
                "brand_text": str(row.get('brand_text', '')),
                "intent": str(row.get('intent', 'unknown')),
                "secondary_intent": str(row.get('secondary_intent', 'other')),
                "similarity": float(sims[idx]),
                "adjusted_score": float(adjusted_scores[idx])
            })

        return results

    def get_intent_resolution_summary(self, intent: str, n_samples: int = 5) -> Dict[str, Any]:
        """Get resolution patterns and common action links for a specific intent."""
        if 'intent' not in self.df.columns:
            return {}

        intent_rows = self.df[self.df['intent'] == intent]
        if intent_rows.empty:
            return {}

        sample_replies = intent_rows['brand_text'].head(n_samples).tolist()
        return {
            "intent": intent,
            "total_historical_cases": len(intent_rows),
            "sample_brand_replies": sample_replies
        }
