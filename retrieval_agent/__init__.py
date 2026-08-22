"""AutoModelAdvisor retrieval agent - see api.py / README.md for library usage."""

from .api import (  # noqa: F401
    Recommendation,
    RecommendationError,
    recommend,
    recommend_batch,
    revise,
)

__all__ = ["Recommendation", "RecommendationError", "recommend", "recommend_batch", "revise"]
