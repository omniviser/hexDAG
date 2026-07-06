"""Google Vertex AI / Gemini adapters (requires ``hexdag[google]``)."""

from hexdag.stdlib.adapters.google.vertex_adapter import (
    VertexAIAdapter,
    aclose_all_vertex_clients,
)

__all__ = ["VertexAIAdapter", "aclose_all_vertex_clients"]
