"""Best-effort model parameter count (in billions) from a model id.

Primary: explicit size tokens in the name ('7B', '0.5B', '500m', bloom-style
'7b1'). Fallback: approximate sizes for well-known architecture families whose
names carry no token (vit_base, bert-large, ...). Returns None when unknown -
callers treat unknown as "fails the constraint" (conservative).
"""

import re

# Approximate parameter counts (billions) for common size-less families.
# Checked in order; first match wins.
_FAMILY_SIZES = [
    (r"vit[-_]?huge", 0.632), (r"vit[-_]?large", 0.307), (r"vit[-_]?base", 0.086),
    (r"vit[-_]?small", 0.022), (r"vit[-_]?tiny", 0.006),
    (r"deberta[-_]?v?\d?[-_]?xsmall", 0.022), (r"deberta[-_]?v?\d?[-_]?small", 0.044),
    (r"deberta[-_]?v?\d?[-_]?base", 0.086), (r"deberta[-_]?v?\d?[-_]?large", 0.304),
    (r"roberta[-_]?large", 0.355), (r"roberta[-_]?base", 0.125),
    (r"distilbert", 0.066), (r"albert", 0.012),
    (r"bert[-_]?large", 0.34), (r"bert[-_]?base", 0.11), (r"electra[-_]?base", 0.11),
    (r"(flan[-_]?)?t5[-_]?xxl", 11.0), (r"(flan[-_]?)?t5[-_]?xl", 3.0),
    (r"(flan[-_]?)?t5[-_]?large", 0.77), (r"(flan[-_]?)?t5[-_]?base", 0.22),
    (r"(flan[-_]?)?t5[-_]?small", 0.06),
    (r"whisper[-_]?large", 1.55), (r"whisper[-_]?medium", 0.77),
    (r"whisper[-_]?small", 0.24), (r"whisper[-_]?base", 0.074), (r"whisper[-_]?tiny", 0.039),
    (r"resnet[-_]?152", 0.06), (r"resnet[-_]?101", 0.044), (r"resnet[-_]?50", 0.026),
    (r"convnext[-_]?large", 0.198), (r"convnext[-_]?base", 0.089),
    (r"swin[-_]?large", 0.197), (r"swin[-_]?base", 0.088),
    (r"clip[-_]?vit[-_]?l", 0.428), (r"clip[-_]?vit[-_]?b", 0.15),
]


def model_size_b(model_name: str):
    """Parameter count in billions, or None if it cannot be determined."""
    s = model_name.split("/")[-1]
    m = re.search(r"(?<![\d.])(\d+)b(\d)(?![\d])", s, re.I)          # 7b1 -> 7.1
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*b\b", s, re.I)       # 7B / 0.5B
    if m:
        return float(m.group(1))
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*m\b", s, re.I)       # 500m -> 0.5
    if m:
        return float(m.group(1)) / 1000.0
    low = s.lower()
    for pattern, size in _FAMILY_SIZES:
        if re.search(pattern, low):
            return size
    return None


def passes_max_params(model_name: str, max_b: float) -> bool:
    """True if the model is known to be <= max_b billions of parameters."""
    size = model_size_b(model_name)
    return size is not None and size <= max_b
