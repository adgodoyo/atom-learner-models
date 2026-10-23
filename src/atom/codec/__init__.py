"""Codec: semantic encoder (mid-stack states -> strip dominant components -> whiten) and probe panel.

`transform` (the deflate+whiten codec) is import-light; `backbone` is imported lazily by
callers that actually need the model, so `import atom.codec` never pulls transformers.
"""

from atom.codec.transform import CODEC_VERSION, CodecTransform

__all__ = ["CodecTransform", "CODEC_VERSION"]
