"""Index structures: positional, zone, facet, tiered, and on-disk storage."""

import os as _os

if _os.environ.get("KB_DEV_SHIM") == "1":
    # TEMPORARY: fills in index methods that still raise NotImplementedError (see dev/reference_index.py)
    from kanoon_bridge.dev.reference_index import install as _install

    _install()
