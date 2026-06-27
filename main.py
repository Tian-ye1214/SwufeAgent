import os
import sys

if getattr(sys, "frozen", False):
    os.environ["LOGFIRE_PYDANTIC_RECORD"] = "off"

_root = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from redlotus.agent_core.entrypoint import main

if __name__ == "__main__":
    main()