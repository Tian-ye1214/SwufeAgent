import os
import sys
from agent_app import main

if getattr(sys, "frozen", False):
    os.environ["LOGFIRE_PYDANTIC_RECORD"] = "off"

if __name__ == "__main__":
    main()