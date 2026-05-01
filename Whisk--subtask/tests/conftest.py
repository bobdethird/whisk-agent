"""
Add the whisk_agent package root to sys.path so that tests can import
from perception, arm, agent, etc. regardless of how pytest is invoked.
"""

import os
import sys

# Insert whisk_agent/ (the parent of this tests/ directory) at the front
# of sys.path so package-relative imports resolve correctly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
