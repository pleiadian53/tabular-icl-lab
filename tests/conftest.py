"""Test-session settings.

TabPFN's gated weights (2.5 and later) open a browser to accept the license on
first use. Tests must never block on that, so the browser flow is disabled here
and a missing license makes the gated case skip. Accept the license once outside
the tests (see the README), or set ``TABPFN_TOKEN``.
"""

import os

os.environ.setdefault("TABPFN_NO_BROWSER", "1")
