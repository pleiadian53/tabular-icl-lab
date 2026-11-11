"""tabicllab: tabular foundation models and in-context learning for computational biology.

The lab uses TabPFN as a library, never as a fork. Code here builds the tables
an in-context learner sees (features and targets for each application), runs
the learner, and evaluates it against the references each application already
has. Applications live under ``examples/``; reusable pieces live here.
"""

__version__ = "0.1.0"
