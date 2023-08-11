"""
Common cache for pipeline
"""

import joblib

cache = joblib.Memory("__mcpipeline_cache__")
