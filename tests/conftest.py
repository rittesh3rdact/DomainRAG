import os

# Importing app.config instantiates a module-level Settings() that requires
# GOOGLE_API_KEY. Provide a dummy value for tests so collection doesn't fail
# for a real key that only matters at runtime.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
