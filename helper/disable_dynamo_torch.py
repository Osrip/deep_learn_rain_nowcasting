# Must be imported before ANY torch imports
import os
import sys

# Set environment variables to completely disable torch._dynamo
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["PYTHONTRACEMALLOC"] = "0"

print("PyTorch dynamo completely disabled for debugging")