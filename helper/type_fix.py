"""
Compatibility fix for typing.Self issues in Python 3.10 with PyTorch
This resolves the "Plain typing.Self is not valid as type argument" error
"""
import sys
import types
import typing
from typing import TypeVar, Any

# Create a compatibility version of Self
Self = TypeVar('Self', bound=Any)


def apply_typing_fixes():
    """Apply monkey patch to fix typing issues with Self in Union types"""
    # Only needed for Python 3.10
    if sys.version_info < (3, 11):
        # First attempt: patch typing.Self
        typing.Self = Self

        # Second approach: monkey patch the problematic class
        # The error happens in torch._dynamo.variables.lazy.LazyVariableTracker
        if 'torch._dynamo.variables.lazy' in sys.modules:
            lazy_module = sys.modules['torch._dynamo.variables.lazy']
            if hasattr(lazy_module, 'LazyVariableTracker'):
                # Get the original unwrap method
                cls = lazy_module.LazyVariableTracker
                if hasattr(cls, 'unwrap'):
                    original_unwrap = cls.unwrap

                    # Replace the method with a version that doesn't use Union[..., Self]
                    def patched_unwrap(self):
                        return original_unwrap(self)

                    cls.unwrap = patched_unwrap