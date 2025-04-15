# disable_dynamo.py
import sys
import types
import typing


# Check if we're in debug mode
def is_debugger_active():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    else:
        return gettrace() is not None


# Only apply the patch in debug mode
if is_debugger_active():
    # Add a 'Self' type to typing if it doesn't exist
    if not hasattr(typing, 'Self'):
        # Create a simple Self type
        class _SelfType:
            def __repr__(self):
                return "typing.Self"


        typing.Self = _SelfType()

        # Patch _type_check to accept our Self type
        original_type_check = typing._type_check


        def patched_type_check(arg, msg):
            if arg is typing.Self:
                return arg
            return original_type_check(arg, msg)


        typing._type_check = patched_type_check

    print("PyTorch typing.Self patched for debugging")