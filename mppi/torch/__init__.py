import sys

if sys.version_info >= (3, 9):
    raise RuntimeError(
        "Torch backend currently supports only Python 3.8. "
        "Please use Python 3.8 or install the JAX backend."
    )