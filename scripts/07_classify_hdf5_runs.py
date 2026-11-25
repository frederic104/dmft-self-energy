"""Compatibility entry point for ``python -m dmft_data storage``."""

import sys

from dmft_data.__main__ import main

if __name__ == "__main__":
    main(["storage", *sys.argv[1:]])
