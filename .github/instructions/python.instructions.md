
---
applyTo: "**/*.py"
---

# Python Development Instructions

- Use Python 3.10 or later.
- Follow PEP 8.
- Use type hints for public functions.
- Add docstrings to public functions and classes.
- Prefer pathlib over manually constructed paths.
- Use logging instead of print for pipeline diagnostics.
- Use explicit exceptions for invalid data.
- Avoid global mutable state.
- Keep functions small and independently testable.
- Do not hardcode local absolute paths.
- Read configuration from YAML files.
- Use NumPy arrays for numerical operations.
- Use h5py for HDF5 access.
- Use pytest for unit tests.
- Do not silently suppress exceptions.
- Make file writes atomic where practical.
