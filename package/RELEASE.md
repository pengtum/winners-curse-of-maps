# Releasing mapselect

The licence is MIT (`LICENSE`); the home page in `pyproject.toml` is the GitHub repository (https://github.com/pengtum/winners-curse-of-maps). Before the first
release, confirm that the name mapselect is still free on PyPI.

Run from this folder. PyPI needs your own account and an API token; twine asks for the token when you upload.

```bash
python -m pip install build twine
python -m pytest tests
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mapselect
python examples/quickstart.py
python -m twine upload dist/*
```

A version on PyPI cannot be replaced: to fix a release, raise the version in `pyproject.toml` and `src/mapselect/__init__.py`, and upload again.
