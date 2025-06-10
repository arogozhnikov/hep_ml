#!/usr/bin/env bash
# simple script to upload new versions to PyPI

rm -rf build/*
rm -rf dist/*
python3 -m pip install --upgrade build twine
python3 -m build
python3 -m twine upload dist/*
