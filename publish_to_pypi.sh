#!/usr/bin/env bash
# simple script to upload new versions to PyPI

pip install wheel
rm dist/*
python setup.py sdist
python setup.py bdist_wheel --universal
twine upload -r pypi dist/* 
