#!/usr/bin/env bash
rm dist/*
python setup.py sdist
python setup.py bdist_wheel --universal
twine upload -r pypi dist/* 