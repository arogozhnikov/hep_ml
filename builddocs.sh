#!/usr/bin/env bash
# script should be sourced from repo root
# documentation is served from docs/ folder, not separate branch

pip install sphinx_rtd_theme
pip install -e .
pushd docsrc
# Forcing to ignore caches
make SPHINXOPTS="-E" html
popd

rm -r docs
mkdir -p docs
cp -a docsrc/_build/html/* docs
touch docs/.nojekyll
