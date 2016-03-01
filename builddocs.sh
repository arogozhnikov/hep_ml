#!/usr/bin/env bash
# assuming that in ../hepml_doc there is another clone of repository


cd docs
# Forcing to ignore caches
make SPHINXOPTS="-E" html
cd ..
# TODO delete there everything for the except of git folder
# currently we simply copy, while leaving everythong there
rsync -avh docs/_build/html/* ../hep_mldoc

cd ../hep_mldoc
touch .nojekyll

git add .
git commit -am 'autoupdate of documentation'
git push origin gh-pages

# return back
cd ../hep_ml