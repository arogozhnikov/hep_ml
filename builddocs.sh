#!/usr/bin/env bash
# script should be running from its directory
# assumed that in ../hepml_doc there is another clone of repository
# with gh-pages branch active


cd docs
# Forcing to ignore caches
make SPHINXOPTS="-E" html
cd ..

# deleting everythong for the exception
# of .git/ and .gitignore
mv ../hep_mldoc/.git/ /tmp/githepmldoc
mv ../hep_mldoc/.gitignore /tmp/gitignorehepmldoc
rm -r ../hep_mldoc/*
mv /tmp/githepmldoc ../hep_mldoc/.git/
mv /tmp/gitignorehepmldoc ../hep_mldoc/.gitignore

# copying new files to hep_mldoc
rsync -avh docs/_build/html/* ../hep_mldoc

cd ../hep_mldoc
touch .nojekyll

git add .
git commit -am 'autoupdate of documentation'
git push origin gh-pages

# return
cd ../hep_ml
