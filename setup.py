from setuptools import setup
import codecs


with codecs.open('README.rst', encoding='utf-8') as readme_file:
    long_description = readme_file.read()

setup(
    name="hep_ml",
    version='0.4.0',
    description="Machine Learning for High Energy Physics",
    long_description=long_description,

    url='https://github.com/arogozhnikov/hep_ml',

    # Author details
    author='Alex Rogozhnikov',
    author_email='axelr@yandex-team.ru',

    # Choose your license
    license='Apache 2.0',
    packages=['hep_ml', 'hep_ml.experiments'],

    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7 ',
        'Programming Language :: Python :: 3.4 ',
    ],

    # What does your project relate to?
    keywords='machine learning, supervised learning, '
             'uncorrelated methods of machine learning, high energy physics, particle physics',

    # List run-time dependencies here. These will be installed by pip when your project is installed.
    install_requires=[
        'numpy >= 1.9',
        'scipy >= 0.15.0',
        'pandas >= 0.14.0',
        'scikit-learn >= 0.15.2',
        'theano >= 0.7',
        'six',
    ],
)