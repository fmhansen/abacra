"""This is setup.py of abacra"""

from setuptools import setup

# for developers: recommended way of installing is to run in this directory
# pip install -e .
# This creates a link insteaed of copying the files,
# so modifications in this directory are
# modifications in the installed package.

setup(name="abacra",
      version="0.1",
      description="to be added",
      url="to be added",
      author="Finn Müller-Hansen",
      author_email="mhansen@pik-potsdam.de",
      license="MIT",
      packages=["abacra"],
      install_requires=[
            "numpy>=1.11.0",
            "matplotlib>=2.0.0",
            "pandas>=0.19.0",
            "networkx>=2.0",
            "pyshp>=1.2.0",
            # "scipy>=0.17.0",
            # "sympy>=1.0",
            # "mpi4py>=2.0.0",
      ],
      # see http://stackoverflow.com/questions/15869473/
      # what-is-the-advantage-of-setting-zip-safe-to-true-when-packaging-a-python-projec
      zip_safe=False
      )
