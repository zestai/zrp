import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = open((HERE / "README.rst"),  encoding="utf-8")

# The contents of the requirements.txt
with open("requirements.txt", 'r') as file:
    requirements_raw = file.readlines()

requirements = []
for r in requirements_raw:
    requirements.append(r.replace('\n', ''))

# This call to setup() does all the work
setup(
    name="zrp",
    version="0.2.0",
    description="The Zest Race Predictor tool predicts race/ethnicity using a name and address as inputs.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/zestai/zrp",
    author="Kasey Matthews et al.",
    author_email="abetterway@zest.ai",
    license="Apache-2.0",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    keywords="race ethnicity names address acs geocode",
    packages=find_packages(exclude=("extras")),
    package_data={
        "zrp": [
        ],
    },
    include_package_data=True,
    install_requires=requirements, 
    entry_points={
        "console_scripts": [

        ]
    },
) 
