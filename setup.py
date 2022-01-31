import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.rst").read_text()

# This call to setup() does all the work
setup(
    name="zrp",
    version="0.1.0",
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
        "License :: OSI Approved :: Apache-2.0 License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    keywords="race ethnicity names address acs geocode",
    packages=["zrp"],
    package_data={ #TODO
        "ethnicolr": [
            "data/census/census_2000.csv",
            "data/census/census_2010.csv",
            "data/census/readme.md",
            "data/census/*.pdf",
            "data/census/*.R",
            "data/wiki/*.*",
            "models/*.ipynb",
            "models/*.md",
            "models/census/lstm/*.h5",
            "models/census/lstm/*.csv",
            "models/wiki/lstm/*.h5",
            "models/wiki/lstm/*.csv",
            "models/fl_voter_reg/lstm/*.h5",
            "models/fl_voter_reg/lstm/*.csv",
            "models/nc_voter_reg/lstm/*.h5",
            "models/nc_voter_reg/lstm/*.csv",
            "data/input*.csv",
            "examples/*.ipynb",
        ],
    },
    include_package_data=True,
    install_requires=["feedparser", "html2text"],  # TODO
    entry_points={
        "console_scripts": [
            "realpython=reader.__main__:main",
            # TODO: Once we know what endpoints are being exposed (see ethnicolr's as example)
        ]
    },
) #TODO: Update Manifest.in file to include data and other files as suggested here: https://realpython.com/pypi-publish-python-package/#adding-files-to-your-package
