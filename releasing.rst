Preparing A New Release
_______________________

Observe the following steps in order to safely and correctly prepare and push new Pypi, and Github releases for the zrp packages. To ensure consistency, we ask you push updates to Pypi, AND Github releases.

Pypi 
====
(`Reference <https://widdowquinn.github.io/coding/update-pypi-package/>`_)

1. Once you've updated the package, ensure you have an up to date local repo and push/merge all commits to Github
2. Incremenet the version number for the package
  We use the tool `Bump Version <https://pypi.org/project/bumpversion/>`_ to ensure all version numbers are kept consistent. 
  
  You can install Bumpversion from PyPI:
::

  $ pip install bumpversion
   
To increment the MINOR version of reader, for example you would do something like this:
::

$ bumpversion --current-version 0.1.0 minor setup.py zrp/about.py

3. Update local packages for distribution
::

  python -m pip install --user --upgrade setuptools wheel
  python -m pip install --user --upgrade twine

4. Create distribution packages on your local machine, and check the dist/ directory for the new version files
::

  python setup.py sdist bdist_wheel
  ls dist


5. Remove the old package version distribution packages in '/dist'



6. Upload the distribution files to pypi’s test server
 ::
 
  python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

* Check the upload on the test.pypi server [https://test.pypi.org/project/PACKAGE/VERSION/]
  
7. Test the upload with a local installation
::
 
  python -m pip install --index-url https://test.pypi.org/simple/ --no-deps <PACKAGE>
  
* Start Python, import the package, and test the version

8. Upload the distribution files to pypi
 ::
 
  python -m twine upload dist/*
  
* Check the upload at pypi [https://pypi.org/project///](https://pypi.org/project///)


Github Releases
===============

1. On the repo home Github page, click "Create a new release"

2. Enter the new relesae title in the following format: "zrp" + "-" + "VERSION"

* Ex.: "zrp-0.1.0"

3. Enter in any details describing the release

4. Publish Release
