import shutil
import zipfile
import os
import sys
from urllib.request import urlretrieve
from tqdm import tqdm
from zrp import about


# This is used to show progress when downloading.
# see here: https://github.com/tqdm/tqdm#hooks-and-callbacks
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_progress(url, fname):
    """
    Download a file and show a progress bar.
    :param url: A string for the url of the release zip to download.
    :param fname: A string for the local file name under which the downloaded file can be found.
    :return:
    """

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
              desc=url.split('/')[-1]) as t:  # all optional kwargs
        urlretrieve(url, filename=fname, reporthook=t.update_to, data=None)
        t.total = t.n
    return fname


def download_and_clean(url, release_zip_fname):
    """
    Download look up tables and file them within the module.
    This downloads the zip file from the source, extracts it, renames the moves the
    tables to the correct directory, and removes large files not used at runtime.
    :param url: A string for the url of the release zip to download.
    :param release_zip_fname: A string for the name of the zip file downloaded.
    :return:
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(cwd, release_zip_fname)
    print("Downloading zrp extras...", file=sys.stderr)
    download_progress(url, fname)
    print("Finished download.")
    print("\n")

    print("Filing extras...")
    with zipfile.ZipFile(fname, 'r') as zf:
        zf.extractall(cwd)
    os.remove(fname)

    # Clear old look up table directories
    data_dir = os.path.join(cwd, 'data')
    geo_data_dir = os.path.join(data_dir, 'geo')
    acs_data_dir = os.path.join(data_dir, 'acs')
    if os.path.isdir(geo_data_dir):
        shutil.rmtree(geo_data_dir)
    if os.path.isdir(acs_data_dir):
        shutil.rmtree(acs_data_dir)

    # Migrate lookup tables
    dl_folder = release_zip_fname.split(".zip")[0]
    dl_geo_dir = os.path.join(cwd, dl_folder, 'extras', 'geo')
    dl_acs_dir = os.path.join(cwd, dl_folder, 'extras', 'acs')
    shutil.move(dl_geo_dir, geo_data_dir)
    shutil.move(dl_acs_dir, acs_data_dir)

    # Remove rest of release folder
    shutil.rmtree(dl_folder)

    # save a version file so we can tell what it is
    vpath = os.path.join(data_dir, 'version')
    with open(vpath, 'w') as vfile:
        vfile.write('zrp release --> {}'.format(dl_folder))

    print("Filed zrp extras", file=sys.stderr)


def get_release():
    version = about.__version__
    dl_tpl = "{m}-{v}"
    return dl_tpl.format(m="zrp", v=version)


def download():
    release = get_release() + ".zip"
    url = about.__download_url_prefix__ + release
    download_and_clean(url, release)