import shutil
import zipfile
import os
import sys
import warnings
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
        print(f"Retrieving url at {url}")
        urlretrieve(url, filename=fname, reporthook=t.update_to, data=None)
        t.total = t.n
    return fname


def download_and_clean(url, release_pkg_fname, release_zip_fname, geo_yr="2019", acs_yr="2019", acs_range="5yr"):
    """
    Download look up tables and file them within the module.
    This downloads the zip file from the source, extracts it, renames the moves the
    tables to the correct directory, and removes large files not used at runtime.
    :param acs_range: A string for the year range the acs lookup table data will be from.
    :param acs_yr: A string for the year the acs lookup table data will be from.
    :param geo_yr: A string for the year the geo lookup table data will be from.
    :param url: A string for the url of the release zip to download.
    :param release_zip_fname: A string for the name of the zip file downloaded.
    :return:
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    path_release_zip_fname = os.path.join(cwd, release_zip_fname)
    print("Downloading zrp release...", file=sys.stderr)
    download_progress(url, path_release_zip_fname)
    print("Finished download.")
    print("\n")

    print("Filing extras...")
    with zipfile.ZipFile(path_release_zip_fname, 'r') as zf:
        zf.extractall(cwd)
    os.remove(path_release_zip_fname)

    # Get rid of prefix that unzipping prepends
    curr_folder = cwd.split("/")[-1]
    extracted_src_fname = curr_folder + "-" + release_pkg_fname
    path_extracted_src_fname = os.path.join(cwd, extracted_src_fname)
    path_release_pkg_fname = os.path.join(cwd, release_pkg_fname)
    os.rename(path_extracted_src_fname, path_release_pkg_fname)

    # Clear old look up table directories
    data_dir = os.path.join(cwd, 'data')
    geo_data_dir = os.path.join(data_dir, f'processed/geo/{geo_yr}')
    acs_data_dir = os.path.join(data_dir, f'processed/acs/{acs_yr}/{acs_range}')
    if os.path.isdir(geo_data_dir):
        shutil.rmtree(geo_data_dir)
    if os.path.isdir(acs_data_dir):
        shutil.rmtree(acs_data_dir)
    print("Old geo and acs lookup table data cleared out.")

    # Migrate lookup tables
    dl_geo_dir = os.path.join(cwd, release_pkg_fname, f'extras/processed/geo/{geo_yr}')
    dl_acs_dir = os.path.join(cwd, release_pkg_fname, f'extras/processed/acs/{acs_yr}/{acs_range}')
    if os.path.isdir(dl_geo_dir):
        shutil.move(dl_geo_dir, geo_data_dir)
        print("New geo lookup tables successfully migrated.")
    else:
        warnings.warn(f"The geo lookup data was not found in {dl_geo_dir}. Ensure you're requesting a valid year. "
                      "Consult the zrp release to troubleshoot.")
    if os.path.isdir(dl_acs_dir):
        shutil.move(dl_acs_dir, acs_data_dir)
        print("New acs lookup tables successfully migrated.")
    else:
        warnings.warn(f"The acs lookup table was not found in {dl_acs_dir}. Ensure you're requesting a valid year and/or"
                      f"year range. Consult the zrp release to troubleshoot.")

    # Remove rest of release folder
    shutil.rmtree(path_release_pkg_fname)

    # save a version file so we can tell what it is
    vpath = os.path.join(data_dir, 'version')
    with open(vpath, 'w') as vfile:
        vfile.write('zrp release --> {}'.format(release_pkg_fname))

    print("Filed zrp extras successfully.", file=sys.stderr)


def get_release():
    version = about.__version__
    dl_tpl = "{m}-{v}"
    return dl_tpl.format(m="zrp", v=version)


def download():
    release_pkg = get_release()
    release_pkg_zip = release_pkg + ".zip"
    url = about.__download_url_prefix__ + release_pkg_zip
    download_and_clean(url, release_pkg, release_pkg_zip)