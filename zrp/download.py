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


def download_and_clean_lookup_tables(url, lookup_tables_output_fname, lookup_tables_output_zip_fname, geo_yr="2019",
                                     acs_yr="2019", acs_range="5yr"):
    """
    Downloads look up tables and file them within the module.
    This downloads the zip file from the repository, extracts it, renames it, then moves the
    tables to the correct directory, and removes large files not used at runtime.
    :param lookup_tables_output_fname: A string for the name of the file downloaded after unzipping.
    :param acs_range: A string for the year range the acs lookup table data will be from.
    :param acs_yr: A string for the year the acs lookup table data will be from.
    :param geo_yr: A string for the year the geo lookup table data will be from.
    :param url: A string for the url of the release zip to download.
    :param lookup_tables_output_zip_fname: A string for the name of the zip file downloaded.
    :return:
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    path_to_lt_zip = os.path.join(cwd, lookup_tables_output_zip_fname)
    print("Downloading zrp release...", file=sys.stderr)
    download_progress(url, path_to_lt_zip)
    print("Finished download.")
    print("\n")

    print("Filing extras...")
    with zipfile.ZipFile(path_to_lt_zip, 'r') as zf:
        zf.extractall(cwd)
    os.remove(path_to_lt_zip)

    # Get rid of prefix that unzipping prepends
    # curr_folder = cwd.split("/")[-1]
    # unzipped_src_fname = curr_folder + "-" + lookup_tables_output_fname
    # path_to_unzipped_src = os.path.join(cwd, unzipped_src_fname)
    path_to_lookup_tables = os.path.join(cwd, lookup_tables_output_fname)
    # os.rename(path_to_unzipped_src, path_to_lookup_tables)

    # Clear old look up table directories
    data_dir = os.path.join(cwd, 'data')
    geo_data_dir = os.path.join(data_dir, f'processed/geo/{geo_yr}')
    acs_data_dir = os.path.join(data_dir, f'processed/acs/{acs_yr}/{acs_range}')

    if os.path.isdir(geo_data_dir):
        shutil.rmtree(geo_data_dir)
    if os.path.isdir(acs_data_dir):
        shutil.rmtree(acs_data_dir)

    print("Old geo lookup table data cleared out.")

    # Migrate lookup tables
    dl_geo_dir = os.path.join(cwd, lookup_tables_output_fname, f'geo/{geo_yr}')
    dl_acs_dir = os.path.join(cwd, lookup_tables_output_fname, f'acs/{acs_yr}/{acs_range}')

    if os.path.isdir(dl_geo_dir):
        shutil.move(dl_geo_dir, geo_data_dir)
        print("New geo lookup tables successfully migrated.")
    else:
        warnings.warn(f"The geo lookup data was not found in {dl_geo_dir}. Ensure you're requesting a valid year. "
                      "Consult the lookup_tables release to troubleshoot.")
    if os.path.isdir(dl_acs_dir):
        shutil.move(dl_acs_dir, acs_data_dir)
    else:
        warnings.warn(f"The acs lookup data was not found in {dl_acs_dir}. Ensure you're requesting a valid year and"
                      "year range. Consult the lookup_tables release to troubleshoot.")

    # Remove rest of lookup table folder
    shutil.rmtree(path_to_lookup_tables)

    # save a version file so we can tell what it is
    vpath = os.path.join(data_dir, 'version')
    with open(vpath, 'w') as vfile:
        vfile.write('zrp release --> {}'.format(lookup_tables_output_fname))

    print("Filed lookup tables successfully.", file=sys.stderr)


def download_and_clean_pipelines(url, pipelines_output_fname, pipelines_output_zip_fname):
    """
    Downloads pipeline pickle files and file them within the module.
    This downloads the zip file from the repository, extracts it, renames it, then moves the
    tables to the correct directory, and removes large files not used at runtime.
    :param pipeline_output_fname: A string for the name of the file downloaded after unzipping.
    :param url: A string for the url of the release zip to download.
    :param pipelines_output_zip_fname: A string for the name of the zip file downloaded.
    :return:
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    path_to_ppln_zip = os.path.join(cwd, pipelines_output_zip_fname)
    print("Downloading zrp release...", file=sys.stderr)
    download_progress(url, path_to_ppln_zip)
    print("Finished download.")
    print("\n")

    print("Filing extras...")
    with zipfile.ZipFile(path_to_ppln_zip, 'r') as zf:
        zf.extractall(cwd)
    os.remove(path_to_ppln_zip)

    # Get rid of prefix that unzipping prepends
    # curr_folder = cwd.split("/")[-1]
    # unzipped_src_fname = curr_folder + "-" + pipelines_output_fname
    # path_to_unzipped_src = os.path.join(cwd, unzipped_src_fname)
    path_to_pipelines = os.path.join(cwd, pipelines_output_fname)
    # os.rename(path_to_unzipped_src, path_to_pipelines)

    # Clear old look up table directories
    model_dir = os.path.join(cwd, 'modeling/models')
    block_group_dir = os.path.join(model_dir, 'block_group')
    census_tract_dir = os.path.join(model_dir, 'census_tract')
    zip_code_dir = os.path.join(model_dir, 'zip_code')

    block_group_pipeline = os.path.join(block_group_dir, 'pipe.pkl')
    census_tract_pipeline = os.path.join(census_tract_dir, 'pipe.pkl')
    zip_code_pipeline = os.path.join(zip_code_dir, 'pipe.pkl')

    if os.path.isfile(block_group_pipeline):
        os.remove(block_group_pipeline)
    if os.path.isfile(census_tract_pipeline):
        os.remove(census_tract_pipeline)
    if os.path.isfile(zip_code_pipeline):
        os.remove(zip_code_pipeline)
    print("Old pipelines cleared out.")

    # Migrate pipelines
    dl_bg_pipe_file = os.path.join(path_to_pipelines, 'block_group_pipe.pkl')
    dl_ct_pipe_file = os.path.join(path_to_pipelines, 'census_tract_pipe.pkl')
    dl_zp_pipe_file = os.path.join(path_to_pipelines, 'zip_code_pipe.pkl')

    if os.path.isfile(dl_bg_pipe_file):
        shutil.move(dl_bg_pipe_file, os.path.join(block_group_dir, 'pipe.pkl'))
        print("Block group pipeline successfully migrated.")
    else:
        warnings.warn(f"The block group pipeline was not found in {dl_bg_pipe_file}."
                      "Consult the pipelines release to troubleshoot.")

    if os.path.isfile(dl_ct_pipe_file):
        shutil.move(dl_ct_pipe_file, os.path.join(census_tract_dir, 'pipe.pkl'))
        print("Census tract pipeline successfully migrated.")
    else:
        warnings.warn(f"The census tract pipeline was not found in {dl_ct_pipe_file}."
                      "Consult the pipelines release to troubleshoot.")

    if os.path.isfile(dl_zp_pipe_file):
        shutil.move(dl_zp_pipe_file, os.path.join(zip_code_dir, 'pipe.pkl'))
        print("Zip code pipeline successfully migrated.")
    else:
        warnings.warn(f"The zip code pipeline was not found in {dl_zp_pipe_file}."
                      "Consult the pipelines release to troubleshoot.")

    # Remove rest of pipelines folder
    shutil.rmtree(path_to_pipelines)

    # save a version file so we can tell what it is
    data_dir = os.path.join(cwd, 'data')
    vpath = os.path.join(data_dir, 'version')
    with open(vpath, 'w') as vfile:
        vfile.write('zrp release --> {}'.format(pipelines_output_fname))

    print("Filed pipelines successfully.", file=sys.stderr)


def get_release():
    version = about.__version__
    dl_tpl = "{m}-{v}"
    return dl_tpl.format(m="zrp", v=version)


def download():
    release_pkg = get_release()

    # lookup_tables_output_fname = release_pkg + "_lookup_tables"
    lookup_tables_output_fname = "lookup_tables"
    lookup_tables_output_zip_fname = release_pkg + "_lookup_tables" + ".zip"
    lookup_table_url = about.__download_url_prefix__ + release_pkg + "/lookup_tables.zip"
    download_and_clean_lookup_tables(lookup_table_url, lookup_tables_output_fname, lookup_tables_output_zip_fname)

    pipelines_output_fname = "pipelines"
    pipelines_output_zip_fname = release_pkg + "_pipelines" + ".zip"
    pipelines_url = about.__download_url_prefix__ + release_pkg + "/pipelines.zip"
    download_and_clean_pipelines(pipelines_url, pipelines_output_fname, pipelines_output_zip_fname)
