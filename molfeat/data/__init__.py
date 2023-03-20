import importlib.resources as pkg_resources
import pandas as pd


def get_df(name: str):
    """Get a static dataframe file in this folder

    Args:
        name: name of the data to get

    Returns:
        data (pd.DataFrame): Loaded data
    """
    if not name.endswith(".xz"):
        name = f"{name}.xz"
    with pkg_resources.open_binary("molfeat.data", name) as instream:
        data = pd.read_pickle(instream, compression="xz")
    return data
