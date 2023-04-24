from typing import Union

import datamol as dm
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder


def MHFP(
    x: Union[dm.Mol, str],
    n_permutations: int = 128,
    radius: int = 3,
    min_radius: int = 1,
    rings: bool = True,
    isomeric: bool = False,
    kekulize: bool = False,
    seed: int = 0,
    **kwargs,
):
    """Compute Unfolded MHFP fingerprint from rdkit

    Args:
        x: input molecule
        n_permutations (int, optional): Number of permutations of the fp. Defaults to 512.
        radius (int, optional): Radius of the fingerprint. Defaults to 3.
        min_radius (int, optional): Minimum radius to consider. Defaults to 1.
        rings (bool, optional): Whether to consider rings
        isomeric (bool, optional): Whether to use isomeric variant. Defaults to False.
        kekulize (bool, optional): Whether to kekulize molecule first. Defaults to False.
        seed (int, optional): optional seed to use. Default to 0

    Returns:
        fp: fingerprint
    """
    if isinstance(x, str):
        x = dm.to_mol(x)
    mhfp_encoder = MHFPEncoder(n_permutations, seed)
    encoded_fp = mhfp_encoder.EncodeMol(
        x,
        radius=radius,
        rings=rings,
        isomeric=isomeric,
        kekulize=kekulize,
        min_radius=min_radius,
    )
    return encoded_fp


def SECFP(
    x: Union[dm.Mol, str],
    n_permutations: int = 128,
    nBits: int = 2048,
    radius: int = 3,
    min_radius: int = 1,
    rings: bool = True,
    isomeric: bool = False,
    kekulize: bool = False,
    seed: int = 0,
    **kwargs,
):
    """Compute SECFP (folded MHFP) fingerprint

    Args:
        x: input molecule
        n_permutations (int, optional): Number of permutations of the fp. Defaults to 512.
        nBits (int, optional): Length of the fingerprint. Defaults to 2048.
            We use nBits to be consistent with other fingerprints.
        radius (int, optional): Radius of the fingerprint. Defaults to 3.
        min_radius (int, optional): Minimum radius to consider. Defaults to 1.
        rings (bool, optional): Whether to consider rings
        isomeric (bool, optional): Whether to use isomeric variant. Defaults to False.
        kekulize (bool, optional): Whether to kekulize molecule first. Defaults to False.
        seed (int, optional): optional seed to use. Default to 0

    Returns:
        fp: fingerprint
    """
    if isinstance(x, str):
        x = dm.to_mol(x)
    mhfp_encoder = MHFPEncoder(n_permutations, seed)
    encoded_fp = mhfp_encoder.EncodeSECFPMol(
        x,
        radius=radius,
        rings=rings,
        isomeric=isomeric,
        kekulize=kekulize,
        min_radius=min_radius,
        length=nBits,
    )
    return encoded_fp
