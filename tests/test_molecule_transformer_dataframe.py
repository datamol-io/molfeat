import pytest

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from molfeat.trans.base import MoleculeTransformer


@pytest.fixture
def smiles():
    return [
        "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C(=O)CO1)O)C)O)C",
        "CN(CCOC(c1ccccc1)c1ccccc1)C",
        "O/N=C(/c1csc(n1)N)\C(=O)N[C@@H]1C(=O)N2[C@@H]1SCC(=C2C(=O)O)C=C",
        "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
    ]


@pytest.fixture(params=["list", "series", "dataframe"])
def mols(request, smiles):
    if request.param == "list":
        return smiles
    elif request.param == "series":
        return pd.Series(smiles, name="smiles")
    elif request.param == "dataframe":
        return pd.DataFrame({"smiles": smiles, "column_2": [1, 0, 1, 1]})


def test_list_series_dataframe(mols):
    transformer_ecfp = MoleculeTransformer(featurizer="ecfp", dtype=float)
    results = transformer_ecfp.fit_transform(mols)

    assert results.shape == (4, 2048)
    assert isinstance(results, np.ndarray)


def test_with_pipeline_column_transformer(smiles):
    # setup data
    mols = pd.DataFrame({"smiles": smiles, "column_2": [1, 0, 1, 1]})

    # setup pipeline
    transformer_ecfp = MoleculeTransformer(featurizer="ecfp", dtype=float)
    column_preprocessor = ColumnTransformer(
        transformers=[
            ("ecfp_trans", transformer_ecfp, ["smiles"]),
            ("column_2", "passthrough", ["column_2"]),
        ]
    )

    pipeline = Pipeline([("preprocess", column_preprocessor), ("classifier", GaussianNB())])

    # fit/predict pipeline
    pipeline.fit(mols, [1, 0, 1, 0])
    r = pipeline.predict(mols)

    # tests
    expect = np.ndarray
    assert isinstance(r, expect)

    expect = (4,)
    assert r.shape == expect
