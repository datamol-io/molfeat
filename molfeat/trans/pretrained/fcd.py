import numpy as np

from molfeat.trans.pretrained.base import PretrainedMolTransformer
from molfeat.utils import requires

if requires.check("fcd_torch"):
    from fcd_torch import FCD


class FCDTransformer(PretrainedMolTransformer):
    r"""
    FCD transformer based on the ChemNet pretrained model

    Attributes:
        featurizer (FCD): FCD featurizer object
        dtype (type, optional): Data type. Use call instead
    """

    def __init__(self, n_jobs=1, dtype=np.float32, **params):
        super().__init__(dtype=dtype, **params)
        if not requires.check("fcd_torch"):
            raise ImportError(
                "`fcd_torch` is not available, please install it `conda install -c conda-forge fcd_torch'`"
            )

        self.n_jobs = n_jobs
        self.featurizer = FCD(n_jobs=n_jobs)

    def _embed(self, smiles, **kwargs):
        """Compute embedding"""
        return self.featurizer.get_predictions(smiles)

    def _update_params(self):
        super()._update_params()
        self.featurizer = FCD(n_jobs=self.n_jobs)
