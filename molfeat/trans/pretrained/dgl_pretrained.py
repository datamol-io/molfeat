from typing import List
from typing import Union
from typing import Callable
from typing import Optional

import os
import re
import tempfile
import fsspec
import joblib
import torch
import numpy as np
import datamol as dm
from loguru import logger
from torch.utils.data import DataLoader

from molfeat.utils.commons import _parse_to_evaluable_str
from molfeat.store.loader import PretrainedStoreModel
from molfeat.trans.pretrained.base import PretrainedMolTransformer
from molfeat.utils import requires
from molfeat.store import ModelStore

if requires.check("dgl") and requires.check("dgllife"):
    import dgl
    from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling

    from dgllife.utils import (
        mol_to_bigraph,
        PretrainAtomFeaturizer,
        PretrainBondFeaturizer,
        JTVAEVocab,
    )
    from dgllife.data import JTVAEDataset, JTVAECollator


class DGLModel(PretrainedStoreModel):
    r"""
    Load one of the pretrained DGL models for molecular embedding:
    """
    AVAILABLE_MODELS = [
        "gin_supervised_contextpred",
        "gin_supervised_infomax",
        "gin_supervised_edgepred",
        "gin_supervised_masking",
        "jtvae_zinc_no_kl",
    ]

    def __init__(
        self,
        name: str,
        cache_path: Optional[os.PathLike] = None,
        store: Optional[ModelStore] = None,
    ):
        super().__init__(name, cache_path=cache_path, store=store)
        self._model = None

    @classmethod
    def available_models(cls, query: Optional[str] = None):
        """List available models
        Args:
            query (str, optional): Query to filter the list of available models. Defaults to None.
        """
        if query is None:
            return cls.AVAILABLE_MODELS
        else:
            return [x for x in cls.AVAILABLE_MODELS if re.search(query, x, re.IGNORECASE)]

    @classmethod
    def from_pretrained(cls, model_name: str):
        """Load pretrained model using the dgllife API and not the store"""
        if not requires.check("dgllife"):
            raise ValueError("dgllife is not installed")
        import dgllife

        base_model = dgllife.model.load_pretrained(model_name)
        model = DGLModel(name=model_name)
        model.eval()
        model._model = base_model
        return model

    def load(self):
        """Load GIN model"""
        if self._model is not None:
            return self._model
        download_output_dir = self._artifact_load(
            name=self.name, download_path=self.cache_path, store=self.store
        )
        model_path = dm.fs.join(download_output_dir, self.store.MODEL_PATH_NAME)
        with fsspec.open(model_path, "rb") as f:
            model = joblib.load(f)
        model.eval()
        return model


class PretrainedDGLTransformer(PretrainedMolTransformer):
    r"""
    DGL Pretrained transformer

    Attributes:
        featurizer (DGLModel): DGL featurizer object
        dtype (type, optional): Data type.
        pooling (str, optional): Pooling method for GIN's embedding layer (Default: mean)
        batch_size (int, optional): Batch size to consider for model
    """

    def __init__(
        self,
        kind: Union[str, DGLModel] = "gin_supervised_contextpred",
        dtype: Callable = np.float32,
        pooling: str = "mean",
        batch_size: int = 32,
        preload: bool = False,
        **params,
    ):
        """DGL pretrained featurizer

        Args:
            kind (str, optional): name of the pretrained gin. Defaults to "gin_supervised_contextpred".
            dtype: datatype. Defaults to np.float32.
            pooling: global pooling to perform. Defaults to "mean".
            batch_size: batch size for featurizing the molecules. Defaults to 32.
            preload: whether to preload the internal pretrained featurizer or not

        """
        if not requires.check("dgllife"):
            raise ValueError("Cannot find dgl|dgllife. It's required for this featurizer !")
        super().__init__(
            dtype=dtype,
            pooling=pooling,
            batch_size=batch_size,
            preload=preload,
            kind=kind,
            **params,
        )
        self.pooling = pooling
        self.preload = preload
        self._pooling_obj = self.get_pooling(pooling)
        if isinstance(kind, DGLModel):
            self.kind = kind.name
            self.featurizer = kind
        else:
            self.kind = kind
            self.featurizer = DGLModel(name=self.kind)
        self.batch_size = int(batch_size)
        if self.preload:
            self._preload()

    def __repr__(self):
        return "{}(kind={}, pooling={}, dtype={})".format(
            self.__class__.__name__,
            _parse_to_evaluable_str(self.kind),
            _parse_to_evaluable_str(self.pooling),
            _parse_to_evaluable_str(self.dtype),
        )

    def _update_params(self):
        super()._update_params()
        self._pooling_obj = self.get_pooling(self.pooling)
        featurizer = DGLModel(name=self.kind)
        self.featurizer = featurizer.load()

    @staticmethod
    def get_pooling(pooling: str):
        """Get pooling method from name

        Args:
            pooling: name of the pooling method
        """
        pooling = pooling.lower()
        if pooling in ["mean", "avg", "average"]:
            return AvgPooling()
        elif pooling == "sum":
            return SumPooling()
        elif pooling == "max":
            return MaxPooling()
        else:
            raise ValueError(f"Pooling: {pooling} not supported !")

    def _embed_gin(self, dataset):
        """Embed molecules using GIN"""
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=dgl.batch,
            shuffle=False,
            drop_last=False,
        )

        mol_emb = []
        for batch_id, bg in enumerate(data_loader):
            if self.verbose:
                logger.debug("Processing batch {:d}/{:d}".format(batch_id + 1, len(data_loader)))
            nfeats = [
                bg.ndata.pop("atomic_number").to(torch.device("cpu")),
                bg.ndata.pop("chirality_type").to(torch.device("cpu")),
            ]
            efeats = [
                bg.edata.pop("bond_type").to(torch.device("cpu")),
                bg.edata.pop("bond_direction_type").to(torch.device("cpu")),
            ]
            with torch.no_grad():
                node_repr = self.featurizer(bg, nfeats, efeats)
            mol_emb.append(self._pooling_obj(bg, node_repr))
        mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
        return mol_emb

    def _embed_jtvae(self, dataset):
        """Embed molecules using JTVAE"""
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=JTVAECollator(training=False))

        mol_emb = []
        for tree, tree_graph, mol_graph in dataloader:
            _, tree_vec, mol_vec = self.featurizer.encode(tree_graph, mol_graph)
            enc = torch.cat([tree_vec, mol_vec], dim=1).detach()
            mol_emb.append(enc)
        mol_emb = torch.cat(mol_emb, dim=0).cpu().numpy()
        return mol_emb

    def _embed(self, smiles: List[str], **kwargs):
        """Embed molecules into a latent space"""
        self._preload()
        dataset, successes = self.graph_featurizer(smiles, kind=self.kind)
        if self.kind in DGLModel.available_models(query="^jtvae"):
            mol_emb = self._embed_jtvae(dataset)
        else:
            mol_emb = self._embed_gin(dataset)

        mol_emb = list(mol_emb)
        out = []
        k = 0
        for success in successes:
            if success:
                out.append(mol_emb[k])
                k += 1
            else:
                out.append(None)
        return out

    @staticmethod
    def graph_featurizer(smiles: List[str], kind: Optional[str] = None):
        """
        Construct graphs from SMILES and featurize them

        Args:
            smiles: SMILES of molecules for embedding computation

        Returns:
            dataset: List of graphs constructed and featurized
            list of bool: Indicators for whether the SMILES string can be parsed by RDKit
        """
        if kind in DGLModel.available_models(query="^jtvae"):
            vocab = JTVAEVocab()

            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            with fsspec.open(tmp_file.name, "w") as f:
                f.write("\n".join(smiles))
            dataset = JTVAEDataset(tmp_file.name, vocab, training=False)
            os.unlink(tmp_file.name)
            # JTVAE does not support failure
            success = [True] * len(smiles)
            if len(dataset) != len(smiles):
                raise ValueError("JTVAE failed to featurize some molecules !")
            return dataset, success

        else:
            graphs = []
            success = []
            for smi in smiles:
                try:
                    mol = dm.to_mol(smi)
                    if mol is None:
                        success.append(False)
                        continue
                    g = mol_to_bigraph(
                        mol,
                        add_self_loop=True,
                        node_featurizer=PretrainAtomFeaturizer(),
                        edge_featurizer=PretrainBondFeaturizer(),
                        canonical_atom_order=False,
                    )
                    graphs.append(g)
                    success.append(True)
                except Exception as e:
                    logger.error(e)
                    success.append(False)
            return graphs, success
