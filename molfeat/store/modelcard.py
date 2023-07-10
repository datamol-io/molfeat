from datetime import datetime
from typing import List, Literal, Optional, Union

import datamol as dm
from pydantic import BaseModel, ConfigDict, Field


def get_model_init(card):
    """Get the model initialization code

    Args:
        card: model card to use
    """
    if card.group == "all" and card.type != "pretrained":
        import_statement = "from molfeat.trans import MoleculeTransformer"
        loader_statement = f"MoleculeTransformer(featurizer='{card.name}', dtype=float)"
    elif card.group in ["rdkit", "fp", "shape"]:
        import_statement = f"from molfeat.trans.fp import FPVecTransformer"
        loader_statement = f"FPVecTransformer(kind='{card.name}', dtype=float)"
    elif card.group == "dgllife":
        import_statement = "from molfeat.trans.pretrained import PretrainedDGLTransformer"
        loader_statement = f"PretrainedDGLTransformer(kind='{card.name}', dtype=float)"
    elif card.group == "graphormer":
        import_statement = "from molfeat.trans.pretrained import GraphormerTransformer"
        loader_statement = f"GraphormerTransformer(kind='{card.name}', dtype=float)"
    elif card.group == "fcd":
        import_statement = "from molfeat.trans.pretrained import FCDTransformer"
        loader_statement = f"FCDTransformer()"
    elif card.group == "pharmacophore":
        name = card.name.split("-")[-1]
        if card.require_3D:
            import_class = "Pharmacophore3D"
        else:
            import_class = "Pharmacophore2D"
        import_statement = f"from molfeat.trans.base import MoleculeTransformer\nfrom molfeat.calc.pharmacophore import {import_class}"
        loader_statement = (
            f"MoleculeTransformer(featurizer={import_class}(factory='{name}'), dtype=float)"
        )
    elif card.group == "huggingface":
        import_statement = (
            "from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer"
        )
        loader_statement = (
            f"PretrainedHFTransformer(kind='{card.name}', notation='{card.inputs}', dtype=float)"
        )
    else:
        raise ValueError(f"Unknown model group {card.group}")
    return import_statement, loader_statement


class ModelInfo(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(
            "protected_",
        )  # Prevents warning from usage of model_ prefix in fields
    )

    name: str
    inputs: str = "smiles"
    type: Literal["pretrained", "hand-crafted", "hashed", "count"]
    version: int = 0
    group: Optional[str] = "all"
    submitter: str
    description: str
    representation: Literal["graph", "line-notation", "vector", "tensor", "other"]
    require_3D: Optional[bool] = False
    tags: Optional[List[str]] = []
    authors: Optional[List[str]]
    reference: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    sha256sum: Optional[str] = None
    model_usage: Optional[str] = None

    def path(self, root_path: str):
        """Generate the folder path where to save this model

        Args:
            root_path: path to the root folder
        """
        version = str(self.version or 0)
        return dm.fs.join(root_path, self.group, self.name, version)

    def match(self, new_card: Union["ModelInfo", dict], match_only: Optional[List[str]] = None):
        """Compare two model card information and returns True if they are the same

        Args:
            new_card: card to search for in the modelstore
            match_only: list of minimum attribute that should match between the two model information
        """

        self_content = self.model_dump().copy()
        if not isinstance(new_card, dict):
            new_card = new_card.model_dump()
        new_content = new_card.copy()
        # we always remove the datetime field
        self_content.pop("created_at", None)
        new_content.pop("created_at", None)
        if match_only is not None:
            self_content = {k: self_content.get(k) for k in match_only}
            new_content = {k: new_content.get(k) for k in match_only}
        return self_content == new_content

    def set_usage(self, usage: str):
        """Set the usage of the model

        Args:
            usage: usage of the model
        """
        self.model_usage = usage

    def usage(self):
        """Return the usage of the model"""
        if self.model_usage is not None and self.model_usage:
            return self.model_usage
        import_statement, loader_statement = get_model_init(self)
        comment = "# sanitize and standardize your molecules if needed"
        if self.require_3D:
            comment += "\n# <generate 3D coordinates here> "
        usage = f"""
        import datamol as dm
        {import_statement}
        smiles = dm.freesolv().iloc[:100].smiles
        {comment}
        transformer = {loader_statement}
        features = transformer(smiles)
        """
        usage = "\n".join([x.strip() for x in usage.split("\n")])
        return usage
