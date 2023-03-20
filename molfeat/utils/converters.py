import types
import datamol as dm


class SmilesConverter:
    """Molecule line notation conversion from smiles to selfies or inchi"""

    SUPPORTED_LINE_NOTATIONS = [
        "none",
        "smiles",
        "selfies",
        "inchi",
    ]

    def __init__(self, target: str = None):
        """
        Convert input smiles to a target line notation

        Args:
            target: target representation.
        """
        self.target = target

        if self.target is not None and self.target not in self.SUPPORTED_LINE_NOTATIONS:
            raise ValueError(
                f"{target} is not a supported line representation. Choose from {self.SUPPORTED_LINE_NOTATIONS}"
            )

        if self.target == "smiles" or (self.target is None or self.target == "none"):
            self.converter = None
        elif self.target == "inchi":
            self.converter = types.SimpleNamespace(decode=dm.from_inchi, encode=dm.to_inchi)
        elif self.target == "selfies":
            self.converter = types.SimpleNamespace(decode=dm.from_selfies, encode=dm.to_selfies)

    def decode(self, inp: str):
        """Decode inputs into smiles

        Args:
            inp: input representation to decode
        """
        if self.converter is None:
            return inp
        with dm.without_rdkit_log():
            try:
                decoded = self.converter.decode(inp)
                return decoded.strip()
            except:  # (deepsmiles.DecodeError, ValueError, AttributeError, IndexError):
                return None

    def encode(self, smiles: str):
        """Encode a input smiles into target line notation

        Args:
            smiles: input smiles to encode
        """
        if self.converter is None:
            return smiles
        with dm.without_rdkit_log():
            try:
                encoded = self.converter.encode(smiles)
                return encoded.strip()
            except:
                return None
