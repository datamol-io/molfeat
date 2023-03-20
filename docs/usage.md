# Getting started

Molfeat is organized in two main modules:

- `molfeat.store`: The model store loads, lists and registers all featurizers.
- `molfeat.calc`: A calculator is a callable that featurizes a single molecule. 
- `molfeat.trans`: A transformer is a scikit-learn compatible class that wraps a calculator in a featurization pipeline.
  - All transformers in Molfeat inherit from `MoleculeTransformer`.
  - Override `postprocess()` or `preprocess()` to customize the featurization pipeline.
  - Further customize the featurization pipeline through callbacks.
  - Override `get_collate_fn()` to associate the featurizer with a specific collate function.
  - Easily parallelize the featurization pipeline by setting the `n_jobs` parameter.
  - Save and load transformers through serialized state dicts.

## Quick API Tour
```python
import datamol as dm
from molfeat.calc import get_calculator
from molfeat.trans import MoleculeTransformer
from molfeat.store.modelstore import ModelStore

# Load some dummy data
data = dm.data.freesolv().sample(500).smiles.values

# Featurize a single molecule
calc = get_calculator("ecfp")
calc(data[0])

# Define a parallelized featurization pipeline
trans = MoleculeTransformer(calc, n_jobs=-1)
trans(data)

# Easily save and load featurizers
trans.to_state_yaml_file("state_dict.yml")
trans = MoleculeTransformer.from_state_yaml_file("state_dict.yml")
trans(data)

# List all availaible featurizers
store = ModelStore()
store.available_models

# Find a featurizer and learn how to use it
model_card = store.search(name="DeepChem-ChemBERTa-77M-MLM")[0]
model_card.usage()

# Load a featurizer through the store
trans, model_info = store.load(model_card)
```