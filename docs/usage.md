# Getting started

## Organization
Molfeat is organized in three main modules:

- `molfeat.store`: The model store loads, lists and registers all featurizers.
- `molfeat.calc`: A calculator is a callable that featurizes a single molecule. 
- `molfeat.trans`: A transformer is a scikit-learn compatible class that wraps a calculator in a featurization pipeline.

!!! note annotate "Learn more about the different types of featurizers"
    Consult [this tutorial](../tutorials/types_of_featurizers.html) to dive deeper into the differences between the calculator and transformer.
    It provides a good overview of the different types of featurizers and has pointers to learn about more advanced features. 

## Quick API Tour
```python
import datamol as dm
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
from molfeat.store.modelstore import ModelStore

# Load some dummy data
data = dm.data.freesolv().sample(500).smiles.values

# Featurize a single molecule
calc = FPCalculator("ecfp")
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
model_card = store.search(name="ChemBERTa-77M-MLM")[0]
model_card.usage()

# Load a featurizer through the store
trans, model_info = store.load(model_card)
```

