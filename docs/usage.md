# Usage
## Structure
Molfeat is organized in three main modules:

- `molfeat.store`: The model store loads, lists and registers all featurizers.
- `molfeat.calc`: A calculator is a callable that featurizes a single molecule. 
- `molfeat.trans`: A transformer is a scikit-learn compatible class that wraps a calculator in a featurization pipeline.

!!! note annotate "Learn more about the different types of featurizers"
    Consult [this tutorial](./tutorials/types_of_featurizers.ipynb) to dive deeper into the differences between the calculator and transformer.
    It provides a good overview of the different types of featurizers and has pointers for learning about more advanced features. 

## Quick API Tour

!!! note tip "Community contribution"
    Curious how molfeat can simplify training QSAR models? See this tutorial contributed by [@PatWalters](https://github.com/PatWalters): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/QSAR_in_8_lines.ipynb)  

```python
import datamol as dm
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
from molfeat.store.modelstore import ModelStore

# Load some dummy data
data = dm.data.freesolv().sample(100).smiles.values

# Featurize a single molecule
calc = FPCalculator("ecfp")
calc(data[0])

# Define a parallelized featurization pipeline
mol_transf = MoleculeTransformer(calc, n_jobs=-1)
mol_transf(data)

# Easily save and load featurizers
mol_transf.to_state_yaml_file("state_dict.yml")
mol_transf = MoleculeTransformer.from_state_yaml_file("state_dict.yml")
mol_transf(data)

# List all available featurizers
store = ModelStore()
store.available_models

# Find a featurizer and learn how to use it
model_card = store.search(name="ChemBERTa-77M-MLM")[0]
model_card.usage()
```

