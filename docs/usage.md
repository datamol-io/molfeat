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


## FAQ
<details>
<summary>What is a molecular featurizer ?</summary>

A molecular featurizer is function or model that provides numerical representations from molecular structures. These numerical features can then be used as input for machine learning models to predict molecular properties and activities, to design new molecules, to perform molecular analyses, or to search for similar molecules. 
</details>


<details>
<summary>Why so many molecular featurizers in `molfeat`?</summary>

To date, it's not clear which molecular representation performs better. There are multiple ways of representing molecules (e.g using their physico-chemical descriptors, using a fingerprint corresponding to a hash of the molecular structure, using deep learning embeddings, etc). Depending on your tasks, one representation could perform better than another, this is why `molfeat` attempt to provide a broad range of featurizer to ensure, everyone has access to their favorite featurizers.
</details>


<details>
<summary>What is the difference between a calculator and a featurizer in `molfeat`?</summary>

In `molfeat`,
- a `calculator` operate on the level of a single molecule, it dictates how to transform an input molecule into a numerical representation. 

- a `featurizer` operates on batches of molecules, because deep learning models are often more efficient on batch of samples. Some  `featurizers` uses `calculator` internally to each molecule individually and stitch them together. `featurizers` also provide convenient tools such as parallelism, caching, etc to make computation of molecular representation efficient. 

`molfeat` is designed to be extremely flexible. This is because the space of actions that users often wish to perform is huge and there are often not "wrong" ways.
</details>

<details>
<summary>What are the function I should know when using a `featurizer` ?</summary>


Every featurizer would have: 
  - a `preprocess` method  that can perform preprocessing of your input molecules, to ensure compatibility with the expected featurizer class you are using. The preprocess steps is not called automatically for you to decouple it from the molecular transformation. It's a suggestion for the preprocessing steps you should perform when using a given featurizer.

The `preprocess` function expect your molecule inputs, but also some optional labels and can be redefined when creating your own custom featurizer.

  - a `transform` method that operates on a batch of molecules and returns a list of representation, this is where the `magic` happens. Position where featurization failed can be `None` when you elect to `ignore_errors`.
  - a `_transform` method that operates on a single input molecule, this is where the `magic` happens
  - a `__call__` method that uses `transform` under the hood and add some convenient argument such as enforcing the datatype you defined when initializing your model to the outputs.  If you ask to `ignore_errors`, a vector of indexes where featurization did not fail will also be returned. 

In addition to the method described above, `PretrainedMolTransformer` also defines the following functions:

- `_embed`: since pre-trained models benefit from batched featurization, this method is called by internally during `transform` instead of an internal calculator. 
- `_convert`: this method is called by the transformer to convert the molecule input into the expected format of the underlying ML model. For example for a pre-trained language model expecting SELFIES strings, we will convert for input into SELFIES strings here.

</details>

<details>
<summary>I am getting an error and I am not sure what to do. </summary>

User can decide to `ignore_errors` when featurization fails on some molecules of their dataset, with the hope of filtering them after. Therefore, some silent errors are caught in the `transform` errors. Set the verbosity of the featurizer to True to get a log of all errors.

```python
from molfeat.trans.concat import FeatConcat
from molfeat.trans.fp import FPVecTransformer
import numpy as np
featurizer = MoleculeTransformer(..., dtype=np.float32, verbose=True)
featurizer(["CSc1nc2cc3c(cc2[nH]1)N(Cc1ccc(S(=O)(=O)c2ccccc2)cc1)CCC3"], enforce_dtype=True)
```

you will alway have a log of all errors. 
</details>

<details>
<summary>What are the base featurizers class in molfeat and how to use them ?</summary>

|    Class  	| Module	| Why ? 	|
|-------------	|-----------------------	|----------------------	|
| [`BaseFeaturizer`](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.base.html#molfeat.trans.base.BaseFeaturizer) 	| `molfeat.trans.base`          	| Lowest level featurizer class. All featurizers (even if not molecular) inherits from this class.  It's recommended to use `MoleculeTransformer` as root class instead          	|
| [`MoleculeTransformer`](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.base.html#molfeat.trans.base.MoleculeTransformer) 	| `molfeat.trans.base`          	| <ul><li> Base class for all molecule featurizers. This is where you start if you want to implement a new featurizer.</li> <li> You can provide either an existing `calculator` or your own (a **python callable**) directly to define a new `featurizer`</li></ul>|
|[`PrecomputedMolTransformer``](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.base.html#molfeat.trans.base.PrecomputedMolTransformer) 	| `molfeat.trans.base` | Class for dealing with precomputed features. You can leverage this class to compute features, save them in a file, and reload them after for other task efficiently. [See this tutorial !](https://molfeat-docs.datamol.io/stable/tutorials/datacache.html#using-a-cache-with-a-precomputed-transformer) |
|[`FeatConcat`](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.concat.html#molfeat.trans.concat.FeatConcat) 	| `molfeat.trans.concat` | Convenient class for concatenating multiple vector-featurizers automatically. If you want to combine multiple 'fingerprints' and descriptors, this is the class you use. [See example !](https://molfeat-docs.datamol.io/stable/tutorials/types_of_featurizers.html#concatenate-featurizers) |
|[`PretrainedMolTransformer`](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.pretrained.base.html)	| `molfeat.trans.pretrained.base` | Base class for all `pretrained featurizers`. A `pretrained featurizer` is a `featurizer` that is derived from a pretrained machine learning model. Implement a subclass of this to define your new pretrained featurizer.  [See example !](https://molfeat-docs.datamol.io/stable/tutorials/add_your_own.html#define-your-own-transformer) |
|`PretrainedDGLTransformer` 	| `molfeat.trans.pretrained.dgl_pretrained` | Base class for all `dgl pretrained featurizers`. You can initialize a new dgl/dgllife pretrained model as a `molfeat featurizer` easily using this class. You only need to add the dgl model object to a store. |
|[`PretrainedHFTransformer``](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.pretrained.hf_transformers.html#molfeat.trans.pretrained.hf_transformers.PretrainedHFTransformer) 	| `molfeat.trans.pretrained.hf_transformer` | Base class for all `huggingface pretrained featurizers`. You can initialize a new 🤗 Transformers pretrained model as a `molfeat featurizer` easily using this class. [See this example !](https://github.com/datamol-io/molfeat/blob/main/nb/etl/molt5-etl.ipynb) |

</details>
