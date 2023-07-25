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
#### What is a molecular featurizer ?
A molecular featurizer is a function or model that provides numerical representations for molecular structures. These numerical features serve as inputs for machine learning models, enabling them to predict molecular properties and activities, design novel molecules, perform molecular analyses, or conduct searches for similar molecules.

#### Why so many molecular featurizers in `molfeat`?

The reason for providing a diverse range of molecular featurizers in `molfeat` is to address the inherent uncertainty in determining which molecular representation performs best for a given task. Different featurization methods exist, such as using physico-chemical descriptors, molecular structure fingerprints, deep learning embeddings, and more. The effectiveness of these representations varies depending on the specific application. Therefore, the availability of multiple featurizers in `molfeat` ensures that users can access the most suitable featurizer for their unique needs.


#### What is the difference between a calculator and a featurizer in `molfeat`?

In `molfeat`,

- a `calculator` operates on individual molecules and specifies the process of transforming an input molecule into a numerical representation.
- a `featurizer`  works with batches of molecules, leveraging the efficiency of deep learning models on batch processing. Some  `featurizers` uses a `calculator` internally to feature each molecule individually and then stitch their outputs together. Additionally, `featurizers` offer convenient tools, such as parallelism and caching, to optimize the computation of molecular representations efficiently.

`molfeat` has been designed with utmost flexibility, recognizing that the actions users wish to perform with molecular data can be vast and diverse, and there often isn't a single "right" way to approach them.


#### What functions should I be familiar with when using the featurizer classes ?

When using a `featurizer` in `molfeat`, you should be familiar with the following functions:

- `preprocess()`: This method performs preprocessing of your input molecules to ensure compatibility with the expected featurizer class you are using. It's essential to note that the preprocessing steps **are not automatically applied to your inputs** to maintain independence from the molecular transformation. The preprocess function takes your molecule inputs, along with optional labels, and can be redefined when creating a custom featurizer.

-  `transform()`: This method operates on a batch of molecules and returns a list of representations, where the actual featurization occurs. In cases where featurization fails, the position can be denoted as `None`, especially when you choose to `ignore_errors`.
- `_transform()`: This method operates on a single input molecule, performing the actual featurization.
- `__call__()`: This method uses `transform()` under the hood and provides convenient arguments, such as enforcing the datatype defined during the initialization of your model, to the outputs. If you specify `ignore_errors`, a vector of indexes where featurization did not fail will also be returned.

In addition to the methods described above, PretrainedMolTransformer introduces the following functions:

- `_embed()`: For pre-trained models that benefit from batched featurization, this method is internally called during transform instead of an internal calculator.
- `_convert()`: This method is called by the transformer to convert the molecule input into the expected format of the underlying ML model. For example, for a pre-trained language model expecting SELFIES strings, we will perform the conversion to SELFIES strings here.



#### I am getting an error and I am not sure what to do ?

When encountering an error during the featurization process, you have a couple of options to handle it:

- Ignore Errors: You can choose to set the `ignore_errors` parameter to `True` when using the featurizer. This allows the featurizer to continue processing even if it encounters errors on some molecules in your dataset. The featurizer will still attempt to calculate representations for all molecules, and any molecules that failed featurization will have their position in the output list marked as `None`.

- Increase Verbosity: If you're unsure about the specific errors occurring during featurization, you can set the verbosity of the featurizer to True. This will enable the featurizer to log all errors encountered during the process, providing more detailed information about the cause of the issue, since because of the above features, some silent errors are often caught but not propagated.

For example, the following will ensure that all errors are logged.

```python
from molfeat.trans.concat import FeatConcat
from molfeat.trans.fp import FPVecTransformer
import numpy as np
featurizer = MoleculeTransformer(..., dtype=np.float32, verbose=True)
featurizer(["CSc1nc2cc3c(cc2[nH]1)N(Cc1ccc(S(=O)(=O)c2ccccc2)cc1)CCC3"], enforce_dtype=True)
```


#### What are the base featurizers class in molfeat and how to use them ?


| Class                                                                                                                                                                                | Module                                    | Why?                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [BaseFeaturizer](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.base.html#molfeat.trans.base.BaseFeaturizer)                                                               | `molfeat.trans.base`                      | Lowest level featurizer class. All featurizers (even if not molecular) inherit from this class. It's recommended to use `MoleculeTransformer` as the root class instead.                                                                                                                                                                 |
| [MoleculeTransformer](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.base.html#molfeat.trans.base.MoleculeTransformer)                                                     | `molfeat.trans.base`                      | <ul><li> Base class for all molecule featurizers. This is where you start if you want to implement a new featurizer.</li><li> You can provide either an existing `calculator` or your own (a **python callable**) directly to define a new `featurizer`.</li></ul>                                                                       |
| [PrecomputedMolTransformer](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.base.html#molfeat.trans.base.PrecomputedMolTransformer)                                         | `molfeat.trans.base`                      | Class for dealing with precomputed features. You can leverage this class to compute features, save them in a file, and reload them after for other tasks efficiently. [See this tutorial!](https://molfeat-docs.datamol.io/stable/tutorials/datacache.html#using-a-cache-with-a-precomputed-transformer)                                 |
| [FeatConcat](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.concat.html#molfeat.trans.concat.FeatConcat)                                                                   | `molfeat.trans.concat`                    | Convenient class for concatenating multiple vector-featurizers automatically. If you want to combine multiple 'fingerprints' and descriptors, this is the class you use. [See example!](https://molfeat-docs.datamol.io/stable/tutorials/types_of_featurizers.html#concatenate-featurizers)                                              |
| [PretrainedMolTransformer](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.pretrained.base.html)                                                                            | `molfeat.trans.pretrained.base`           | Base class for all `pretrained featurizers`. A `pretrained featurizer` is a `featurizer` that is derived from a pretrained machine learning model. Implement a subclass of this to define your new pretrained featurizer. [See example!](https://molfeat-docs.datamol.io/stable/tutorials/add_your_own.html#define-your-own-transformer) |
| [PretrainedDGLTransformer](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.pretrained.dgl_pretrained.html#molfeat.trans.pretrained.dgl_pretrained.PretrainedDGLTransformer) | `molfeat.trans.pretrained.dgl_pretrained` | Base class for all `dgl pretrained featurizers`. You can initialize a new dgl/dgllife pretrained model as a `molfeat featurizer` easily using this class. You only need to add the dgl model object to a store. [See this example!](https://github.com/datamol-io/molfeat/blob/main/nb/etl/dgl-etl.ipynb)                                |
| [PretrainedHFTransformer](https://molfeat-docs.datamol.io/stable/api/molfeat.trans.pretrained.hf_transformers.html#molfeat.trans.pretrained.hf_transformers.PretrainedHFTransformer) | `molfeat.trans.pretrained.hf_transformer` | Base class for all `huggingface pretrained featurizers`. You can initialize a new 🤗 Transformers pretrained model as a `molfeat featurizer` easily using this class. [See this example!](https://github.com/datamol-io/molfeat/blob/main/nb/etl/molt5-etl.ipynb)                                                                         |


