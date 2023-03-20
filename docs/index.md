MolFeat: An all in one package for molecule featurization
---

`molfeat` is a python library to featurize molecules. It offers a growing list of featurization schemes, including:

- Molecular descriptors (rdkit and mordred)
- 2D and 3D pharmacophores
- Pretrained featurizers (chemberta, chemgpt, etc)
- Graph featurization (tensors and dgl)
- Fingerprints 
- Scaffold Key

## Installation

### Installing MolFeat
Use conda:

```bash

mamba install -c invivoai -c conda-forge molfeat

```

### Installing Plugins

The functionality of `MolFeat` can be extended through plugins. There are various types of functionality (new calculators, support for 3D, new pretrained featurizers, etc) that can be extended through plugins.

Any additional plugins to `MolFeat` can be installed using the prefered installation guide of the extension (mainly through `pip` or `conda`):

Note that all `MolFeat` extensions follows the `molfeat-{myplugin}` naming convention.


### Dependencies

`molfeat` depends primarly on `rdkit` and `datamol`. However, a few featurization methods require the installation of additional packages. For example, 3D pharmacophore featurization requires `pmapper`, while the DGL graph featurizers require `dgl` and in some rare cases `dgllife`.

Except for `dgllife` and `map4`, all the optional dependencies are installed through conda.

- To install `dgllife`, run `mamba install -c dglteam dgllife`
- To install `map4`: see https://github.com/reymond-group/map4

For the dependencies of any specific plugin, please refer to the plugin installation guide. 


## More

- Check the [getting started tutorial](tutorials/getting-started.ipynb) to learn more how to use `molfeat`
- Check the [Developers](developers/) section if you plan to extend molfeat functionalities with new featurizers
- Check the [API](api/) to get the full documentation


## Licence

`MolFeat` is under the `Apache 2.0` Licence. For more information, see [Licence](./license.md)