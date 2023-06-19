# What is molfeat ?

Molfeat is a hub of molecular featurizers. It supports a wide variety of out-of-the-box molecular featurizers and can be easily extended to include your own custom featurizers.

- 🚀 Fast, with a simple and efficient API.
- 🔄 Unify pre-trained molecular embeddings and hand-crafted featurizers in a single package.
- ➕ Easily add your own featurizers through plugins.
- 📈 Benefit from increased performance through a trouble-free caching system.

Visit our website at https://molfeat.datamol.io.

## Installation

### Installing Molfeat

Use mamba:

```bash
mamba install -c conda-forge molfeat
```

_**Tips:** You can replace `mamba` by `conda`._

_**Note:** We highly recommend using a [Conda Python distribution](https://github.com/conda-forge/miniforge) to install Molfeat. The package is also pip installable if you need it: `pip install molfeat`._

### Installing Plugins

The functionality of molfeat can be extended through plugins. The use of a plugin system ensures that the core package remains easy to install and as light as possible, while making it easy to extend its functionality with plug-and-play components. Additionally, it ensures that plugins can be developed independently from the core package, removing the bottleneck of a central party that reviews and approves new plugins. Consult the molfeat documentation for more details on how to [create](developers/create-plugin.md) your own plugins.

However, this does imply that the installation of a plugin is plugin-dependent: please consult the relevant documentation to learn more.

### Optional dependencies

Not all featurizers in Molfeat core package are supported by default. Some featurizers require additional dependencies. If you try to use a featurizer that requires additional dependencies, Molfeat will raise an error and tell you which dependencies are missing and how to install them.

- To install `dgl`: run `mamba install -c dglteam dgl`
- To install `dgllife`: run `mamba install -c conda-forge dgllife`
- To install `fcd_torch`: run `mamba install -c conda-forge fcd_torch`
- To install `pyg`: run `mamba install -c conda-forge pytorch_geometric`
- To install `graphormer-pretrained`: run `mamba install -c conda-forge graphormer-pretrained`
- To install `map4`: see https://github.com/reymond-group/map4
- To install `bio-embeddings`: run `mamba install -c conda-forge 'bio-embeddings >=0.2.2'`

If you install Molfeat using pip, there are optional dependencies that can be installed with the main package. For example, `pip install "molfeat[all]"` allows installing all the compatible optional dependencies for small molecule featurization. There are other options such as `molfeat[dgl]`, `molfeat[graphormer]`, `molfeat[transformer]`, `molfeat[viz]`, and `molfeat[fcd]`.

## How to cite

Please cite Molfeat if you use it in your research: [![DOI](https://zenodo.org/badge/613548667.svg)](https://zenodo.org/badge/latestdoi/613548667).
