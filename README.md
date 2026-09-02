<div align="center">
    <img src="docs/images/logo-title.svg" width="100%">
</div>

<p align="center">
    <b>molfeat - the hub for all your molecular featurizers </b> <br />
</p>
<p align="center">
  <a href="https://molfeat-docs.datamol.io/" target="_blank">
      Docs
  </a> |
  <a href="https://molfeat.datamol.io/" target="_blank">
      Homepage
  </a>
</p>

---

[![DOI](https://zenodo.org/badge/613548667.svg)](https://zenodo.org/badge/latestdoi/613548667)
[![PyPI](https://img.shields.io/pypi/v/molfeat)](https://pypi.org/project/molfeat/)
[![Conda](https://img.shields.io/conda/v/conda-forge/molfeat?label=conda&color=success)](https://anaconda.org/conda-forge/molfeat)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/molfeat)](https://pypi.org/project/molfeat/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/molfeat)](https://anaconda.org/conda-forge/molfeat)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/molfeat)](https://pypi.org/project/molfeat/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/datamol-io/molfeat/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/datamol-io/molfeat)](https://github.com/datamol-io/molfeat/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/datamol-io/molfeat)](https://github.com/datamol-io/molfeat/network/members)
[![test](https://github.com/datamol-io/molfeat/actions/workflows/test.yml/badge.svg)](https://github.com/datamol-io/molfeat/actions/workflows/test.yml)
[![code-check](https://github.com/datamol-io/molfeat/actions/workflows/code-check.yml/badge.svg)](https://github.com/datamol-io/molfeat/actions/workflows/code-check.yml)
[![doc](https://github.com/datamol-io/molfeat/actions/workflows/doc.yml/badge.svg)](https://github.com/datamol-io/molfeat/actions/workflows/doc.yml)
[![release](https://github.com/datamol-io/molfeat/actions/workflows/release.yml/badge.svg)](https://github.com/datamol-io/molfeat/actions/workflows/release.yml)

Molfeat is a hub of molecular featurizers. It supports a wide variety of out-of-the-box molecular featurizers and can be easily extended to include your own custom featurizers.

- 🚀 Fast, with a simple and efficient API.
- 🔄 Unify pre-trained molecular embeddings and hand-crafted featurizers in a single package.
- ➕ Easily add your own featurizers through plugins.
- 📈 Benefit from increased performance through a trouble-free caching system.

Visit our website at <https://molfeat.datamol.io>.

## Updates

Molfeat 1.x narrows the project to small-molecule featurization and a modern,
maintainable stack. It removes the legacy DGL, DGLLife, Graphormer and protein
adapters, keeps PyTorch Geometric as the graph backend, and adds maintained
foundation-model integrations including CheMeleon and Mol-JEPA. Model loading
is lazy, external checkpoint licences are explicit, and official
representations are covered by integration tests.

See the [complete changelog](CHANGELOG.md) and the
[1.x migration guide](docs/migration.md). The 1.x work is currently on the
`dev` branch; the published PyPI and conda-forge packages remain the stable
releases.

## Installation

### Installing Molfeat

Add Molfeat to a uv-managed project:

```bash
uv add molfeat
```

Pip and conda-forge remain supported: `python -m pip install molfeat` or
`mamba install -c conda-forge molfeat`.

The next major release requires Python 3.11 or newer and RDKit 2024.09 or
newer. PyTorch 2.5 or newer is used on maintained platforms; macOS Intel uses
the final available 2.2 wheel series with NumPy 1.26 and is tested on Python 3.12.

### Optional dependencies

Not all featurizers in the Molfeat core package are supported by default. Some featurizers require additional dependencies. If you try to use a featurizer that requires additional dependencies, Molfeat will raise an error and tell you which dependencies are missing and how to install them.

- To install Hugging Face Transformers support: `python -m pip install "molfeat[transformer]"`.
- To install PyTorch Geometric support: `python -m pip install "molfeat[pyg]"`.
- To install FCD support: `python -m pip install "molfeat[fcd]"`.
- To install HDF5 and Parquet cache support: `python -m pip install "molfeat[cache]"`.
- To install S3 and Google Cloud model stores: `python -m pip install "molfeat[cloud]"`.
- To install Mordred descriptors: `python -m pip install "molfeat[mordred]"`.
- To install `map4`: see <https://github.com/reymond-group/map4>

`python -m pip install "molfeat[all]"` installs every maintained optional
dependency compatible with the current interpreter. DGL, DGLLife and the
legacy Graphormer adapter have been removed: their binary and build constraints
conflict with the supported modern stack. PyTorch Geometric is the maintained
graph backend. Protein featurizers have also been removed so Molfeat 1.x has a
precise small-molecule scope. See the [migration guide](docs/migration.md) for
details.

### Compatibility

| Molfeat | Python | RDKit | PyTorch |
| --- | --- | --- | --- |
| `1.x` (development) | `3.11–3.14` | `2024.09+` | `2.5+` (`2.2.x` on macOS Intel) |
| `0.x` | See the release metadata | See the release metadata | See the release metadata |

### Installing Plugins

The functionality of Molfeat can be extended through plugins. The use of a plugin system ensures that the core package remains easy to install and as light as possible, while making it easy to extend its functionality with plug-and-play components. Additionally, it ensures that plugins can be developed independently from the core package, removing the bottleneck of a central party that reviews and approves new plugins. Consult the molfeat documentation for more details on how to [create](docs/developers/create-plugin.md) your own plugins.

However, this does imply that the installation of a plugin is plugin-dependent: please consult the relevant documentation to learn more.

## API tour

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

## How to cite

Please cite Molfeat if you use it in your research: [![DOI](https://zenodo.org/badge/613548667.svg)](https://zenodo.org/badge/latestdoi/613548667).

## Contribute

See [developers](docs/developers/) for a comprehensive guide on how to contribute to `molfeat`. `molfeat` is a community-led
initiative and whether you're a first-time contributor or an open-source veteran, this project greatly benefits from your contributions.
To learn more about the community and [datamol.io](https://datamol.io/) ecosystem, please see [community](docs/community/).

## Maintainers

- @cwognum
- @maclandrol
- @hadim

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).
