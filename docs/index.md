# What is molfeat ?

Molfeat is a hub of molecular featurizers. It supports a wide variety of out-of-the-box molecular featurizers and can be easily extended to include your own custom featurizers.

- 🚀 Fast, with a simple and efficient API.
- 🔄 Unify pre-trained molecular embeddings and hand-crafted featurizers in a single package.
- ➕ Easily add your own featurizers through plugins.
- 📈 Benefit from increased performance through a trouble-free caching system.

Visit our website at https://molfeat.datamol.io.

## Updates

The upcoming 1.x release focuses on small molecules, replaces legacy DGL and
protein adapters, and adds CheMeleon and Mol-JEPA integrations. See the
[changelog](https://github.com/datamol-io/molfeat/blob/dev/CHANGELOG.md) and
[migration guide](migration.md) for changes, model licensing and upgrade details.
These changes are not yet a published release.

## Installation

### Installing Molfeat

Add Molfeat to a uv-managed project:

```bash
uv add molfeat
```

Pip and conda-forge remain supported: `python -m pip install molfeat` or
`mamba install -c conda-forge molfeat`.

The next major release requires Python 3.11 or newer, RDKit 2024.09 or newer,
and PyTorch 2.5 or newer. macOS Intel uses the final PyTorch 2.2 wheel series
with NumPy 1.26 and Transformers 4.57, and is tested on Python 3.12.
See the [migration guide](migration.md) for checkpoint-format restrictions and
co-installation with SAFE.

### Installing Plugins

The functionality of molfeat can be extended through plugins. The use of a plugin system ensures that the core package remains easy to install and as light as possible, while making it easy to extend its functionality with plug-and-play components. Additionally, it ensures that plugins can be developed independently from the core package, removing the bottleneck of a central party that reviews and approves new plugins. Consult the molfeat documentation for more details on how to [create](developers/create-plugin.md) your own plugins.

However, this does imply that the installation of a plugin is plugin-dependent: please consult the relevant documentation to learn more.

### Optional dependencies

Not all featurizers in Molfeat core package are supported by default. Some featurizers require additional dependencies. If you try to use a featurizer that requires additional dependencies, Molfeat will raise an error and tell you which dependencies are missing and how to install them.

- To install Hugging Face Transformers support: `python -m pip install "molfeat[transformer]"`.
- To convert SELFIES without Transformers: `python -m pip install "molfeat[selfies]"`.
- To install PyTorch Geometric support: `python -m pip install "molfeat[pyg]"`.
- To install FCD support: `python -m pip install "molfeat[fcd]"`.
- To install HDF5 and Parquet cache support: `python -m pip install "molfeat[cache]"`.
- To install S3 and Google Cloud model stores: `python -m pip install "molfeat[cloud]"`.
- To install Mordred descriptors: `python -m pip install "molfeat[mordred]"`.
- To install `map4`: see <https://github.com/reymond-group/map4>.

`python -m pip install "molfeat[all]"` installs every maintained optional
dependency compatible with the current interpreter. DGL, DGLLife and the
legacy Graphormer adapter have been removed because they conflict with the
supported modern stack. PyTorch Geometric is the maintained graph backend.
Protein featurizers have also been removed so Molfeat 1.x has a precise
small-molecule scope. See the [migration guide](migration.md) for details.

### Compatibility

| Molfeat | Python | RDKit | PyTorch |
| --- | --- | --- | --- |
| `1.x` (development) | `3.11–3.14` | `2024.09+` | `2.5+` (`2.2.x` on macOS Intel) |
| `0.x` | See the release metadata | See the release metadata | See the release metadata |

## How to cite

Please cite Molfeat if you use it in your research: [![DOI](https://zenodo.org/badge/613548667.svg)](https://zenodo.org/badge/latestdoi/613548667).
