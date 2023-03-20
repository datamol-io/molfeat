# `MolFeat`: An all in one package for molecule featurization

`molfeat` is a python library to featurize molecules. It offers a growing list of featurization schemes, including:

- Molecular descriptors (rdkit and mordred)
- 2D and 3D pharmacophores
- Pretrained featurizers (chemberta, chemgpt, etc)
- Graph featurization (tensors and dgl)
- Fingerprints
- Scaffold Keys

## Documentation

For the documentation visit https://doc.molfeat.io

## Installation

Use conda:

```bash
mamba install -c invivoai -c conda-forge -c dglteam molfeat
```

### Optional dependencies

`molfeat` has optional dependencies. By default `dgl`, `fcd_torch`, and `pmapper` will be installed with the package.

- To install `dgllife`, run `mamba install -c dglteam dgllife`
- To install `graphormer-pretrained`, run `mamba install -c conda-forge -c invivoai -c dglteam 'graphormer-pretrained>=0.0.5'`
- To install `map4`: see https://github.com/reymond-group/map4
- To install `bio-embeddings` run `mamba install -c conda-forge 'bio-embeddings >=0.2.2'`

## Changelogs

See the latest changelogs at [CHANGELOG.rst](./CHANGELOG.rst).

## License

Copyright 2021 Valence. See [LICENSE](LICENSE).

## Authors

See [AUTHORS.rst](./AUTHORS.rst).

## Development Lifecycle

### Setup a local dev environment

```bash
mamba env create -n molfeat -f env.yml
conda activate molfeat
pip install -e .
```

### Tests

```bash
pytest tests/ 
```

### Build the documentation

You can build and serve the documentation locally with:

```bash
# Build and serve the doc
mkdocs serve
```
