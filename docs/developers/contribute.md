# Contribute

This document details the development lifecycle of Molfeat.

## Setup a dev environment

First you'll need to fork and clone the repository. Once you have a local copy, run the following command. 
It is strongly recommended that you do this within a new conda environment.


```bash
mamba env create -n molfeat -f env.yml
conda activate molfeat
pip install -e .
```


## Continuous Integration

MolFeat uses Github Actions to:

- **Build and test** `molfeat`.
- **Check code formating** the code: `black`.
- **Documentation**: build and deploy the documentation on `main` and for every new git tag.

## Run tests

```bash
pytest
```

## Build the documentation

You can build and serve the documentation locally with:

```bash
# Build and serve the doc
mkdocs serve
```

## Submitting Pull Requests

If you're considering a large code contribution to `MolFeat`, please prefer to open an issue first to get early feedback on the idea.

Once you think the code is ready to be reviewed, push it to your fork and send a pull request. We will assign a reviewer to your PR.
For a change to be accepted it will need to have exiting tests passing, and additional tests and documentation for any new features. 


If you are developping and extension (plugin) of `MolFeat`, please refer to the corresponding section [Extending Functionalities with a Plugin](./plugin.md)

## Release a new version

- Run check: `rever check`.
- Bump and release new version: `rever VERSION_NUMBER`.
- Releasing a new version will do the following things in that order:
  - Update `AUTHORS.rst`.
  - Update `CHANGELOG.rst`.
  - Bump the version number in `setup.py` and `_version.py`.
  - Add a git tag.
  - Push the git tag.
  - Add a new release on the GH repo associated with the git tag.
