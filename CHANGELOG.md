# Molfeat Changelogs

_Only changelogs previous to 0.8.10. See the GitHub releases for new changelogs._

## Next major release

**Changed:**

- Require Python 3.11+, RDKit 2024.09+, PyTorch 2.5+, NumPy 1.26+ and
  scikit-learn 1.6+.
- Support the current Transformers 5 and PyTorch Geometric stacks, including a
  strict safetensors fallback for Mol-JEPA's upstream meta-device initialization issue.
- Make model-store discovery lazy so constructing a featurizer does not perform
  an unnecessary network request.
- Separate the cross-platform core matrix from a complete all-extras and
  published-model validation lane in CI.
- Use uv for development, CI, isolated wheel/source smoke tests and publishing.
- Publish releases through PyPI Trusted Publishing with PEP 740 attestations;
  retain conda-forge as a bot-updated downstream channel.
- Apply the serialization defaults from PR #112, scikit-learn compatibility
  from PR #118, delivery updates from PR #121 and FCFP invariants from PR #123.
- Require newly registered pretrained model cards to declare the checkpoint
  licence, and document the permissive-dependency policy for external models.

**Fixed:**

- Use current RDKit valence APIs in atom and bond feature calculation.
- Handle boolean masks correctly for max, mean and sum pooling on current
  PyTorch, including two-dimensional ESM embeddings.
- Return a one-dimensional, deterministic index array from
  `Pharmacophore3D(..., raw=True)`.
- Download HTTP-backed model directories whose listing endpoint also appears as
  a file through the corresponding Datamol fix.

**Removed:**

- Remove DGL, DGLLife, Graphormer, `bio-embeddings` and their adapters; their
  upstream releases and binary constraints are incompatible with the supported
  modern stack. PyTorch Geometric is the maintained graph backend.
- Remove protein featurizers and their plugin entry-point group to focus the
  package on small molecules.

## v0.8.9

**Added:**

- Support for new featurizers (GPT2-Zinc480M-87M and
  Roberta-Zinc480M-102M)
- ETL scripts/notebooks for transparency

**Fixed:**

- Nav Bar with Tabs
- Typos in docs

**Authors:**

- Cas Wognum
- Emmanuel Noutahi

## v0.8.8

**Added:**

- A model named Molt5
- Existing plugins to a the `plugin.yml` file
- Documents missing featurizers

**Changed:**

- Different docs style, with Tabs on the top for `Overview`, `Usage`,
  `Tutorials`, `API`, `Contribute`, `License`
- Small updates in docs

**Fixed:**

- Fix #43: Pretrained models should now work better with
  `batch_transform`, allowing efficient parallelization, while
  retaining all cached features.

**Authors:**

- DomInvivo
- dessygil
- maclandrol

## v0.8.7

**Added:**

- Support for `ignore_padding` in Graphormer
- More flexibility overall for graphormer embeddings

**Changed:**

- Phased out rdchem.Mol, rdchem.Atom and rdkit.Bond in favor of
  datamol versions
- Fully automated release process.

**Fixed:**

- Random logging in cache coming from testing if an input is a
  molecule
- Some small typos in doc strings
- Naming of JTVAE models
- Fix issue #37 by making WeaveFeaturizer faster
- Usage card for rdkit and fingerprints/descriptors featurizers

**Authors:**

- Hadrien Mary
- maclandrol
- rbyrne-momatx

## v0.8.6

**Added:**

- Support for batch transformation in `MoleculeTransformer` for
  calculators that implements `batch_compute`

**Changed:**

- Pull request template for better directive.

**Authors:**

- maclandrol

## v0.8.5

**Authors:**

## v0.8.4

**Added:**

- Add Google Analytics support.

**Removed:**

- Remove support for the `np.float128` dtype (issue #26)

**Fixed:**

- Color bug of the search input bar

**Authors:**

- Cas Wognum
- Honore Hounwanou
- maclandrol

## v0.8.3

**Added:**

- More documentation and tutorials
- pip dependencies and optional dependencies in pyproject.toml
- Github issue templates

**Changed:**

- Updated all occurrences of old molfeat links with new ones.
- Documentation and readme

**Removed:**

- duplicated CODEOWNER file

**Fixed:**

- Bug in serialization of transformers with a serializable calculator
- Minor typos and function definition
- Links in pyprojects

**Authors:**

- Cas Wognum
- Hadrien Mary
- Honore Hounwanou
- Saurav Maheshkar
- Therence
- maclandrol

## v0.8.1

**Added:**

- Extended the simple benchmark in the docs to also include a search
  benchmark, based on the RDKit benchmarking platform
- Added missing API documentation for `molfeat.plugins` and
  `molfeat.store`.

**Changed:**

- Changed the styling of the docs to match the new datamol.io styling.

**Authors:**

- Cas Wognum
- Hadrien Mary
- Therence

## v0.8.0

**Authors:**

- Hadrien Mary
