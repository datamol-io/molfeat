# Molfeat Changelogs

This file records user-visible changes. See the [migration guide](docs/migration.md)
for upgrade instructions and [GitHub releases](https://github.com/datamol-io/molfeat/releases)
for earlier release notes.

## Next major release (unreleased)

### Highlights

- Focus Molfeat 1.x on small molecules and remove incompatible legacy model
  stacks.
- Support current PyTorch, Transformers and PyTorch Geometric releases while
  keeping optional dependencies isolated from the core package.
- Add tested foundation-model adapters with explicit checkpoint provenance and
  licensing.

### Changed

- Require Python 3.11+, RDKit 2024.09+, NumPy 1.26+ and scikit-learn 1.6+.
  PyTorch 2.5+ is used on maintained platforms; macOS Intel remains on the
  final available 2.2 wheel series.
- Support the current Transformers 5 and PyTorch Geometric stacks, including a
  strict safetensors fallback for Mol-JEPA's upstream meta-device initialization issue.
- Make model-store discovery lazy so constructing a featurizer does not perform
  an unnecessary network request.
- Apply the serialization defaults from PR #112, scikit-learn compatibility
  from PR #118, delivery updates from PR #121 and FCFP invariants from PR #123.
- Require newly registered pretrained model cards to declare the checkpoint
  licence, and document the permissive-dependency policy for external models.
- Keep historical DGL and Graphormer cards discoverable, but report that their
  removed runtimes are unsupported instead of generating invalid import code.

### Added

- Add CheMeleon using core RDKit and PyTorch operations, without a Chemprop
  dependency. Mol-JEPA uses the existing `transformer` and `pyg` extras.
- Pin external custom-code revisions and require explicit trust and
  non-commercial licence acknowledgement for Mol-JEPA. No external code or
  weights are redistributed by Molfeat.
- Add official-checkpoint integration tests that lock the expected embedding
  shape and numerical representation.

### Fixed

- Use current RDKit valence APIs in atom and bond feature calculation.
- Handle boolean masks correctly for max, mean and sum pooling on current
  PyTorch, including two-dimensional token embeddings.
- Return a one-dimensional, deterministic index array from
  `Pharmacophore3D(..., raw=True)`.
- Download HTTP-backed model directories whose listing endpoint also appears as
  a file through the corresponding Datamol fix.
- Preserve FCFP aliases, radii and count/binary output invariants.
- Serialize transformer state without unserializable list parameters and use
  the current scikit-learn validation API.
- Resolve portable test and cache paths on Windows and macOS Intel.
- Keep NumPy below 2 on macOS Intel for compatibility with its PyTorch 2.2
  wheels; other platforms retain current NumPy support.

### Deprecated

- Delegate `molfeat.utils.commons.fold_count_fp` and `align_conformers` to
  Datamol. The compatibility wrappers keep their signatures and output types;
  new code should use `datamol.fold_count_fp` and
  `datamol.conformers.align_conformers` directly.

### Removed

- Remove DGL, DGLLife, Graphormer, `bio-embeddings` and their adapters; their
  upstream releases and binary constraints are incompatible with the supported
  modern stack. PyTorch Geometric is the maintained graph backend.
- Remove protein featurizers and their plugin entry-point group to focus the
  package on small molecules.

### Compatibility and delivery

- Validate the core on Linux, Windows, macOS Apple Silicon and macOS Intel;
  run `molfeat[all]` and published-checkpoint integration tests in a separate
  Linux job.
- Keep publication manual through the `release` action and `PYPI_API_TOKEN`,
  with PEP 740 attestations. Release tests and isolated wheel/source checks
  gate publication; prereleases never replace the stable documentation.
- Add a non-publishing dry run and a [release guide](docs/releasing.md).
  Conda-forge remains a separate channel requiring recipe updates.

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
