# Migrating to Molfeat 1.x

Molfeat 1.x is a deliberate platform reset. It keeps the established
featurization APIs while moving the supported environment to versions that are
actively maintained and available on current Python releases.

## Supported runtime

- Python 3.11 through 3.14
- RDKit 2024.09 or newer
- PyTorch 2.5 or newer
- NumPy 1.26 or newer
- pandas 2.2 or newer
- scikit-learn 1.6 or newer
- Transformers 4.57 through 5.x for the `transformer` extra. The lower bound keeps the extra co-installable with SAFE, whose constrained-generation API still requires Transformers 4.x.

Create a fresh environment rather than upgrading a long-lived 0.x environment
in place:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "molfeat[transformer,pyg]"
```

## Optional dependency changes

`molfeat[all]` now means all maintained extras compatible with the current
interpreter. DGL 1.x and DGLLife are restricted to Python 3.11 and tested in a
separate compatibility job:

```bash
python -m pip install "molfeat[dgl]"
```

The `graphormer` extra has been removed. The latest
`graphormer-pretrained` source release does not build with the supported Python
and Cython toolchain. `bio-embeddings` is likewise no longer a maintained extra
because its dependency stack predates the supported runtime. The corresponding
adapter classes have not been deleted, so controlled legacy environments can
still provide those dependencies themselves.

## Observable behavior corrections

- Creating `ModelStore` or a Hugging Face transformer no longer fetches the
  remote model index. The first operation that needs discovery loads it.
- `Pharmacophore3D(..., raw=True)` now returns a sorted one-dimensional integer
  array. Earlier releases returned a zero-dimensional object array containing a
  Python set.
- Pooling masks consistently use true values for elements included in max, mean
  and sum pooling. Max pooling previously inverted this meaning and failed on
  current PyTorch because it passed a floating-point mask to `masked_fill`.
- FCFP fingerprints retain the same feature invariants across supported RDKit
  releases, and atom/bond calculations use the current valence API.
- Scikit-learn calls use `ensure_all_finite`, replacing the removed
  `force_all_finite` keyword.

## Serialization

The default values included in serialized transformer state now match the
constructor defaults. Existing state files remain supported where their
calculator and optional dependencies are available. Re-save important models
after validating them in 1.x so future environments record the current package
metadata.
