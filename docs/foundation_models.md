# Recent foundation models

Molfeat includes focused adapters for recent small-molecule representation
models when they can be supported without making the default installation
heavier. Protein and nucleotide models are deliberately outside this scope. The
model authors' weights and licences remain authoritative.

## Third-party licensing policy

Molfeat's source and its required or optional Python dependencies must remain
compatible with the Apache-2.0 project. New integrations must not copy or add a
runtime dependency on GPL, LGPL or AGPL software. A restrictively licensed
external model can only be exposed when Molfeat redistributes neither its code
nor its weights, the download comes from the publisher, and the caller must
explicitly accept the external terms. Mol-JEPA follows this isolated opt-in
pattern.

Every newly registered pretrained artifact must declare both its checkpoint
licence and, when available, the authoritative licence URL. A permissive code
repository licence is not assumed to cover separately published model weights.

[CLAMP](https://github.com/ml-jku/clamp) is deliberately not integrated. Its
authoritative licence file places the software under GPL-3.0 and adds separate
model terms, while its package metadata claims BSD-2-Clause. Molfeat will not
depend on, copy or redistribute CLAMP until the publisher provides consistent,
explicitly compatible terms.

## CheMeleon

[CheMeleon](https://arxiv.org/abs/2506.15792) is a 2025 descriptor-prediction
foundation model. Molfeat reproduces the inference path of the authors'
Chemprop D-MPNN with its existing RDKit and PyTorch dependencies, so Chemprop
is not installed at runtime. The official 2,048-dimensional checkpoint is
downloaded from the authors' [Zenodo record](https://doi.org/10.5281/zenodo.15460715)
and verified against its published MD5 checksum.

```python
from molfeat.trans.pretrained import CheMeleonTransformer

featurizer = CheMeleonTransformer()
features = featurizer(["CCO", "c1ccccc1"])
assert features.shape == (2, 2048)
```

By default the checkpoint is cached in the platform-specific Molfeat cache
directory under `chemeleon/chemeleon_mp.pt`. Pass `checkpoint_path` to use a
controlled cache or an already downloaded checkpoint.

## Mol-JEPA

[Mol-JEPA](https://arxiv.org/abs/2608.22642) is a 2026 joint-embedding model
with sequence, graph and fingerprint views. Its
[official implementation](https://github.com/Boehringer-Ingelheim/mol-jepa)
and Hugging Face checkpoint use custom Python code and are licensed CC BY-NC
4.0. Molfeat does not redistribute either one. Loading therefore requires the
existing Transformer and PyTorch Geometric extras, an immutable revision, and
two explicit acknowledgements:

```bash
uv add "molfeat[transformer,pyg]"
# or: python -m pip install "molfeat[transformer,pyg]"
```

```python
from molfeat.trans.pretrained import MolJEPATransformer

featurizer = MolJEPATransformer(
    output="cls",
    trust_remote_code=True,
    accept_noncommercial_license=True,
)
features = featurizer(["CCO", "c1ccccc1"])
assert features.shape == (2, 512)
```

Inspect the pinned remote repository before enabling `trust_remote_code`, and
confirm that the intended use is compatible with the upstream licence. The
`predictions` and `embeddings` outputs are available as flattened vectors of
6,144 and 6,656 values respectively.

Transformers 5 currently exposes a meta-device initialization defect in the
upstream custom model constructor. Molfeat detects that exact failure, builds
the pinned model on CPU, and strictly loads the pinned `model.safetensors`
checkpoint before moving it to the requested device.

## Where existing pretrained models are stored

`ModelStore` discovers model cards from
`https://fs.molfeat.datamol.io/artifacts/` by default. Downloaded artifacts are
placed in the operating system's Molfeat user cache (for example,
`~/Library/Caches/molfeat` on macOS). A `PretrainedModel` can override this with
`cache_path`, while `MOLFEAT_MODEL_STORE_ROOT` selects another remote or local
store. S3 and Google Cloud paths require `molfeat[cloud]`; the public HTTPS
store does not.

## Models evaluated but not bundled

SMI-TED, MuMo and Uni-Mol2 were reviewed for this release. SMI-TED's official
Hugging Face repository is Apache-2.0, but its current `config.json` declares an
unregistered `SMI-TED` model type without an `auto_map`. The existing generic
Hugging Face adapter was tested against revision
`414c3ea0a8603ef49d1c5bb3db336e09877c01ce` and cannot load it with
Transformers 5. Molfeat will use the generic adapter if the upstream repository
adopts a standard Transformers inference contract; it will not duplicate the
large custom implementation in the meantime. MuMo and Uni-Mol2 likewise remain
out because their current reference implementations require custom or
substantially heavier stacks.
