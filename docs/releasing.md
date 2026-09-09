# Releasing molfeat

Publication is manual. Merging into `main`, pushing a tag or publishing a
GitHub Release does not upload a package to PyPI.

## Before publishing

1. Finalize the release notes in `CHANGELOG.md` with an exact version heading,
   for example `## 1.0.0 - YYYY-MM-DD`. Date the release and update README
   and website wording that still calls it unreleased or specific to `dev`.
2. Merge the release changes into `main`, preserving contributor history.
3. Confirm that PyPI's Trusted Publisher matches the `datamol-io/molfeat`
   repository, `.github/workflows/release.yml` workflow and `pypi` environment.
4. Confirm tests pass with published dependencies. Datamol 0.13.0 or newer is
   required and is available from PyPI.

## Run the release action

Open **Actions → release → Run workflow**. Select `main`, enter
`1.0.0` in **release-version**, and leave **dry-run** checked for a
rehearsal. Dry runs may also run from `dev` and accept unfinished release
notes. They create a tag only inside the temporary runner checkout, never on
GitHub, and cannot publish packages or documentation.

For publication, launch the action again from `main` with **dry-run**
unchecked. Use canonical versions without a `v` prefix, such as `1.0.0`
or `1.0.0rc1`; `a`, `b` and `rc` suffixes identify prereleases.

The action validates the version and release notes, reruns the complete test
and quality workflows on the selected commit, and builds the distributions.
Both wheel and source installations are checked with Python's isolated mode,
including their version and import location. Documentation must also build
successfully before anything is uploaded.

The publish job uses PyPI Trusted Publishing: GitHub exchanges its short-lived
OpenID Connect identity for a project-scoped PyPI token. No long-lived PyPI
credential is stored in GitHub. The action also generates and uploads PEP 740
attestations for both distributions.

Only after PyPI succeeds does the action create the GitHub tag and Release
at the tested commit, then deploy versioned documentation. A prerelease never
moves the `stable` documentation alias. A failed upload leaves that alias
unchanged. If a later step fails, rerun the failed jobs rather than rebuilding
an already published version.

## Conda-forge

The [feedstock](https://github.com/conda-forge/molfeat-feedstock) is a separate
release channel. Its update bot proposes version changes after PyPI publication,
but maintainers must review dependencies and run the recipe tests there.
The package release action does not publish to conda-forge.

Remove the old Python 3.10 upper bound, require Python 3.11+, and align the
core and optional dependencies with `pyproject.toml`. Preserve the macOS Intel
PyTorch 2.2 / NumPy 1.26 / Transformers 4.57 exception and the SELFIES extras.
A version-only bot update is insufficient.

Do not merge a version-only feedstock update for this major release. Its
dependency metadata and import/CLI tests must reflect the new installation
boundaries.
