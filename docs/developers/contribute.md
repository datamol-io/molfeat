# Contribute

Welcome to the molfeat community! We appreciate everyone's contribution and welcome all. Apart from code contributions, there are various other ways to support the community. If you would like to get involved, please see the [community documentation](../community/get_involved.md) for more details on how to contribute. However you choose to contribute, please be mindful and respect our [code of conduct](https://github.com/datamol-io/molfeat/blob/main/.github/CODE_OF_CONDUCT.md).

!!! note annotate "Get inspired by community contributions"
    As we love community contributions, we decided to create a dedicate space in our documentation to highlight and celebrate such contributions.
    Visit [Community Contribitons](../community/contributions.md) to learn more!

The rest of this page details the development lifecycle of molfeat.

## Setup a dev environment

To contribute, you will first need to setup a dev environment. Follow the steps below:

1. Fork the [repository](https://github.com/datamol-io/molfeat) by
   clicking on the **[Fork](https://github.com/datamol-io/molfeat/fork)** button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your Github handle>/molfeat.git
   cd molfeat
   git remote add upstream https://github.com/datamol-io/molfeat.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b useful-branch-name
   ```

   **Do not** work on the `main` branch!

4. Once you have a local copy, create the complete development environment:

   ```bash
   uv sync --all-extras
   ```

   This creates an isolated `.venv` and installs every maintained Molfeat extra
   in editable mode. The checked-in `env.yml` remains available when a Conda
   environment is required.
   
5. Make your changes and modifications on your branch.

   As you work on your code, you should make sure the test suite passes. Run the tests impacted by your changes like this:

   ```bash
   uv run pytest tests/<TEST_TO_RUN>.py
   ```

6. Commit your code, push it to your forked repository and open a pull request with a detailed description of your changes and why they are valuable. 

## Continuous Integration

Molfeat uses GitHub Actions to:

- **Build and test** Molfeat across the supported Python, RDKit and operating-system matrix.
- **Validate every maintained extra** and the published foundation-model artifacts on every push, pull request and weekly schedule.
- **Check** formatting with Black, linting with Ruff, and both package distributions.
- **Build documentation** on every change and deploy it from `main` and published releases.

## Run tests globally

```bash
uv run python -m pytest -m "not integration"
uv run python -m pytest -m integration --no-cov -n 0
```

The second command downloads and validates the maintained public model
artifacts. GitHub Actions additionally installs `molfeat[all]` and runs the
complete suite in that environment. The fast core suite runs on Linux x86-64,
Windows x86-64, macOS Apple Silicon and macOS Intel; network-backed model
integrations run separately on Linux so failures remain diagnosable.

## Build the documentation

You can build and serve the documentation locally with:

```bash
# Build and serve the doc
uv run mkdocs serve
```

## Submitting Pull Requests

If you're considering a large code contribution to molfeat, please open an issue first to get early feedback on the idea.

Once you think the code is ready to be reviewed, push it to your fork and open a pull request. We will assign a reviewer to your PR.
For a change to be accepted all existing tests will need to pass. We expect additional tests and documentation for any new features.

If you are developing a plugin for molfeat, please refer to the corresponding section [Extending molfeat](./create-plugin.md)

## Adding ETL Notebooks

The ETL (extraction, transformation, and loading) scripts document the process of creating new featurizers, and we make our ETL notebooks open to the community for transparency purposes. As a developer adding new featurizers, please document your process in the [etl notebook folder](https://github.com/datamol-io/molfeat/tree/main/nb/etl).

By documenting your process in the ETL notebook, you help ensure that the registration of new models can be reviewed by the community and provide greater visibility into the development process. This can help build trust with our users and contributors, and encourage collaboration and feedback.

## Releasing a New Version

To release a new version, code maintainers publish a GitHub Release after
coordinating with the code owners. The workflow builds and smoke-tests both
distributions with uv, generates PEP 740 attestations, and publishes to PyPI
through Trusted Publishing. The conda-forge feedstock remains supported: its
update bot opens the version change after PyPI publication, and maintainers
review the feedstock dependency and test changes separately.

The release roadmap should be followed to ensure that the new version is stable, functional, and meets the requirements of the release. This includes proper testing, documentation, and ensuring backward compatibility where necessary. By following these guidelines, we can ensure that new versions are released smoothly and efficiently, with clear communication to our users and contributors.
