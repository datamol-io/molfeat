# Overview

Molfeat is a python library to simplify molecular featurization. It supports a wide variety of molecular featurizers out-of-the-box and can be easily extended to add your own.

- :snake: Simple pythonic API.
- :rocket: Fast and efficient featurization.
- :arrows_counterclockwise: Unifies pre-trained embeddings and hand-crafted featurizers in a single package. 
- :heavy_plus_sign: Easily extend Molfeat with your own featurizers through plugins. 
- :chart_with_upwards_trend: Benefit from increased performance through a trouble-free caching system.

Visit our website at https://molfeat.datamol.io.

## Installation

### Installing Molfeat

Use mamba:

```bash
mamba install -c conda-forge molfeat
```

_**Tips:** You can replace `mamba` by `conda`._

_**Note:** We highly recommend using a [Conda Python distribution](https://github.com/conda-forge/miniforge) to install Molfeat. The package is also pip installable if you need it: `pip install molfeat`._


### Installing Plugins

The functionality of Molfeat can be extended through plugins. The usage of a plugin system ensures that the core package remains easy to install and as light as possible, while making it easy to extend its functionality with plug-and-play components. Additionally, it ensures that plugins can be developed independently of the core package, removing the bottleneck of a central party that reviews and approves new plugins. Consult the Molfeat documentation for more details on how to [create](developers/create-plugin.md) your own plugins.

This, however, does imply that the installation of a plugin is plugin-dependent: Please consult its documentation to learn more.

### Optional dependencies
Not all featurizers of the Molfeat core package are supported by default. Some featurizers require additional dependencies. If you try to use a featurizer that requires additional dependencies, Molfeat will raise an error and will tell you which dependencies are missing and how to install these.

## How to cite
Please cite Molfeat if you use it in your research: [![DOI]()]().