# MolFeat plugin registry

This document explains how to officially register a plugin in `MolFeat`. 

The `MolFeat` plugin registry aims to be the home for all publicly available `MolFeat` plugins. It
collects information on the type of plugins provided by your package, which `MolFeat` versions it is compatible with, etc.

If you are starting to develop a new plugin or if you already have one, please register it here.
We strongly encourage to **register at early stages of development**, both to reserve the name of your plugin and to inform the community of your ongoing work.



## How to register a plugin

1. Fork this repository
2. Add your plugin to the end of the `plugins.yaml` file, e.g.
    ```
    ...
    molfeat-new:
        entry_point_prefix: new
        available_model_infos: ~
        home_url: ~
        documentation_url: ~ 

    ```
3. Create a [Pull Request](https://github.com/datamol-io/molfeat/pulls) to this repository

### Valid keys for each plugin

- __top-level key__ (required):
The name under which your plugin will be distributed.
By convention, names of MolFeat plugins are lowercase and prefixed by `molfeat-` (e.g `molfeat-padelpy`)

- __entry_point_prefix__ (required):
The prefix of all entry points provided by the plugin.
By convention, a plugin `molfeat-xxx` should use `entry_point_prefix: xxx`.
For example: `molfeat-padelpy` uses the entry point prefix `padelpy` and provides numerous entry points, all of which start with `padelpy.`.

- __available_model_infos__ (required):
A path to a json file specifying the information (model card) of all models that are offered in your plugin. For more information about a model card. Please refer to the [create-plugin](./create-plugin.md) section.

- __home_url__ (required):
The link to the homepage of the plugin, for example its github repository.

- __molfeat_version__ (required):
The molfeat version required for your plugin to work.

- __documentation_url__ (optional):
The link to the online documentation for your plugin, for example on readthedocs.org .

