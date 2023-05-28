# molfeat plugin registry

This document explains how to register a plugin in `molfeat` for listing on our website. 

The `molfeat` plugin registry aims to be the home for all publicly available `molfeat` plugins. It
collects information on the type of plugins provided by your package and which `molfeat` versions it is compatible with.

If you are starting to develop a new plugin or if you already have one, please register it here.
We strongly encourage you to **register at early stages of development**, both to reserve the name of your plugin and to inform the community of your ongoing work.

!!! note annotate "Curious what other plugins exist?"
    Visit [Community contributions](../community/contributions.md) to learn more about the current plugins!

## How to register a plugin

1. Fork this repository
2. Add your plugin to the end of the `plugins.yaml` file, e.g.
    ```
    ...
    molfeat-new:
        entry_point_prefix: new
        home_url: ~
        documentation_url: ~
        molfeat_version: ~

    ```
3. Create a [Pull Request](https://github.com/datamol-io/molfeat/pulls) to this repository

### Valid keys for each plugin

- __top-level key__ (required):
the name under which your plugin will be distributed.
By convention, names of molfeat plugins are lowercase and prefixed by `molfeat-` (e.g `molfeat-myplugin`)

- __entry_point_prefix__ (required):
the prefix of all entry points provided by the plugin.
By convention, a plugin `molfeat-xxx` should use `entry_point_prefix: xxx`. You can also use the module name of your plugin. For example: `molfeat-myplugin` uses the entry point prefix `myplugin` and provides numerous entry points, all of which start with `myplugin.`. The entry point is also how you signal to users how they can load your plugin through molfeat. 

```python
from molfeat.trans import MoleculeTransformer
from molfeat.plugins import load_registered_plugins
load_registered_plugins(add_submodules=True, plugins=["new"])
```

- __home_url__ (required):
the link to the homepage of the plugin, for example its github repository.

- __molfeat_version__ (required):
the molfeat version required for your plugin to work.

- __documentation_url__ (optional):
the link to the online documentation for your plugin, for example on readthedocs.org .


### Model Card in pull request

In you pull request, please  provide a json or yaml file (or its content) that should list information on available models that are offered in your plugin. 

If your plugin does not provide any additional model, you can ignore this step.
An example of such a file is provided below. You will need to provide all keys:

```yaml
# name of the featurizer, str
name: awesome-featurizer 
# description of your featurizer, str
description: Concise description for the awesome-featurizer
# type of input the featurizer takes, str
inputs: smiles
# which group does the featurizer belong to. 
# This helps for the usage card. Ask a maintainer for help
group: "rdkit"
# link to the reference of the featurizer
reference: https://link-to-the-awesome-paper/
# type of featurizer, Literal["pretrained", "hand-crafted", "hashed"]
type: "pretrained" 
# output representation of the featurizer, Literal["graph", "line-notation", "vector", "tensor", "other"]
representation: vector 
# Whether 3D information are required, Optional[bool]
require_3D: false 
# up to 5 tags you want to add for searching your featurizer, List[str]
tags:
    - pretrained
    - 3D
    - GNN
authors: # list the authors, List[str]
    - Awesome Team 1
```