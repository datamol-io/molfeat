# Plugins

Molecular featurization is an active area of research leading to the steady emergence of new approaches to solve this complex set of problems. As new molecular featurizers emerge, you can easily add yours to molfeat and share it with the rest of the community.

For developers that are planning to extend molfeat functionality, we recommend using the [plugin system](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/). The use of a plugin system ensures that the core package remains easy to install and as light as possible, while making it easy to extend its functionality with plug-and-play components. Additionally, it ensures that plugins can be developed independently of the core package, removing the bottleneck of a central party that reviews and approves new plugins.

However, plugins are not always required and sometimes a simple pull request is the better option. 

| :heavy_check_mark: **Do** use plugins if...                                          | :x: **Do not** use plugins if...                                                                                           |
|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| If a new featurizer can not be loaded by current classes.                            | If a new featurizer can already be loaded through existing classes.                                                        |
| If a new featurizer requires additional, possibly difficult-to-install dependencies. | If a new featurizer requires no additional dependencies.                                                                   |
| If you want to make a new class of featurizers available to the molfeat community.   | If you can show through benchmarks that a featurizer is so performant, it should be available as part of the core package. |
| If you want to extend the functionality of molfeat for private / internal use.       |                                                                                                                            |


!!! note annotate "Decided you don't need a plugin after all?"
    Consult the [tutorials](../tutorials/add_your_own.ipynb) to learn how to extend molfeat's functionality without plugins.

The rest of this document details how to *package* molfeat plugins (or _extensions_) so that they can be tested, published and eventually reused by others. 

We recommend molfeat plugins be bundled and distributed as a [Python package](https://docs.python.org/3/tutorial/modules.html#packages) that provides a set of extensions to molfeat.

## Quickstart

The fastest way to jumpstart a molfeat plugin package is to start with an existing template repository.
The [molfeat-padel](https://github.com/datamol-io/molfeat-padel) demo plugin package provides a starting point to understand the basic folder structure and our naming convention. Please reachout to us on github if you need help to get started.

In the following document, we explain the conventions used when building a molfeat plugin.

## Choosing a name

The naming convention for molfeat plugin packages is `

- `molfeat-<myplugin>` for the plugin distribution on [PyPI](https://pypi.python.org) or [Conda](https://docs.conda.io/en/latest/).
- `molfeat_<myplugin>` for the corresponding python package (since python package names cannot contain dashes), leading to the following folder structure:

```
    molfeat-myplugin/
       molfeat_myplugin/
          __init__.py
```

If you intend to eventually publish your plugin package, please go to the [Register a plugin](./register-plugin.md) and choose a name that is not already taken. You are also encouraged to pre-register your package (see instructions provided), both to reserve your plugin name and to inform others of your ongoing development.

## Folder structure

The overall folder structure of your plugin is up to you, but it is
useful to follow a set of basic conventions. Here is an example of a
folder structure for a molfeat plugin, illustrating different levels of
nesting

    molfeat-myplugin/           - distribution folder
       tests/                   - tests directory (possibly with subdirectories)
       molfeat_myplugin/        - top-level package (from molfeat_myplugin import ..)
          __init__.py
          calc/
             __init__.py
             myplugin.py      - contains my plugin SerializableCalculator
          trans/
             __init__.py
             myplugin.py       - contains my plugin MoleculeTransformer
          data/
             __init__.py    - contains code-specific MyData data format
          ...
       setup.py             - setup.py file (optional based on your pyproject.toml content)
       LICENSE              - license of your plugin
       MANIFEST.in          - lists non-python files to be installed, such as LICENSE
       README.md            - project description for github and PyPI
       pyproject.toml       - plugin metadata: installation requirements, author, entry points, etc.
       ...


## Registering plugins through entry points

A molfeat plugin is an extension of molfeat that announces itself by means of a new *entry point*. Adding a new entry point consists of the following steps:

> 1.  Deciding on a name. We *strongly* suggest starting the name of each
>     entry point with the name of the plugin package (omitting the
>     'molfeat-' prefix). For a package `molfeat-myplugin`, this will
>     usually mean `"myplugin.<something>"`
>
> 2.  Finding the right entry point group. Three main entry point are defined in molfeat `molfeat.calc` (for single molecule calculator: `SerializableCalculator`, which your class should inherit from), `molfeat.trans` (for `MoleculeTransformers`) and `molfeat.trans.pretrained` (for `PretrainedMolTransformer`)
>
> 3.  Adding the entry point to the `entry_points` field in the
>     `pyproject.toml` file:
>
>         ...
>         [project.entry-points."molfeat.calc"]
>         "myplugin.<something>" = "molfeat_myplugin.calc.some:MysomethingCalculator"
>         [project.entry-points."molfeat.trans"]
>         "myplugin.<something>" = "molfeat_myplugin.trans.some:MysomethingMoleculeTransformer"
>         ...


For further details, see the Python [packaging user guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).


The core molfeat package will automatically discover new molfeat plugins and appropriately make them accessible and importable. 

For example, upon proper registration of a plugin offering a new `SerializableCalculator`, it should be directly accessible in the list of available calculators for `MoleculeTransformer`.

The following example shows how to discover the newly added functionality of the `molfeat-padel` plugin package automatically when installed. 
In this example, all three scenarios are valid.

#### 1. Initializing the calculator through the plugin package.

```python

from molfeat.trans import MoleculeTransformer

from molfeat_padel.calc.padel import PadelDescriptors
mol_transf = MoleculeTransformer(featurizer=PadelDescriptors())
```

#### 2. Explicitly load registered plugins to automatically discover `PadelDescriptors`.

```python
from molfeat.trans import MoleculeTransformer
from molfeat.plugins import load_registered_plugins
load_registered_plugins(add_submodules=True)

# PadelDescriptors is now available under the core `molfeat.calc`
from molfeat.calc import PadelDescriptors
mol_transf = MoleculeTransformer(featurizer=PadelDescriptors())
mol_transf = MoleculeTransformer(featurizer="PadelDescriptors")
```

#### 3. Import the plugin to automatically discover the `PadelDescriptors`.

```python
from molfeat.trans import MoleculeTransformer
import molfeat_padel

# works because PadelDescriptors is imported in the root init of molfeat_padel
mol_transf = MoleculeTransformer(featurizer="PadelDescriptors")
```


## Testing a plugin package

Writing tests for your molfeat plugins and running continuous integration
tests using free platforms like [GitHub
Actions](https://github.com/features/actions) is the best way to ensure
that your plugin works and keeps working as it is being developed. We
recommend using the [pytest](https://pytest.org) framework for testing molfeat plugins.


## Documenting a plugin package

molfeat plugin packages are python packages, and general [best practises
for writing python documentation](https://docs.python-guide.org/writing/documentation/) apply.

In the following, we mention a few hints that apply specifically to molfeat plugins.

### Repository-level documentation

Since the source code of most molfeat plugins is hosted on GitHub, the
first contact of a new user with your plugin package is likely the
landing page of your GitHub repository.

> -   Make sure to have a useful `README.md`, describing what your
>     plugin does, how to install it and how to run it.
> -   Leaving a contact email and adding a license is also a good idea.
> -   Make sure the information in the `pyproject.toml` file is correct
>     and up to date (in particular the version number)
> -   Optionally add a documentation website to provide tutorials

### Source-code-level documentation

Source-code level documentation matters both for users of your
plugin's python API and, particularly, for attracting contributions
from others.

When adding new types of calculations or workflows, make sure to use
[docstrings](https://www.python.org/dev/peps/pep-0257/#what-is-a-docstring),
and use the `help` argument to document input ports and output ports.


## Registering your plugin as official molfeat plugin

Once you have designed and tested your plugin package, you can officially register it to be listed on the molfeat
website by following the instructions at [Register a plugin](./register-plugin.md) 


