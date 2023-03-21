# Plugins

For developers that are planning to extend `MolFeat` functionalities, we recommend using the [plugin system](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/). 

The following document focuses on how to *package* `MolFeat` extensions (plugins) so that they can be tested, published and eventually reused by others.

`MolFeat` plugins can be bundled and distributed in a [Python package](https://docs.python.org/3/tutorial/modules.html#packages) that provides a set of extensions to `MolFeat`.


## Quickstart

The fastest way to jumpstart a `MolFeat` plugin package is to use the
[MolFeat plugin template](...) in order to template the basic folder structure, already customized
according to the desired name of your plugin, following `MolFeat` conventions.

See also the
[molfeat-padel](https://github.com/datamol-io/molfeat-padel) demo plugin package for an in-depth explanation of the files & folders.

In the following, we explain some of the conventions used when building a `MolFeat` plugin.

## Choosing a name

The naming convention for `MolFeat` plugin packages is `

- `molfeat-<myplugin>` for the plugin distribution on [PyPI](https://pypi.python.org) or [Conda](https://docs.conda.io/en/latest/) 
- `molfeat_<myplugin>` for the corresponding python package (since python package names cannot contain dashes), leading to the following folder structure:

    molfeat-myplugin/
       molfeat_myplugin/
          __init__.py

If you intend to eventually publish your plugin package, please go to the [Register a plugin](./register-plugin.md) and choose a name that is not already taken. You are also encouraged to pre-register your package (see instructions provided), both to reserve your plugin name and to inform others of your ongoing development.

## Folder structure

The overall folder structure of your plugin is up to you, but it is
useful to follow a set of basic conventions. Here is an example of a
folder structure for a `MolFeat` plugin, illustrating different levels of
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

A `MolFeat` plugin is an extension of `MolFeat` that announces itself by means of a new *entry point*. Adding a new entry point consists of the following steps:

> 1.  Deciding a name. We *strongly* suggest to start the name of each
>     entry point with the name of the plugin package (omitting the
>     \'molfeat-\' prefix). For a package `molfeat-myplugin`, this will
>     usually mean `"myplugin.<something>"`
>
> 2.  Finding the right entry point group. Three mains entry point are defined in molfeat `molfeat.calc` (for single molecule calculator: `SerializableCalculator`, which your class should inherit from), `molfeat.trans` (for `MoleculeTransformers`) and `molfeat.trans.pretrained` (for `PretrainedMolTransformer`)
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


The main `MolFeat` package will automatically discover new molfeat plugins and appropriately make them accessible and importable. 

For example, upon proper registration of a plugin offereing a new `SerializableCalculator`, it should be directly accessible in the list of available calculator for `MoleculeTransformer`.

The following example show how to leverage the `molfeat-padel` plugin package automatically when installed. In this example, all three scenarios are valid.

1. initializing the calculator from the plugin package

```python

from molfeat.trans import MoleculeTransformer

from molfeat_padel.calc.padel import PadelDescriptors
trans = MoleculeTransformer(featurizer=PadelDescriptors())
```

2. enable autodiscovery and addition of the `PadelDescriptors` as importable attribute to the entry point group `molfeat.calc`

```python
from molfeat.trans import MoleculeTransformer
from molfeat.plugins import load_registered_plugins
load_registered_plugins(add_submodules=True)

# this is now possible
from molfeat.calc import PadelDescriptors
trans = MoleculeTransformer(featurizer=PadelDescriptors())
```

3. auto discovery of PadelDescriptors 

```python
from molfeat.trans import MoleculeTransformer
import molfeat_padel

trans = MoleculeTransformer(featurizer="PadelDescriptors")
# works because PadelDescriptors is imported in the root init of molfeat_padel
```

```python
from molfeat.trans import MoleculeTransformer
from molfeat.plugins import load_registered_plugins
load_registered_plugins(add_submodules=True)
trans = MoleculeTransformer(featurizer="PadelDescriptors")
```


## Testing a plugin package

Writing tests for your `MolFeat` plugins and running continuous integration
tests using free platforms like [GitHub
Actions](https://github.com/features/actions) is the best way to ensure
that your plugin works and keeps working as it is being developed. We
recommend using the [pytest](https://pytest.org) framework for testing `MolFeat` plugins.


## Documenting a plugin package

`MolFeat` plugin packages are python packages, and general [best practises
for writing python documentation](https://docs.python-guide.org/writing/documentation/) apply.

In the following, we mention a few hints that apply specifically to `MolFeat` plugins.

### Repository-level documentation

Since the source code of most `MolFeat` plugins is hosted on GitHub, the
first contact of a new user with your plugin package is likely the
landing page of your GitHub repository.

> -   Make sure to have a useful `README.md`, describing what your
>     plugin does, how to install it and how to run it.
> -   Leaving a contact email and adding a license is also a good idea.
> -   Make sure the information in the `pyproject.toml` file is correct
>     and up to date (in particular the version number)
> -   Optionally add a documentation website to provide tutorials

### Source-code-level documentation

Source-code level documentations matters both for users of your
plugin\'s python API and, particularly, for attracting contributions
from others.

When adding new types of calculations or workflows, make sure to use
[docstrings](https://www.python.org/dev/peps/pep-0257/#what-is-a-docstring),
and use the `help` argument to document input ports and output ports.


## Registering your plugin as official MolFeat plugin

Once you have designed, and tested your plugin package, you can officially register it to be listed on the `MolFeat`
website by following the instructions at [Register a plugin](./register-plugin.md) 


