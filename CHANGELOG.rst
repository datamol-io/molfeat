==================
molfeat Change Log
==================

.. current developments

v0.8.7
====================

**Added:**

* Support for `ignore_padding` in Graphormer
* More flexibility overall for graphormer embeddings

**Changed:**

* Phased out rdchem.Mol, rdchem.Atom and rdkit.Bond in favor of datamol versions
* Fully automated release process.

**Fixed:**

* Random logging in cache coming from testing if an input is a molecule
* Some small typos in doc strings
* Naming of JTVAE models
* Fix issue #37 by making WeaveFeaturizer faster
* Usage card for rdkit and fingerprints/descriptors featurizers

**Authors:**

* Hadrien Mary
* maclandrol
* rbyrne-momatx



v0.8.6
====================

**Added:**

* Support for batch transformation in `MoleculeTransformer` for calculators that implements `batch_compute`

**Changed:**

* Pull request template for better directive.

**Authors:**

* maclandrol



v0.8.5
====================

**Authors:**




v0.8.4
====================

**Added:**

* Add Google Analytics support.

**Removed:**

* Remove support for the `np.float128` dtype (issue #26)

**Fixed:**

* Color bug of the search input bar

**Authors:**

* Cas Wognum
* Honore Hounwanou
* maclandrol



v0.8.3
====================

**Added:**

* More documentation and tutorials
* pip dependencies and optional dependencies in pyproject.toml
* Github issue templates

**Changed:**

* Updated all occurrences of old molfeat links with new ones.
* Documentation and readme

**Removed:**

* duplicated CODEOWNER file

**Fixed:**

* Bug in serialization of transformers with a serializable calculator
* Minor typos and function definition
* Links in pyprojects

**Authors:**

* Cas Wognum
* Hadrien Mary
* Honore Hounwanou
* Saurav Maheshkar
* Therence
* maclandrol



v0.8.1
====================

**Added:**

* Extended the simple benchmark in the docs to also include a search benchmark, based on the RDKit benchmarking platform
* Added missing API documentation for `molfeat.plugins` and `molfeat.store`.

**Changed:**

* Changed the styling of the docs to match the new datamol.io styling.

**Authors:**

* Cas Wognum
* Hadrien Mary
* Therence



v0.8.0
====================

**Authors:**

* Hadrien Mary


