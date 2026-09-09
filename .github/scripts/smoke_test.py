"""Exercise the installed distribution without importing the checkout."""

import os
from importlib.metadata import distribution
from pathlib import Path

import molfeat as package
from molfeat.calc import FPCalculator

installed = distribution("molfeat")
location = Path(package.__file__).resolve()
assert location == Path(installed.locate_file("molfeat/__init__.py")).resolve(), location
assert not location.is_relative_to(Path(__file__).resolve().parents[2]), location
assert package.__version__ == installed.version
if expected := os.environ.get("EXPECTED_VERSION"):
    assert installed.version == expected, (installed.version, expected)

assert FPCalculator("ecfp")("CCO").shape == (2048,)
print(f"molfeat {installed.version}: {location}")
