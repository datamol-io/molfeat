class InvalidEntryPointTypeError(Exception):
    """Raised when an entry point is not of the correct type."""

    ...


class LoadingEntryPointError(Exception):
    """Raised when an entry point cannot be loaded."""

    ...


class MissingEntryPointError(Exception):
    """Raised when an entry point cannot be found."""

    ...


class MultipleEntryPointError(Exception):
    """Raised when multiple entry points are found for a given name."""

    ...
