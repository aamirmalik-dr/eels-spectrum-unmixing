"""Guard that the package version and the installed distribution metadata agree."""

from importlib.metadata import version

import eelsunmix


def test_version_matches_distribution_metadata() -> None:
    assert eelsunmix.__version__ == version("eelsunmix")
