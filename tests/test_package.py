import mofqi_reservoir


def test_package_version():
    """The installed package exposes its expected version."""
    assert mofqi_reservoir.__version__ == "0.1.0"