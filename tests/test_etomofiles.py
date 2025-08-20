import etomofiles


def test_imports_with_version():
    assert isinstance(etomofiles.__version__, str)
