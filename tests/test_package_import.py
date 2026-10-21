import importlib


def test_package_uses_lowercase_python_import_name() -> None:
    package = importlib.import_module("war_ai")

    assert package.__name__ == "war_ai"
