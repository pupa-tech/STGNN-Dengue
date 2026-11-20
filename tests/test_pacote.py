"""Teste mínimo: o pacote instala e importa."""

import dengue


def test_pacote_importa():
    assert dengue.__version__ == "0.1.0"
