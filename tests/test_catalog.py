import re

import pytest

from jomini import __version__, catalog


def test_every_dataset_cites_known_references() -> None:
    for d in catalog.DATASETS.values():
        assert d.cite and all(k in catalog.REFERENCES for k in d.cite)
        assert d.access in ("bundled", "fetched") and d.licence
    for keys in catalog.METHODS.values():
        assert all(k in catalog.REFERENCES for k in keys)


def test_bibtex_is_well_formed_and_deduplicated() -> None:
    text = catalog.bibtex(["kursk", "lucas2004", "kursk_fits"])
    keys = re.findall(r"@\w+\{([^,]+),", text)
    assert keys.count("lucas2004") == 1 and "jomini" in keys and "bracken1995" in keys
    assert text.count("{") == text.count("}")
    assert __version__ in text


def test_short_labels() -> None:
    assert catalog.REFERENCES["lucas2004"].short == "Lucas & Turkes (2004)"
    assert catalog.REFERENCES["brunton2016"].short == "Brunton et al. (2016)"
    assert catalog.REFERENCES["caa1991"].short.startswith("U.S. Army")


def test_unknown_keys() -> None:
    with pytest.raises(KeyError, match="known"):
        catalog.dataset("narnia")
    with pytest.raises(KeyError):
        catalog.references_for(["narnia"])
