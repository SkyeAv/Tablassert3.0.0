__author__ = "Skye Lane Goetz"


import pytest


@pytest.fixture
def eg_table_config() -> dict[str, object]:
    return {
        "template": None,
        "sections": [
            {
                "location": {
                    "download_from": r"https://website.com/file.csv",
                    "ext": "csv",
                    "params": {"delimiter": ","},
                },
                "provenance": {
                    "publication_id": "PMC:18930937",
                    "curator": "person",
                    "org": "organization",
                },
                "attributes": None,
                "triple": {
                    "subj": {"mode": "value", "value": "A"},
                    "obj": {"mode": "value", "value": "A"},
                    "pred": "biolink:pred",
                },
            }
        ],
    }
