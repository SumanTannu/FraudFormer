import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prediction_api


def test_health_and_model_registry_are_available():
    assert prediction_api.health()["status"] == "ok"
    models = prediction_api.available_models()
    identifiers = {model["id"] for model in models}
    assert {"fraudformer-deberta", "deberta", "roberta", "bert"} <= identifiers
    assert sum(model["recommended"] for model in models) == 1


def test_unknown_model_is_rejected():
    with pytest.raises(HTTPException) as error:
        prediction_api.load_model("not-a-model")
    assert error.value.status_code == 404


def test_compiled_frontend_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "web/dist/index.html").is_file()

