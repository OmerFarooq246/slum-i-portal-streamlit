from config.models import MODEL_REGISTRY, get_available_models


def test_model_registry_contains_supported_cities() -> None:
    models = get_available_models()

    assert models is MODEL_REGISTRY
    assert set(models) == {"enb5_islamabad", "enb5_lahore"}
    assert {region for model in models.values() for region in model.regions} == {
        "Islamabad",
        "Lahore",
    }


def test_model_registry_uses_distinct_weight_files() -> None:
    weight_paths = [model.weights_path for model in MODEL_REGISTRY.values()]

    assert len(weight_paths) == len(set(weight_paths))
    assert all(path.startswith("models/") and path.endswith(".h5") for path in weight_paths)
    assert all(model.batch_size > 0 for model in MODEL_REGISTRY.values())
