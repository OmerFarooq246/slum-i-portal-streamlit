from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelConfig:
    name: str
    api_model_name: str
    regions: List[str]
    weights_path: str
    batch_size: int = 2


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "enb5_lahore": ModelConfig(
        name="EfficientNet-B5 Segmentation",
        api_model_name="ENB5_Seg_PAK",
        regions=["Lahore"],
        weights_path="models/enb5_seg_lahore.h5",
        batch_size=2,
    ),
    "enb5_islamabad": ModelConfig(
        name="EfficientNet-B5 Segmentation",
        api_model_name="ENB5_Seg_PAK",
        regions=["Islamabad"],
        weights_path="models/enb5_seg_islamabad.h5",
        batch_size=2,
    ),
}


def get_available_models() -> Dict[str, ModelConfig]:
    return MODEL_REGISTRY
