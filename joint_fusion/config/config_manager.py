from pydantic.dataclasses import dataclass
from pydantic import Field

from typing import Optional, Dict, Any
import yaml
from datetime import datetime


@dataclass
class DataConfig:
    input_base_path: str
    input_base_mapping_data_json_path: str
    input_wsi_path: str
    h5_file: str
    json_file: str
    only_create_new_data_mapping: bool = False
    create_new_data_mapping: bool = False
    create_new_data_mapping_h5: bool = False
    num_tiles_per_slide: int = 400


@dataclass
class ModelConfig:
    embedding_dim_wsi: int = 384
    embedding_dim_omic: int = 256
    mlp_layers: int = 4
    dropout: float = 0.2
    fusion_type: str = "joint"
    joint_embedding: str = "weighted_avg"
    input_mode: str = "wsi_omic"
    wsi_fm: str = "uni"
    pooling: str = "attention"
    use_pretrained_omic: bool = False
    wsi_encoder_tile_batch_size: int = 64
    omic_checkpoint_path: Optional[str] = (
        None  # only use when use_pretrained_omic is True
    )


@dataclass
class TrainingConfig:
    batch_size: int = 80
    val_batch_size: int = 70
    lr: float = 0.0001
    num_epochs: int = 50
    n_folds: int = 5
    use_mixed_precision: bool = True
    plot_survival_distributions: bool = True
    contrast_loss_weight: float = 0.0
    contrast_wsi_weight: float = 0.4
    contrast_omic_weight: float = 0.4
    contrast_joint_weight: float = 0.2
    sim_loss_weight: float = 1.0
    random_state: int = 40


@dataclass
class TestingConfig:
    output_base_dir: Optional[str] = None
    test_batch_size: int = 1
    model_path: Optional[str] = None
    calc_saliency_maps: bool = False
    calc_IG: bool = False
    calc_attention_maps: bool = False
    saliency_tile_selection: str = "all"
    saliency_attention_scores_dir: Optional[str] = None
    # Full-context saliency over ALL tiles via two-stage (chunked) backprop.
    # Mathematically identical to the all-at-once backward (derivation in
    # compute_saliency_chunked) but bounds memory; chunk_size = tiles per
    # stage-2 pass. saliency_self_check validates the identity on a small bag
    # against the reference all-at-once backward before the full run.
    saliency_chunked: bool = False
    saliency_chunk_size: int = 16
    saliency_self_check: bool = False
    # Riemann steps for integrated gradients. The completeness error (logged per
    # patient to ig_completeness.csv) shrinks with more steps; 10 was too coarse.
    ig_steps: int = 50


@dataclass
class SchedulerConfig:
    type: str = "cosine_warmer"
    cosine_warmer: Dict[str, Any] = Field(
        default_factory=lambda: {
            "T_0": 10,
            "T_mult": 2,
            "eta_min": 1e-6,
            "decay_factor": 0.5,
        }
    )
    exponential: Dict[str, Any] = Field(default_factory=lambda: {"gamma": 0.95})
    step_lr: Dict[str, Any] = Field(
        default_factory=lambda: {"step_size": 10, "gamma": 0.95}
    )


@dataclass
class GPUConfig:
    gpu_ids: str = "0"
    use_multi_gpu: bool = False


@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    checkpoint_dir: Optional[str] = None
    profile: bool = False


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    testing: TestingConfig
    scheduler: SchedulerConfig
    gpu: GPUConfig
    logging: LoggingConfig
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        if self.logging.checkpoint_dir is None:
            self.logging.checkpoint_dir = f"checkpoints/checkpoint_{self.timestamp}"

        if self.testing.output_base_dir == "" or self.testing.output_base_dir == None:
            self.testing.output_base_dir = (
                f"checkpoints/checkpoint_{self.timestamp}/test_results"
            )


class ConfigManager:
    @staticmethod
    def load_config(config_path: str, overrides: Optional[Dict] = None) -> Config:
        """Load configuration from YAML file with optional overrides"""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if overrides:
            config_dict = ConfigManager._deep_update(config_dict, overrides)

        return Config(
            data=DataConfig(**config_dict["data"]),
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            testing=TestingConfig(**config_dict["testing"]),
            scheduler=SchedulerConfig(**config_dict["scheduler"]),
            gpu=GPUConfig(**config_dict["gpu"]),
            logging=LoggingConfig(**config_dict["logging"]),
        )

    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                base_dict[key] = ConfigManager._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict

    @staticmethod
    def save_config(config: Config, checkpoint_dir: str):
        import dataclasses
        from pathlib import Path

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        config_path = checkpoint_dir / "config.yaml"

        with open(config_path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)
