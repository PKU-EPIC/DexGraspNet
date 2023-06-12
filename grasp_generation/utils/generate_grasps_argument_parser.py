from utils.hand_model_type import HandModelType
import math
from tap import Tap
from typing import List


class GenerateGraspsArgumentParser(Tap):
    # experiment settings
    hand_model_type: HandModelType = HandModelType.SHADOW_HAND
    wandb_name: str = ""
    visualization_freq: int = 2000
    result_path: str = "../data/graspdata"
    data_root_path: str = "../data/meshdata"
    object_code_list: List[str] = []
    all: bool = False
    overwrite: bool = False
    todo: bool = False
    seed: int = 1
    batch_size_each: int = 500
    max_total_batch_size: int = 1000
    n_iter: int = 6000

    # hyper parameters
    switch_possibility: float = 0.5
    mu: float = 0.98
    step_size: float = 0.005
    stepsize_period: int = 50
    starting_temperature: float = 18
    annealing_period: int = 30
    temperature_decay: float = 0.95
    n_contacts_per_finger: int = 1
    w_fc: float = 1.0
    w_dis: float = 300.0
    w_pen: float = 100.0
    w_spen: float = 100.0
    w_joints: float = 1.0
    w_ff: float = 1.0
    w_fp: float = 0.0

    # initialization settings
    jitter_strength: float = 0.1
    distance_lower: float = 0.2
    distance_upper: float = 0.3
    theta_lower: float = -math.pi / 6
    theta_upper: float = math.pi / 6

    # energy thresholds
    thres_fc: float = 0.3
    thres_dis: float = 0.005
    thres_pen: float = 0.001
