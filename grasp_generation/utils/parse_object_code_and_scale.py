from typing import Tuple


def parse_object_code_and_scale(object_code_and_scale_str: str) -> Tuple[str, float]:
    keyword = "_0_"
    idx = object_code_and_scale_str.rfind(keyword)
    object_code = object_code_and_scale_str[:idx]

    idx_offset_for_scale = keyword.index("0")
    object_scale = float(
        object_code_and_scale_str[idx + idx_offset_for_scale :].replace("_", ".")
    )
    return object_code, object_scale


def object_code_and_scale_to_str(object_code: str, object_scale: float) -> str:
    object_code_and_scale_str = f"{object_code}_{object_scale:.4f}".replace(".", "_")
    return object_code_and_scale_str
