"""Single lightweight registry for the five historical recognition tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneralTaskSpec:
    checkpoint: str
    classes: tuple[str, ...]
    window_size: int
    channels: int
    sample_dtype: str
    input_layout: str
    demo_subdir: str
    architecture: str

    @property
    def input_description(self) -> str:
        layouts = {
            "interleaved_iq": "I/Q 交错双通道",
            "channel_major": "I 通道后接 Q 通道",
            "single_channel": "单通道实信号",
        }
        return f"{self.sample_dtype}，{layouts[self.input_layout]}"


TASKS: dict[str, GeneralTaskSpec] = {
    "信号个体识别": GeneralTaskSpec(
        "models/light/cnnzl_12.pkl",
        (
            "RCT102-171-2289-2001-06-car-A", "RCT102-171-2289-2001-06-car-C",
            "RCT102-171-2289-2001-10-car-A", "RCT102-171-2289-2001-10-car-B",
            "RCT102-171-2289-2001-12-car-A", "RCT102-171-2289-2001-12-car-B",
            "RCT102-171-2289-2001-19-car-A", "RCT102-171-2289-2001-19-car-B",
            "RCT102-171-2289-2001-19-car-C", "RCT102-171-2289-2001-21-car-C",
            "RCT102-171-2289-2001-22-car-A", "RCT102-171-2289-2001-22-car-C",
            "RCT102-171-2289-2001-23-car-A", "RCT102-171-2289-2001-25-car-C",
            "RCT102-171-2289-2001-26-car-A",
        ),
        1024, 2, "float32", "interleaved_iq", "individual", "individual_light",
    ),
    "信号调制识别": GeneralTaskSpec(
        "models/ResNet18.pkl",
        (
            "16QAM", "32QAM", "64QAM", "128QAM", "BFSK", "4FSK", "CPFSK",
            "BPSK", "QPSK", "8PSK", "16PSK", "OQPSK", "FM", "AM", "PAM4",
        ),
        1024, 2, "float32", "channel_major", "modulation", "resnet18",
    ),
    "信号通联识别": GeneralTaskSpec(
        "models/12_92.307_bestModel.pth",
        ("1233", "1235", "1237", "1238", "1809"),
        5000, 2, "float32", "interleaved_iq", "tonglian", "communication_resnet18",
    ),
    "信号业务识别": GeneralTaskSpec(
        "models/ResNet18_3_all.pkl",
        ("数据", "话音"),
        2048, 1, "float32", "single_channel", "yewu", "resnet18",
    ),
    "信号编码识别": GeneralTaskSpec(
        "models/12_99.725_bestModel.pth",
        ("BCH", "Conv", "Hanming", "LDPC", "RS", "TCM"),
        512, 1, "int16", "single_channel", "bianma", "cnn",
    ),
}


def task_spec(task: str) -> GeneralTaskSpec:
    try:
        return TASKS[task]
    except KeyError:
        raise ValueError(f"不支持的历史识别任务：{task}") from None


__all__ = ["GeneralTaskSpec", "TASKS", "task_spec"]
