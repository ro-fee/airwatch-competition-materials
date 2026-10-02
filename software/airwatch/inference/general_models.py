"""Historical model loader; task contracts are lightweight and shared."""
import torch
import models
from airwatch.general_recognition_contract import GeneralTaskSpec, TASKS, task_spec
from airwatch.runtime_paths import resource_path


def load_general_model(task, device='cpu'):
    spec = task_spec(task)
    path = resource_path(spec.checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f'缺少历史识别权重：{path.name}')
    count = len(spec.classes)
    if spec.architecture == 'individual_light':
        net = models.CNNNEW_zl(count)
    elif spec.architecture == 'communication_resnet18':
        net = models.ResNet18_(count)
    elif spec.architecture == 'cnn':
        net = models.CNN(count)
    elif spec.architecture == 'resnet18':
        net = models.ResNet18(count, in_channel=spec.channels)
    else:
        raise ValueError(f'未实现的历史模型结构：{spec.architecture}')
    # Trusted shipped state dict only. Do not unpickle arbitrary user checkpoints.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if task in ('信号通联识别', '信号编码识别'):
        checkpoint = checkpoint['net']
    net.load_state_dict(checkpoint, strict=True)
    return net.to(device).eval(), path


__all__ = ["GeneralTaskSpec", "TASKS", "load_general_model", "task_spec"]
