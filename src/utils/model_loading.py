from src.models import VideoModel


def load_model(model_name: str, spec=None, **kwargs) -> VideoModel:
    """Factory: build a VideoMAE backbone wrapper by name.

    Args:
        model_name: "VideoMAE" | "VideoMAEGiant".
        spec (DataSpec|None): when given, `num_classes` and `task` are taken from
            it, so the head width and output activation follow configs/data.yaml.
            Explicit kwargs still win, which keeps the factory usable standalone.
        **kwargs: forwarded to the model constructor (backbone_id, device, ...).
    """
    if spec is not None:
        kwargs.setdefault("num_classes", spec.num_classes)
        kwargs.setdefault("task", spec.task)

    if model_name == "VideoMAE":
        from src.models import VideoMAE
        return VideoMAE(**kwargs)
    elif model_name == "VideoMAEGiant":
        from src.models import VideoMAEGiant
        return VideoMAEGiant(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not recognized. Use 'VideoMAE' or 'VideoMAEGiant'.")
