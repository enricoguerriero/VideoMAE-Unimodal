from src.models import VideoModel


def load_model(model_name: str, **kwargs) -> VideoModel:
    """Factory: build a VideoMAE backbone wrapper by name."""
    if model_name == "VideoMAE":
        from src.models import VideoMAE
        return VideoMAE(**kwargs)
    elif model_name == "VideoMAEGiant":
        from src.models import VideoMAEGiant
        return VideoMAEGiant(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not recognized. Use 'VideoMAE' or 'VideoMAEGiant'.")
