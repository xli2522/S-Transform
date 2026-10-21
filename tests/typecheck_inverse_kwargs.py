"""Static typing smoke test for ``STransform.inverse_kwargs``."""

from s_transform import s_transform
from s_transform.core import InverseKwargs


def check_inverse_kwargs_type() -> None:
    transform = s_transform([0.0, 1.0], sample_rate=2.0)
    inverse_kwargs: InverseKwargs = transform.inverse_kwargs
    assert inverse_kwargs["original_length"] == 2
