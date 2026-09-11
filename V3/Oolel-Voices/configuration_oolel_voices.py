from transformers import PretrainedConfig


class OolelVoicesConfig(PretrainedConfig):
    r"""

    Args:
        exaggeration (`float`, *optional*, defaults to 0.5):
            Controls emotion expressiveness. Range [0, 1]. Higher = more expressive.
        cfg_weight (`float`, *optional*, defaults to 0.5):
            Classifier-free guidance weight. Higher = more faithful to text.
        temperature (`float`, *optional*, defaults to 0.8):
            Sampling temperature for speech token generation.
        repetition_penalty (`float`, *optional*, defaults to 1.2):
            Penalty for repeated tokens. Values > 1 discourage repetition.
        min_p (`float`, *optional*, defaults to 0.05):
            Minimum probability threshold for nucleus sampling.
        top_p (`float`, *optional*, defaults to 1.0):
            Top-p (nucleus) sampling cutoff.
        sample_rate (`int`, *optional*, defaults to 22050):
            Output audio sample rate in Hz.
    """

    model_type = "oolel_voices"

    def __init__(
        self,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        sample_rate: int = 22050,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
        self.top_p = top_p
        self.sample_rate = sample_rate
