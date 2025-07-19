import dataclasses


@dataclasses.dataclass
class ChatCompletionConfig:
    seed: int
    delay: int
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    no_repeat_ngram_size: int
    do_sample: bool
    system_prompt: str
    user_prompt: str