import numpy as np

import mlem


def translate(text: str):
    """
    Translate
    """
    return " ".join(np.random.choice(list("abcdefg")) for _ in text.split())


mlem.api.save(translate, "model", sample_data="Woof woof!")
