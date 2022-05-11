from importlib import import_module

emoji = import_module("emoji")
np = import_module("numpy")


def translate(text: str):
    """
    Translate dog barks to emoji, as you hear them
    """
    return " ".join(
        np.random.choice(list(emoji.EMOJI_DATA.keys())) for _ in text.split()  # type: ignore
    )


def main():
    import sys

    import mlem

    mlem.api.save(translate, sys.argv[1], sample_data="Woof woof!")


if __name__ == "__main__":
    main()
