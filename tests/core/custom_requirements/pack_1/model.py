from pack_2 import name


class TestM:

    name = name

    def _init_(self, alpha: float, max_lag: int):
        self.alpha = alpha  # pylint: disable=attribute-defined-outside-init
        self.max_lag = (  # pylint: disable=attribute-defined-outside-init
            max_lag
        )
