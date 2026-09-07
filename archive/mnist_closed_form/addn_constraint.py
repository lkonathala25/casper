"""Place-value addition target used by the archived on-the-fly layers."""


def _make_addn_constraint(n_digits):
    """Build a place-value addition constraint function for two N-digit numbers."""
    def _addn_constraint(digit_preds):
        n = n_digits
        first, second = digit_preds[:n], digit_preds[n:]
        first_num = sum(d * (10 ** (n - 1 - i)) for i, d in enumerate(first))
        second_num = sum(d * (10 ** (n - 1 - i)) for i, d in enumerate(second))
        return first_num + second_num
    return _addn_constraint


