import torch


def compute_carry_chain(a_preds, b_preds):
    """Compute carry-chain consistent output digits from input-digit argmaxes.

    Args:
        a_preds: (B, N) int tensor, first operand digits MSB-first.
        b_preds: (B, N) int tensor, second operand digits MSB-first.

    Returns:
        s_targets: (B, N+1) int tensor, output digits MSB-first.
                   s_targets[:, 0] is the carry-out.
    """
    B, N = a_preds.shape
    s = torch.zeros(B, N + 1, dtype=torch.long, device=a_preds.device)
    carry = torch.zeros(B, dtype=torch.long, device=a_preds.device)

    for j in range(N):
        i = N - 1 - j
        raw = a_preds[:, i] + b_preds[:, i] + carry
        s[:, i + 1] = raw % 10
        carry = raw // 10
    s[:, 0] = carry
    return s

