import torch


def depolarization_factors(shape: str, device=None, dtype=None):
    """
    Returns depolarization factors Lx, Ly, Lz
    """
    if shape.lower() == "sphere":
        L = torch.tensor([1/3, 1/3, 1/3], device=device, dtype=dtype)

    elif shape.lower() in ("chain"):
        # Long axis along x
        L = torch.tensor([0.133, 0.435, 0.435], device=device, dtype=dtype)

    elif shape.lower() in ("double_sphere"):
        # Flat in x-y plane
        L = torch.tensor([0.25, 0.375, 0.375], device=device, dtype=dtype)

    elif shape.lower() in ("double_chain"):
        # Flat in x-y plane
        L = torch.tensor([0.133, 0.342, 0.435], device=device, dtype=dtype)

    else:
        raise ValueError(f"Unknown inclusion shape: {shape}")

    return L


def bruggeman_eps(
    eps1: torch.Tensor,
    eps2: torch.Tensor,
    f1: torch.Tensor,
    shape1="sphere",
    shape2="sphere",
    max_iter=50,
    tol=1e-6
):
    device = eps1.device
    dtype = eps1.dtype

    f2 = 1.0 - f1

    L1 = depolarization_factors(shape1, device, dtype)  # (3,)
    L2 = depolarization_factors(shape2, device, dtype)  # (3,)

    # Initial guess
    eps_eff = f1 * eps1 + f2 * eps2

    for _ in range(max_iter):
        eps_old = eps_eff

        def term(eps_i, L):
            # expand to (3, n_wl)
            eps_i_3 = eps_i.unsqueeze(0)
            eps_eff_3 = eps_eff.unsqueeze(0)
            L_3 = L[:, None]

            return (eps_i_3 - eps_eff_3) / (
                eps_eff_3 + L_3 * (eps_i_3 - eps_eff_3)
            )

        F = (
            f1 * term(eps1, L1).mean(dim=0)
            + f2 * term(eps2, L2).mean(dim=0)
        )

        # Relaxed fixed-point update
        eps_eff = eps_eff + 0.5 * eps_eff * F

        if torch.max(torch.abs(eps_eff - eps_old)) < tol:
            break

    return eps_eff



def bruggeman_n(
    n1: torch.Tensor,
    n2: torch.Tensor,
    f1: torch.Tensor,
    shape1="sphere",
    shape2="sphere",
):
    """
    Returns effective refractive index
    """
    eps1 = n1 ** 2
    eps2 = n2 ** 2

    eps_eff = bruggeman_eps(
        eps1, eps2, f1,
        shape1=shape1,
        shape2=shape2,
    )

    return torch.sqrt(eps_eff)
