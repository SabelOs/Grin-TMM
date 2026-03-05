import torch


def depolarization_factors(shape: str, device=None, dtype=None):
    """
    Returns depolarization factors Lx, Ly, Lz
    """
    if shape.lower() == "sphere":
        L = torch.tensor([1/3, 1/3, 1/3], device=device, dtype=dtype)

    elif shape.lower() == "chain":
        L = torch.tensor([0.133, 0.435, 0.435], device=device, dtype=dtype)

    elif shape.lower() == "double_sphere":
        L = torch.tensor([0.25, 0.375, 0.375], device=device, dtype=dtype)

    elif shape.lower() == "double_chain":
        L = torch.tensor([0.133, 0.342, 0.435], device=device, dtype=dtype)

    else:
        raise ValueError(f"Unknown inclusion shape: {shape}")

    return L


def bruggeman_eps_multi(
    eps_list,      # list of tensors (N_materials)
    f_list,        # list of tensors (same length)
    shape_list,    # list of shapes
    max_iter=50,
    tol=1e-6,
):
    """
    General multi-component Bruggeman EMT
    """

    device = eps_list[0].device
    dtype = eps_list[0].dtype

    # initial guess: weighted arithmetic average
    eps_eff = sum(f * eps for f, eps in zip(f_list, eps_list))

    L_list = [
        depolarization_factors(shape, device, dtype)
        for shape in shape_list
    ]

    for _ in range(max_iter):

        eps_old = eps_eff

        F_total = 0.0

        for eps_i, f_i, L_i in zip(eps_list, f_list, L_list):

            eps_i_3 = eps_i.unsqueeze(0)
            eps_eff_3 = eps_eff.unsqueeze(0)
            L_3 = L_i[:, None]

            term = (eps_i_3 - eps_eff_3) / (
                eps_eff_3 + L_3 * (eps_i_3 - eps_eff_3)
            )

            F_total += f_i * term.mean(dim=0)

        eps_eff = eps_eff + 0.5 * eps_eff * F_total

        if torch.max(torch.abs(eps_eff - eps_old)) < tol:
            break

    return eps_eff


def bruggeman_n_multi(
    n_list,      # list of n tensors
    f_list,      # list of fractions (must sum to 1)
    shape_list,  # list of shapes
):
    """
    Returns effective refractive index for arbitrary components
    """

    eps_list = [n**2 for n in n_list]

    eps_eff = bruggeman_eps_multi(
        eps_list=eps_list,
        f_list=f_list,
        shape_list=shape_list,
    )

    return torch.sqrt(eps_eff)