import numpy as np

import pybri17

if __name__ == "__main__":
    dim = 2
    sym = (dim * (dim + 1)) // 2
    shape = dim * (256,)
    L = dim * (1.0,)
    h = tuple(L_i / n_i for L_i, n_i in zip(L, shape))
    patch_ratio = dim * (0.125,)
    grid = pybri17.CartesianGrid2f64(shape, L)
    μ = 1.0
    ν = 0.3
    hooke = pybri17.Hooke2f64(μ, ν, grid)
    τ_in = np.zeros((sym,), dtype=grid.dtype)
    τ_in[-1] = 1.0
    τ_out = np.zeros_like(τ_in)
    patch_size = tuple(int(r_ * n_) for r_, n_ in zip(patch_ratio, shape))

    τ = np.empty(shape + (sym,), dtype=np.complex128)
    τ[...] = τ_out
    τ[tuple(slice(n_) for n_ in patch_size) + (...,)] = τ_in
    τ_hat = np.fft.fftn(τ, axes=range(dim))
    print(τ.strides)
    print(τ_hat.strides)

    η_hat = np.zeros_like(τ)

    sqrt_2 = np.sqrt(2)
    inv_sqrt_2 = 1.0 / sqrt_2

    k = np.zeros((dim,), dtype=np.intc)
    for k[0] in range(shape[0]):
        print(k[0])
        for k[1] in range(shape[1]):
            τ_hat_k = τ_hat[k[0], k[1]]
            η_hat_k = η_hat[k[0], k[1]]
            hooke.modal_eigenstress_to_opposite_strain(k, τ_hat_k, η_hat_k)
            print(η_hat_k)
