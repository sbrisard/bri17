import pybri17

if __name__ == "__main__":
    print(pybri17.__doc__)
    print(pybri17.__author__)
    print(pybri17.__version__)

    grid = pybri17.CartesianGrid2f64((3, 4), (1.1, 1.2))
    print(grid.dtype)
    print(grid.dim)
    print(grid.shape)
    print(grid.L)
    grid.L[1] = 0.0
