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
    print(grid.get_cell_nodes(1))
    print(grid.get_node_at(2, 3))

    hooke = pybri17.Hooke2f64(1., 0.3, grid);
    print(hooke)
