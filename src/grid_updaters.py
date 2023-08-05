import numpy as np

class GridUpdater:
    def __init__(self, frictions_params={}):
        for k, v, in frictions_params.items():
            self.__setattr__(k, v)

    def update_step(self, grid: np.ndarray) -> np.ndarray:
        return grid.copy()


class NNCoulombFrictionUpdater(GridUpdater):
    def __init__(self, f_s, increment, alpha, site_updater=None):
        self.f_s = f_s
        self.increment = increment
        self.alpha = alpha
        self.avalanche_details = {}
        if site_updater is None:
            self.site_updater = _nn_update_ij_on_finite_bc
        else:
            self.site_updater = site_updater
    
    def update_step(self, grid: np.ndarray) -> np.ndarray:
        """Updates the grid according to a coulomb friction law. 

        If force sceeds threshold, will drop to 0 and distribute among neighbors.
        Otherwise will increse the entire grid by the force increment.
         Will update self.avalanche_details with a field 'released_values' that
         enumerates the sites that exceeded the static friction threshold in
         each iteration.

        Args:
            grid (np.ndarray): the grid to update.

        Returns:
            np.ndarray: the updated grid.

        """        

        self.drive()
        self.topple()

        # assert (grid.sum() < (grid.size * self.f_s)), "Sum of stess on grid exceeds maximal capacity"
        # exceed_logical = grid>=self.f_s
        # if np.any(exceed_logical):
            # return_grid = grid.copy()
            # for i, j in zip(*np.where(exceed_logical)):
                # self.site_updater(i, j, return_grid, self.alpha)
            # self.avalanche_details = {
                # 'released_values': grid[exceed_logical]
            # }
            # return return_grid
        # else:
            # self.avalanche_details = {'released_values': []}
            # return grid + self.increment
        
        
        # assert (grid.sum() < (grid.size * self.f_s)), "Sum of stess on grid exceeds maximal capacity"
        # exceed_logical = grid>=self.f_s
        # if np.any(exceed_logical):
        #     i, j = np.unravel_index(np.argmax(grid), grid.shape)
        #     return_grid = grid.copy()
        #     return_grid = self.site_updater(i, j, return_grid, self.alpha)
        #     self.avalanche_details = {
        #         'released_values': grid[i, j],
        #         'released_coords': (i, j),
        #     }
        #     return return_grid
        # else:
        #     self.avalanche_details = {
        #         'released_values': 0,
        #         'released_coords': None,
        #         }
        #     return grid + self.increment

    def drive(self, grid):
        return grid + self.increment

    # def topple(
    #         values: np.ndarray,
    #         visited: np.ndarray,
    #         releases: np.ndarray,
    #         critical_value_current: float,
    #         critical_value: float,
    #         conservation_lvl: float,
    #         boundary_size: int
    #         ) -> int:

    # # find a boolean array of active (overloaded) sites

    # active_sites = common.clean_boundary_inplace(
    #     values >= critical_value_current, boundary_size)
    # number_of_iterations = 0

    # while active_sites.any():
        
    #     releases += active_sites
    #     indices = np.vstack(np.where(active_sites)).T
    #       # a Nx2 array of integer indices for overloaded sites
    #     N = indices.shape[0]
    #     for i in range(N):
    #         x, y = index = indices[i]

    #         neighbors = index + np.array([[0, 1], [-1, 0], [1, 0], [0,-1]])
    #         for j in range(len(neighbors)):
    #             xn, yn = neighbors[j]
    #             values[xn, yn] += conservation_lvl * (values[x, y] - critical_value_current + critical_value)   # Grassberger (1994), eqns (1)
    #             visited[xn, yn] = True

    #         values[x, y] = critical_value_current - critical_value  # Grassberger (1994), eqns (1)
    #         active_sites = common.clean_boundary_inplace(values >= critical_value_current, boundary_size)
    #     number_of_iterations += 1

    # return number_of_iterations



def _nn_update_ij_on_circular_bc(i, j, return_grid, alpha):
    n, m = return_grid.shape
    current_val = return_grid[i, j]
    return_grid[i, j] -= current_val
    return_grid[i-1, j] += alpha*current_val
    return_grid[i, j-1] += alpha*current_val
    return_grid[((i+1)%n), j] += alpha*current_val
    return_grid[i, ((j+1)%m)] += alpha*current_val
    return return_grid


def _nn_update_ij_on_finite_bc(i, j, return_grid, alpha):
    n, m = return_grid.shape
    current_val = return_grid[i, j]
    return_grid[i, j] -= current_val
    if not (i-1<0):
        return_grid[i-1, j] += alpha*current_val
    if not (j-1<0):
        return_grid[i, j-1] += alpha*current_val
    if not (i+1>=n):
        return_grid[i+1, j] += alpha*current_val
    if not (j+1>=m):
        return_grid[i, j+1] += alpha*current_val
    return return_grid


if __name__ == '__main__':
    pass