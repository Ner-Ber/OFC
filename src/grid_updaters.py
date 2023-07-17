import numpy as np

class GridUpdater:
    def __init__(self, frictions_params={}):
        for k, v, in frictions_params.items():
            self.__setattr__(k, v)

    def update_grid(self, grid: np.ndarray) -> np.ndarray:
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
    
    def update_grid(self, grid: np.ndarray) -> np.ndarray:
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
        assert (grid.sum() < (grid.size * self.f_s)), "Sum of stess on grid exceeds maximal capacity"
        exceed_logical = grid>=self.f_s
        if np.any(exceed_logical):
            return_grid = grid.copy()
            for i, j in zip(*np.where(exceed_logical)):
                self.site_updater(i, j, return_grid, self.alpha)
            self.avalanche_details = {
                'released_values': grid[exceed_logical]
            }
            return return_grid
        else:
            self.avalanche_details = {'released_values': []}
            return grid + self.increment
    

def _nn_update_ij_on_circular_bc(i, j, return_grid, alpha):
    n, m = return_grid.shape
    current_val = return_grid[i, j]
    return_grid[i, j] -= current_val
    return_grid[i-1, j] += alpha*current_val
    return_grid[i, j-1] += alpha*current_val
    return_grid[((i+1)%n), j] += alpha*current_val
    return_grid[i, ((j+1)%m)] += alpha*current_val


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


if __name__ == '__main__':
    pass