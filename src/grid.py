
import numpy as np
import grid_updaters

DEFAULT_UPDATER = grid_updaters.NNCoulombFrictionUpdater(10, 0.5, 0.23)

class BaseGrid:
    def __init__(self, n, m=None, updater=DEFAULT_UPDATER, boundary_size=None):
        if m is None:
            m = n
        self.n = n
        self.m = m
        if boundary_size is None:
            boundary_size = int(np.ceil(0.1*n))
        self.boundary_size = boundary_size
        self._create_cache()
        self._initialize_grid()
        self._save_to_cache(self.grid)
        self.updater = updater
    
    def run_n_steps(self, n):
        counter = 0
        n_sites = []
        #TODO: implement the inside and clean boundary methods
        while counter < n+1:    #TODO change to for loop
            updated_grid = self.updater.update_step(self.grid)
            observables = self.updater.avalanche_details
            # n_site_avalanched = len(avalanche_details['released_values'])
            n_site_avalanched = observables['released_values'] > 0
            if n_site_avalanched==0:
                counter += 1
                if counter >= n:
                    break
            n_sites.append(n_site_avalanched)
            self._save_to_cache(updated_grid)
            self.grid = updated_grid
        return np.array(n_sites)
    
    def clear_cache(self):
        self._create_cache()
        
    def _create_cache(self):
        self.cache = np.empty((self.n, self.m, 0))
    
    def _initialize_grid(self):
        self.grid = np.random.rand(
            self.n+self.boundary_size,
            self.m+self.boundary_size
            )
        # self.grid = np.random.beta(1, 3, (self.n, self.m))

    def _inside(self, array: np.ndarray):
        return array[self.boundary_size:(-self.boundary_size), self.boundary_size:(-self.boundary_size)]

    def clean_boundary_inplace(self, array: np.ndarray, fill_value = False) -> np.ndarray:
        """
        Fill `array` at the boundary with `fill_value`.

        Useful to make sure sites on the borders do not become active and don't start toppling.

        Works inplace - will modify the existing array!

        :param array: array to be cleaned
        :param fill_value: value to fill boundaries with
        :rtype: np.ndarray
        """
        array[:self.boundary_size, :] = fill_value
        array[-self.boundary_size:, :] = fill_value
        array[:, :self.boundary_size] = fill_value
        array[:, -self.boundary_size:] = fill_value
        return array
        

    def _save_to_cache(self, grid):
        self.cache = np.concatenate((self.cache, grid[:,:,None]), axis=2)


if __name__ == '__main__':
    updater = grid_updaters.NNCoulombFrictionUpdater(f_s=3,
                                                    increment=0.01,
                                                    alpha=0,
                                                    site_updater=grid_updaters._nn_update_ij_on_finite_bc,
                                                    #  site_updater=grid_updaters._nn_update_ij_on_circular_bc,
                                                    )
    simple_grid = BaseGrid(200, updater=updater)
    steps_in_iteration = 100
    total_iterations = 10
    n_sites = np.array([])
    for _ in range(total_iterations):
        n_sites = np.append(n_sites, simple_grid.run_n_steps(steps_in_iteration))
        simple_grid.clear_cache()