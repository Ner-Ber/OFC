import numpy as np
import grid_updaters

DEFAULT_UPDATER = grid_updaters.NNCoulombFrictionUpdater(10, 0.5, 0.23)

class BaseGrid:
    def __init__(self, n, m=None, updater=DEFAULT_UPDATER):
        if m is None:
            m = n
        self.n = n
        self.m = m
        self._create_cache()
        self._initialize_grid()
        self._save_to_cache(self.grid)
        self.updater = updater
    
    def run_n_steps(self, n):
        counter = 0
        n_sites = []
        while counter < n+1:
            updated_grid = self.updater.update_grid(self.grid)
            avalanche_details = self.updater.avalanche_details
            n_site_avalanched = len(avalanche_details['released_values'])
            print(n_site_avalanched)
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
        self.grid = np.random.rand(self.n, self.m)

    def _save_to_cache(self, grid):
        self.cache = np.concatenate((self.cache, grid[:,:,None]), axis=2)


if __name__ == '__main__':
    pass