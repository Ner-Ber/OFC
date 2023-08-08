
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import grid_updaters
import tqdm
from IPython.display import HTML, display

DEFAULT_UPDATER = grid_updaters.NNCoulombFrictionUpdater(10, 0.5, 0.23)

class BaseGrid:
	# TODO: add docstring
	def __init__(self, n, m=None, boundary_size=None, save_every=3):
		if m is None:
			m = n
		self.n = n
		self.m = m
		if boundary_size is None:
			boundary_size = int(np.ceil(0.1*n))
		self.boundary_size = boundary_size
		self.save_every = save_every
		self._initialize_grid()
		self.observables = []
		self._create_cache()

	def run_n_steps(self, n, progress=False):
		for i in tqdm.tqdm(range(n), disable=progress):
			# print(f'i={i}')
			# time.sleep(0.01)
			observables = self.update_step()
			self.observables.append(observables)
			if (i%self.save_every)==0:
				self._save_to_cache()

	def clear_cache(self):
		self.cache = np.empty_like(self.grid[:, :, 0])
		# np.empty((self.n, self.m, 0))
		# self._create_cache()

	def update_step(self) -> dict:
		"""Abstract method for updating step."""
		return {}

	def _create_cache(self):
		self.cache = self.grid[:, :, None].copy()
		# self.cache = np.empty((self.n, self.m, 0))

	def _initialize_grid(self):
		self.grid = np.random.rand(
			self.n + 2*self.boundary_size,
			self.m + 2*self.boundary_size
			)
		self._define_inside_logical()

	def _define_inside_logical(self):
		x_idxs, y_idxs = np.meshgrid(
			range(self.n + 2*self.boundary_size),
			range(self.m + 2*self.boundary_size),
		)
		self.inside_logical = (
			(x_idxs >= self.boundary_size) &
			(x_idxs < (self.n + self.boundary_size))	&
			(y_idxs >= self.boundary_size) &
			(y_idxs < (self.m + self.boundary_size))
		)

	def _inside(self, array: np.ndarray):
		return array[
			self.boundary_size:(-self.boundary_size), self.boundary_size:(-self.boundary_size), ...
			]

	def _clean_boundary_inplace(
			self,
			array: np.ndarray,
			fill_value = False) -> np.ndarray:
		array[~self.inside_logical] = fill_value
		# array[:self.boundary_size, :] = fill_value
		# array[-self.boundary_size:, :] = fill_value
		# array[:, :self.boundary_size] = fill_value
		# array[:, -self.boundary_size:] = fill_value
		return array

	def _save_to_cache(self):
		self.cache = np.concatenate((self.cache, self.grid[:,:,None]), axis=2)

	def observables_df(self):
		return pd.DataFrame(self.observables)

	def animate_states(
			self,
			notebook: bool = False,
			with_boundaries: bool = False,
			interval: int = 1,
			):
		"""Present animation from cache.

		Args:
				notebook (bool, optional): _description_. Defaults to False.
				with_boundaries (bool, optional): _description_. Defaults to False.
				interval (int, optional): _description_. Defaults to 30.

		Returns:
				_type_: _description_
		"""
		fig, ax = plt.subplots()
		if with_boundaries:
			values = self.cache
		else:
			values = self._inside(self.cache)

		im = ax.imshow(
			values[:, :, 0],
			interpolation='nearest',
			vmin = values.min(),
			vmax = values.max(),
			)
		plt.colorbar(im)
		iterations = values.shape[2]
		title = ax.set_title(f'Iteration {0}/{iterations*self.save_every}')

		def animate(i):
			im.set_data(values[:,:,i])
			title.set_text(
				f'Iteration {i * self.save_every}/{iterations * self.save_every}'
				)
			return im, title

		anim = animation.FuncAnimation(
			fig,
			animate,
			frames=iterations,
			interval=interval,
			)
		if notebook:
			plt.close(anim._fig)
			display(HTML(anim.to_jshtml()))
		else:
			return anim



class NNCoulombFrictionGrid(BaseGrid):
	"""A grid to simulate simple OFC (Coulomb friction and nearest neighbors
		interactions)"""

	def __init__(self, f_s, increment, alpha, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.f_s = f_s
		self.increment = increment
		self.alpha = alpha
		self._adjust_init_grid()


	def update_step(self):
		self.drive()
		number_of_iterations, avalanche_size, number_of_releases = self.topple()
		observables = {
			'number_of_iterations': number_of_iterations,
			'avalanche_size': avalanche_size,
			'number_of_releases': number_of_releases,
		}
		return observables

	def drive(self):
		self.grid += self.increment

	def topple(self) -> int:
		visited = np.full_like(self.grid, False)
		releases = np.full_like(self.grid, 0)

		active_sites = self._clean_boundary_inplace(
			self.grid >= self.f_s, False
			)
		_ = self._clean_boundary_inplace(self.grid, 0)
		number_of_iterations = 0

		while active_sites.any():
			releases += active_sites
			indices = np.vstack(np.where(active_sites)).T
			# a nx2 array of integer indices for overloaded sites
			n_active = indices.shape[0]
			for i in range(n_active):
				x, y = index = indices[i]

				neighbors = index + np.array([[0, 1], [-1, 0], [1, 0], [0,-1]])
				for j in range(len(neighbors)):
					xn, yn = neighbors[j]
					# self.grid[xn, yn] += self.alpha * (self.grid[x, y] - self.f_s)
					self.grid[xn, yn] += self.alpha * self.grid[x, y]
					visited[xn, yn] = True
				# self.grid[x, y] = self.f_s
				self.grid[x, y] = 0
				active_sites = self._clean_boundary_inplace(
					self.grid >= self.f_s, False
					)
				_ = self._clean_boundary_inplace(self.grid, 0)
			number_of_iterations += 1

		avalanche_size = self._inside(visited).sum()
		number_of_releases = self._inside(releases).sum()
		return number_of_iterations, avalanche_size, number_of_releases

	def _adjust_init_grid(self):
		self.grid *= 0.9*self.f_s


if __name__ == '__main__':

	try_grid = NNCoulombFrictionGrid(
		10,
		0.3,
		0.23,
		n=5,
		m=5,
		boundary_size=2,
		save_every=3,
	)
	try_grid.run_n_steps(35)

	pass
