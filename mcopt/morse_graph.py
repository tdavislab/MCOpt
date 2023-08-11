from __future__ import annotations
from typing import Set, Dict, Optional, Union
from inspect import signature


import numpy as np
from numpy.typing import ArrayLike
import networkx as nx
import matplotlib
import matplotlib.colors
# import matplotlib.pyplot as plt

from mcopt.mm_space import (
	MetricProbabilityNetwork, 
	Coupling
)

Colors = Union[ArrayLike, Dict[int, float]]

__all__ = [
	'Colors',
	'MorseGraph'
]

if 'black' not in matplotlib.colormaps:
	matplotlib.colormaps.register(matplotlib.colors.ListedColormap(['black']), name='black')

class MorseGraph(nx.Graph):
	critical_nodes: Set[int]
	
	def __init__(self, critical_nodes : Set[int]):
		super().__init__()
		self.critical_nodes = critical_nodes

	def sample(self, rate: float|int, mode: str = 'step'):
		graph = MorseGraph(self.critical_nodes)
		
		visited = set()
		
		# Graphs can be quite large so recursion might lead to a stack overflow
		def dfs(start):
			if start in visited and start not in self.critical_nodes:
				return
			
			stack = []
			stack.append((start, start, 0))
			
			while len(stack) != 0:
				start, node, length = stack.pop()
				
				if node in visited and node not in self.critical_nodes:
					continue
				
				visited.add(node)
			
				for n in self.neighbors(node):
					if n in visited:
						continue
					
					if n in self.critical_nodes:
						graph.add_node(n, **self.nodes(data=True)[n])

						assert graph.has_node(start)
						graph.add_edge(start, n)
						
						continue
					
					if mode == 'step':
						new_length = length + 1
					elif mode == 'geo':
						new_length = length + np.linalg.norm(self.nodes(data='pos2')[n] - self.nodes(data='pos2')[node])
					else:
						raise ValueError(f'Unrecognized mode {mode}')
						
					if new_length > rate:
						graph.add_node(n, **self.nodes(data=True)[n])
						
						assert graph.has_node(start)
						graph.add_edge(start, n)
						
						stack.append((n, n, 0))
					else:
						stack.append((start, n, new_length))
					 
					
		for crit in self.critical_nodes:
			graph.add_node(crit, **self.nodes(data=True)[crit])
			dfs(crit)
			
		assert nx.is_connected(graph)
		assert all(graph.has_node(n) for n in self.critical_nodes)
		
		return graph

	def node_color_by_position(self) -> Dict[int, float]:
		return {n : np.linalg.norm(pos) for n, pos in self.nodes(data='pos2')}

	def node_color_by_coupling(
		self, 
		src_node_color: Dict[int, float],
		coupling: Coupling
	) -> Dict[int, float]:
		colors = {}
		
		c = np.asarray(coupling)
		
		for n in self.nodes:
			i = coupling.dest_rev_map[n]
			src_i = c[:, i].argmax()
			
			if (np.isclose(c[src_i, i], 0)):
				colors[n] = np.nan
			else:
				src = coupling.src_map[src_i]
				colors[n] = src_node_color[src]
		
		return colors
	
	def draw(
		self,
		ax: matplotlib.axes.Axes,
		cmap = 'black',
		rotation: float = 0,
		node_size: int = 40,
		node_color: Optional[Colors] = None,
		**kwargs
	):
		# Validate kwargs
		valid_node_kwargs = signature(nx.draw_networkx_nodes).parameters.keys()
		valid_edge_kwargs = signature(nx.draw_networkx_edges).parameters.keys()

		valid_kwargs = valid_node_kwargs | valid_edge_kwargs - {
			"G",
			"pos",
			"with_labels",
			"node_shape",
			"alpha"

			"node_size",
			"node_color",
			"cmap",
		}

		if any(k not in valid_kwargs for k in kwargs):
			invalid_args = ", ".join([k for k in kwargs if k not in valid_kwargs])
			raise ValueError(f"Received invalid argument(s): {invalid_args}")
		
		node_kwargs = {k: v for k, v in kwargs.items() if k in valid_node_kwargs}
		edge_kwargs = {k: v for k, v in kwargs.items() if k in valid_edge_kwargs}

		# Preprocess node_color
		if not node_color:
			node_color = self.node_color_by_position()

		# Determine unmapped nodes
		unmapped = {n for n in self.nodes() if np.isnan(node_color[n])}
		unmapped_node_color = np.array([node_color[n] for n in self.nodes() if n in unmapped])

		mapped = set(self.nodes()) - unmapped
		mapped_node_color = np.array([node_color[n] for n in self.nodes() if n in mapped])

		unmapped = [n for n in self.nodes() if n in unmapped]
		mapped = [n for n in self.nodes() if n in mapped]

		# Preprocess positions
		pos = dict(self.nodes(data = 'pos2'))

		if not np.isclose(rotation, 0):
			theta = np.radians(rotation)

			r = np.array([
				[np.cos(theta), -np.sin(theta)],
				[np.sin(theta), np.cos(theta)],
			])
			
			pos = {n : r.dot(p) for n, p in pos.items()}

		nx.draw_networkx_edges(
			self,
			ax = ax,
			pos=pos,
			**edge_kwargs
		)

		if len(unmapped):
			nx.draw_networkx_nodes(
				self,
				ax = ax,
				pos = pos,
				node_color = 'white',
				edgecolors='black',
				node_size=node_size,
				nodelist = list(unmapped),
				cmap = cmap,
				alpha=[1],
				**node_kwargs
			)

		if len(mapped):
			nx.draw_networkx_nodes(
				self,
				ax = ax,
				pos = pos,
				node_color = mapped_node_color,
				node_size=node_size,
				nodelist = list(mapped),
				cmap = cmap,
				alpha=[1],
				**node_kwargs
			)

		ax.set_axis_off()
		ax.set_aspect('equal', adjustable='box')

	def _draw(
		self,
		ax: matplotlib.axes.Axes,
		cmap='black',
		node_color: Optional[Colors] = None,
		node_size: int = 40,
		critical_scale: float = 1,
		rotation: float = 0,
		**kwargs,
	):
		if not node_color:
			node_color = self.node_color_by_position()
			
		if type(node_color) is dict:
			node_color = np.array([node_color[n] for n in self.nodes()])

		node_size = np.array([
			node_size * critical_scale if n in self.critical_nodes else node_size
			for n in self.nodes()
		])

		# Allow for nodes that should be given a "bad color" to have `nan` values.
		vmin = np.nanmin(node_color)
		vmax = np.nanmax(node_color)

		# Transform positions
		pos = {n : p for n, p in self.nodes(data='pos2')}

		if rotation != 0:
			theta = np.radians(rotation)
			
			r = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)],
			])
			
			pos = {n : r.dot(p) for n, p in pos.items()}

		nx.draw(
			self,
			ax=ax,
			cmap = cmap,
			pos = pos,
			node_color = node_color,
			node_size = node_size,
			
			vmin=vmin,
			vmax=vmax,
			alpha=[1],
			**kwargs
		)

		ax.set_aspect('equal', adjustable='box')
	
	def to_mpn(self, hist: str = 'uniform', dist: str = 'step') -> MetricProbabilityNetwork:
		X = np.array(self.nodes())
		X.sort()
		
		if hist == 'uniform':
			measure = np.ones(X.shape[0]) / X.shape[0]
		elif hist == 'degree':
			degs = np.array([self.degree(n) for n in X])
			
			measure = degs / degs.sum()
		else:
			raise ValueError(f'Unrecognized histogram type {hist}')
		
		metric = np.zeros(shape=(X.shape[0], X.shape[0]), dtype=float)
		
		if dist == 'step':
			lens = dict(nx.all_pairs_shortest_path_length(self))
			
			for u_i, u in enumerate(X):
				for v_i, v in enumerate(X):
					metric[u_i, v_i] = lens[u][v]
		elif dist == 'geo':
			lens = dict(nx.all_pairs_dijkstra_path_length(
				self,
				weight=lambda u, v, _: np.linalg.norm(self.nodes(data='pos2')[u] - self.nodes(data='pos2')[v])
			))
			
			for u_i, u in enumerate(X):
				for v_i, v in enumerate(X):
					metric[u_i, v_i] = lens[u][v]
		elif dist == 'adj':
			for u_i, u in enumerate(X):
				for v_i, v in enumerate(X):
					metric[u_i, v_i] = int(v in self.adj[u])
		else:
			raise ValueError(f'Unrecognized distance type {dist}')
		
		return MetricProbabilityNetwork(X, measure, metric)
	
	@staticmethod
	def attribute_cost_matrix(src: MorseGraph, dest: MorseGraph) -> np.ndarray:
		X = list(src.nodes())
		X.sort()
		
		Y = list(dest.nodes())
		Y.sort()
		
		X_attrs = list(src.nodes(data='pos2')[n] for n in X)
		Y_attrs = list(dest.nodes(data='pos2')[n] for n in Y)
		
		M = np.zeros((len(X), len(Y)), dtype=float)
		
		for u_i, u in enumerate(X):
			for v_i, v in enumerate(Y):
				M[u_i, v_i] = np.linalg.norm(X_attrs[u_i] - Y_attrs[v_i])
		
		return M
