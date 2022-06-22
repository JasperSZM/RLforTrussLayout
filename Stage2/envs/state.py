import numpy as np


class State:
    def __init__(self, num_points, dimension):
        self.num_points = num_points
        self.dimension = dimension
        self.nodes = -np.ones((num_points, dimension), dtype=np.float64)
        self.edges = -np.ones((num_points, num_points), dtype=np.float64)

    def obs(self, nonexistent_edge=0.):
        r'''
        transfer self.state to observation
        :param nonexistent_edge: area value of nonexistent edge
        :return: list, a tuple containing nodes coordinates and edges area
        '''
        obs = np.zeros(self.num_points * self.dimension + self.num_points * (self.num_points - 1) // 2, dtype=np.float64)
        for i in range(self.num_points):
            obs[i * self.dimension: (i + 1) * self.dimension] = self.nodes[i]
        loc = self.num_points * self.dimension
        for i in range(self.num_points):
            for j in range(i + 1, self.num_points):
                obs[loc] = nonexistent_edge if self.edges[i][j] < 0 else self.edges[i][j]
                loc += 1
        return obs

    def set(self, obs):
        r'''
        set self.state according to obs, do not set nonexistent edges. Warning: obs must set 1 for nonexistent edges
        :param obs: observation
        :return: None
        '''
        for i in range(self.num_points):
            self.nodes[i] = obs[i * self.dimension: (i + 1) * self.dimension]
        loc = self.num_points * self.dimension
        for i in range(self.num_points):
            for j in range(i + 1, self.num_points):
                self.edges[i][j] = obs[loc]
                self.edges[j][i] = obs[loc]
                loc += 1
