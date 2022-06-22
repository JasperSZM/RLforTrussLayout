import gym
import copy
import os
import numpy as np
import random

from collections import OrderedDict
from .space import StateObservationSpace, AutoregressiveEnvObservationSpace, AutoregressiveEnvActionSpace, ActionSpace
from .dynamic import DynamicModel
from .utils import is_edge_addable, readFile, getlen2, save_file, save_trajectory
from .state import State


class Truss(gym.Env):
    def __init__(self, num_points, initial_state_files,
                 coordinate_range, area_range, coordinate_delta_range, area_delta_range, fixed_points, variable_edges,
                 max_refine_steps, min_refine_steps=10,
                 edge_constraint=True, dis_constraint=True, stress_constraint=True, buckle_constraint=False, self_weight=False,
                 dimension=2, constraint_threshold=1e-7, best_n_results=100,
                 structure_fail_reward=-1., constraint_fail_reward=0., reward_lambda=169000000,
                 best=10000, save_good_threshold=100, normalize_magnitude=False):
        r'''
        Create a Truss Refine environment instance.
        :param num_points: number of nodes
        :param initial_state_files: locations where initial states are stored
        :param coordinate_range: nodes' coordinate range
        :param area_range: edges' area range
        :param coordinate_delta_range: nodes' modification range
        :param area_delta_range: edges' modification range
        :param max_refine_steps: max refine steps
        :param min_refine_steps: another refine steps threshold
        :param edge_constraint: intersection of edges
        :param dis_constraint: nodes' displacement weight
        :param stress_constraint: edges' stress
        :param buckle_constraint: buckle constraint
        :param self_weight: edge's weight
        :param dimension: dimension
        :param constraint_threshold: eps to check constraint
        :param best_n_results: choose best_n_results from initial_state_files
        :param structure_fail_reward: structure fail reward
        :param constraint_fail_reward: constraint fail reward
        :param reward_lambda: reward lambda
        :param best: initial best weight
        :param save_good_threshold: threshold over best weight to best
        :param normalize_magnitude: normalize observation's magnitude
        '''
        if dimension != 2:
            raise NotImplementedError("only support 2D dimension for now")

        # Env Config
        self.num_points = num_points
        self.dimension = dimension
        self.fixed_points = fixed_points
        self.state_observation_space = StateObservationSpace(num_points, coordinate_range, area_range)
        self.env_observation_space = AutoregressiveEnvObservationSpace(num_points, coordinate_range, area_range)
        self.action_space = AutoregressiveEnvActionSpace(coordinate_delta_range, area_delta_range)
        self.normalize_magnitude = normalize_magnitude

        # Initial State
        self.initial_state_files = initial_state_files
        self.best_n_results = best_n_results

        # Done
        self.refine_step = None
        self.max_refine_steps = max_refine_steps
        self.min_refine_steps = min_refine_steps

        # Dynamics
        self.use_edge_constraint = edge_constraint
        self.use_dis_constraint = dis_constraint
        self.use_stress_constraint = stress_constraint
        self.use_buckle_constraint = buckle_constraint
        self.use_self_weight = self_weight
        self.constraint_threshold = constraint_threshold
        self.dynamic_model = DynamicModel(dimension, use_self_weight=self_weight, use_dis_constraint=dis_constraint, use_buckle_constraint=buckle_constraint, use_stress_constraint=stress_constraint)

        # State
        self.initial_state_file = None
        self.initial_state_point = None
        self.initial_state_bar = None
        self.state = State(num_points, dimension)
        self.state_dynamics = None
        self.prev_mass = None
        self.action_id = None
        self.trajectory = None
        self.loads = None
        self.normalize_factor = None

        # Reward
        self.structure_fail_reward = structure_fail_reward
        self.constraint_fail_reward = constraint_fail_reward
        self.reward_lambda = reward_lambda

        # Result
        self.best = best
        self.save_good_threshold = save_good_threshold

    @property
    def observation_space(self):
        return self.env_observation_space

    def valid_truss(self):
        r'''
        check whether self.state is valid
        :return: a list of four bools, a tuple of dynamics
        '''
        ret = [True for _ in range(4)]

        if not self.state_observation_space.contains(self.state.obs(nonexistent_edge=self.state_observation_space.low[-1])):
            ret[0] = False  # Not in valid observation

        for i in range(self.num_points):
            for j in range(i):
                if (self.state.nodes[i] == self.state.nodes[j]).all():
                    ret[1] = False  # Duplicate nodes location

        points = copy.deepcopy(self.initial_state_point)
        for i in range(self.num_points):
            points[i].vec.x = self.state.nodes[i][0]
            points[i].vec.y = self.state.nodes[i][1]
            if self.dimension == 3:
                points[i].vec.z = self.state.nodes[i][2]
        edges = copy.deepcopy(self.initial_state_bar)
        edges_list = []
        for i in range(self.num_points):
            for j in range(i):
                if self.state.edges[i][j] > 0:
                    edges[(j, i)].area = self.state.edges[i][j]
                    edges[(j, i)].len = getlen2(points[j], points[i])
                    edges_list.append((j, i))

        if self.use_edge_constraint:
            for _ in edges_list:
                i, j = _
                left_edges = copy.deepcopy(edges)
                left_edges.pop((i, j))
                if not is_edge_addable(i, j, points, left_edges):
                    ret[2] = False  # Edges intersect

        is_struct, mass, dis_value, stress_value, buckle_value = self.dynamic_model.run(points, edges)
        ret[3] = is_struct and max(dis_value, stress_value, buckle_value) < self.constraint_threshold  # Dynamic constraints
        return ret, (is_struct, mass, dis_value, stress_value, buckle_value)

    def reset(self, file_name=None):
        _ = random.random()
        if _ > 0.5:
            best_n_results = self.best_n_results
        else:
            best_n_results = -1
        if file_name is None:
            _input_file_list = os.listdir(self.initial_state_files)
            input_file_list = []
            for s in _input_file_list:
                if s[-3:] == 'txt':
                    input_file_list.append(s)
            input_file_list.sort()
            if self.best_n_results != -1:
                if len(input_file_list) > self.best_n_results:
                    input_file_list = input_file_list[:self.best_n_results]
            file_name = self.initial_state_files + input_file_list[np.random.randint(len(input_file_list))]

        #print("file_name =", file_name)

        self.initial_state_file = file_name
        points, edges = readFile(file_name)
        self.initial_state_point = OrderedDict()
        self.initial_state_bar = OrderedDict()
        for i in range(self.num_points):
            self.initial_state_point[i] = points[i]
        for e in edges:
            if e.area < 0:
                continue
            u = e.u
            v = e.v
            if u > v:
                tmp = u
                u = v
                v = tmp
            self.initial_state_bar[(u, v)] = e

        self.state = State(self.num_points, self.dimension)
        for i in range(self.num_points):
            self.state.nodes[i][0] = points[i].vec.x
            self.state.nodes[i][1] = points[i].vec.y
            if self.dimension == 3:
                self.state.nodes[i][2] = points[i].vec.z
        for e in edges:
            i = e.u
            j = e.v
            self.state.edges[i][j] = e.area
            self.state.edges[j][i] = e.area

        _ = self.valid_truss()
        #print("valid_truss =", _)
        assert _[0][0] and _[0][1] and _[0][2] and _[0][3], "Initial state {} not valid".format(file_name)
        self.state_dynamics = _[1]
        self.prev_mass = _[1][1]
        self.refine_step = 0

        self.trajectory = [copy.deepcopy(self.state)]

        self.loads = []
        for i in range(self.num_points):
            self.loads.append(self.initial_state_point[i].loadY)

        self.normalize_factor = np.array([1. for _ in range(self.num_points * self.dimension)] +
                                         [100. for _ in range(self.num_points * (self.num_points - 1) // 2)] +
                                         [1. for _ in range(2)] +
                                         [1e-5 for _ in range(self.num_points)])

        return self._observation()

    def _observation(self):
        state_obs = self.state.obs()
        self.action_id = self._generate_action_id()
        ret = np.concatenate((state_obs, np.array(self.loads), self.action_id))
        if self.normalize_magnitude:
            print("ret:", ret)
            print("normalize_factor:", self.normalize_factor)
            ret = ret * self.normalize_factor
            print("new_ret:", ret)
        return ret

    def _generate_action_id(self):
        id = np.random.randint(self.num_points - self.fixed_points + len(self.initial_state_bar))
        if id < self.num_points - self.fixed_points:
            i = np.random.randint(self.num_points - self.fixed_points) + self.fixed_points
            j = -1
        else:
            i = -1
            u, v = list(self.initial_state_bar)[id - (self.num_points - self.fixed_points)]
            j = (u * ((self.num_points - 1) + (self.num_points - u)) // 2) + (v - u - 1)
        return np.array([i, j], dtype=np.float64)

    def _reward_fn(self):
        is_struct, mass, dis_value, stress_value, buckle_value = self.state_dynamics
        if not is_struct:
            return self.structure_fail_reward
        if max(dis_value, stress_value, buckle_value) > self.constraint_threshold:
            return self.constraint_fail_reward
        reward = self.reward_lambda / ((mass * (1 + dis_value + stress_value + buckle_value)) ** 2)
        return reward

    def _stop_fn(self, mass, prev_mass):
        return (mass > prev_mass and self.refine_step > self.min_refine_steps) or self.refine_step >= self.max_refine_steps

    def step(self, action):
        assert self.action_space.contains(action), "actions({}) not in action space({})".format(action, self.action_space)
        obs = self.state.obs(nonexistent_edge=-1)
        n_obs = copy.deepcopy(obs)
        if self.action_id[0] == -1:
            _i = int(self.action_id[1]) + self.num_points * self.dimension
            n_obs[_i] += action[-1]
            n_obs[_i] = max(min(n_obs[_i], self.state_observation_space.high[_i]), self.state_observation_space.low[_i])
        else:
            n_obs[int(self.action_id[0]) * self.dimension: int(self.action_id[0] + 1) * self.dimension] += action[:-1]
            for _i in range(int(self.action_id[0]) * self.dimension, int(self.action_id[0] + 1) * self.dimension):
                n_obs[_i] = max(min(n_obs[_i], self.state_observation_space.high[_i]), self.state_observation_space.low[_i])
        self.state.set(n_obs)
        self.trajectory.append(copy.deepcopy(self.state))

        info = {}
        info['illegal action'] = 0
        reward = 0.
        valid, self.state_dynamics = self.valid_truss()
        if not (valid[0] and valid[1] and valid[2]):
            info['illegal action'] = 1
            reward = -1.
        weight_reward = self._reward_fn()
        reward += weight_reward

        mass = self.state_dynamics[1]
        if not (valid[0] and valid[1] and valid[2] and valid[3]):
            mass = 10000
        done = self._stop_fn(mass, self.prev_mass)
        self.prev_mass = mass
        info['is struct'] = self.state_dynamics[0]
        info['mass'] = mass
        info['displacement'] = self.state_dynamics[2]
        info['stress'] = self.state_dynamics[3]
        info['buckle'] = self.state_dynamics[4]
        info['initial_state_file'] = self.initial_state_file
        self.refine_step += 1

        if (valid[0] and valid[1] and valid[2] and valid[3]) and mass < self.best + self.save_good_threshold:
            save_file(self.initial_state_point, self.state, mass, self.initial_state_files)
            if mass < self.best:
                # if mass < self.best - 1:
                #     save_trajectory(self.initial_state_point, self.trajectory, mass, self.initial_state_files)
                print("best:", mass)
                self.best = mass
        return self._observation(), reward, done, info
