import math
import random
from typing import Hashable
from enum import Enum
from typing import List, Set
import numpy as np

from geometry.TwoDimension import SE2Pose, Point2


class AgentType(Enum):
    Robot = "Robot"
    Beacon = "Beacon"

class Agent(object):
    def __init__(self, name: Hashable, type: AgentType = AgentType.Robot):
        self._name = name
        self._type = type

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type
    def __str__(self) -> str:
        return " ".join([self._type.value,
                         str(self.name)])

    def __hash__(self) -> int:
        return hash(self._type.value+str(self._name))

class GridRobot(Agent):
    FeasibleRotRads = np.array([0, np.pi/2, np.pi, -np.pi/2, -np.pi])
    def __init__(self, name, step_scale:float = 1,
                 range_std: float = .2,
                 odom_cov: np.ndarray = np.diag([.1,.1,.02]),
                 rd_move_probs: np.ndarray = np.array([.5,.2,.2,.1]), noise_free_obs = False):
        super().__init__(name)
        self._step_scale = step_scale
        self._range_std = range_std
        self._odom_cov = odom_cov
        self._move_probs = rd_move_probs #forward, left, right, turn around
        self._noise_free_obs = noise_free_obs

    def get_range_measurement(self, gt_range: float):
        if self._noise_free_obs:
            return gt_range
        return np.random.normal(loc=gt_range, scale=self._range_std)

    def get_odom_measurement(self, gt_rel_pose: SE2Pose):
        if self._noise_free_obs:
            return gt_rel_pose
        lie_noise = np.random.multivariate_normal([0,0,0],self._odom_cov)
        return gt_rel_pose * SE2Pose.by_exp_map(lie_noise)

    def select_goals(self, cur_pose, goals):
        weights = np.zeros(len(goals))
        for i, goal in enumerate(goals):
            r, b = cur_pose.range_and_bearing(Point2(*goal))
            if abs(b)<1e-1:
                weights[i] = self._move_probs[0]
            elif abs(b + np.pi / 2) < 1e-1:
                weights[i] = self._move_probs[1]
            elif abs(b - np.pi / 2) < 1e-1:
                weights[i] = self._move_probs[2]
            elif abs(abs(b) - np.pi) < 1e-1:
                weights[i] = self._move_probs[3]
        return random.choices(goals,weights)[0]


    def local_path_planner(self, cur_pose: SE2Pose, goal: Point2, tol = 1e-4):
        r, b = cur_pose.range_and_bearing(goal)
        q, remainder = divmod(r, self._step_scale)
        steps = math.ceil(q)
        if steps > 0:
            first_move = SE2Pose(x=self._step_scale*np.cos(b),y=self._step_scale*np.sin(b),theta=b)
            rel_poses = [first_move]
            for i in range(1,steps):
                rel_poses.append(SE2Pose(x=self._step_scale))
            if remainder > tol:
                print("Caution: the last step moves by "+str(remainder))
                rel_poses.append(SE2Pose(x=remainder))
        else:
            first_move = SE2Pose(x=remainder*np.cos(b),y=remainder*np.sin(b),theta=b)
            rel_poses = [first_move]
        return rel_poses

class GridBeacon(Agent):
    def __init__(self, name):
        super().__init__(name, AgentType.Beacon)