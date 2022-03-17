from copy import deepcopy

import numpy as np
from typing import Tuple, List, Union, Dict

from manhattan_world_with_range.Agent import GridRobot, GridBeacon
from geometry.TwoDimension import SE2Pose, Point2

def find_nearest(array, value):
    array = np.asarray(array)
    distances = np.abs(array - value)
    idx = distances.argmin()
    delta = value - array[idx]
    return idx, delta, array[idx]

class ManhattanWaterworld:
    """
    This class creates a simulated environment of Manhattan world with landmarks.

    ---
    :param
    grid_vertices_shape: a tuple defining the shape of grid vertices; note that the vertices follow ij indexing
    cell_scale: width and length of a cell
    robot_area: top left and bottom right vertices of a rectangular area; all the rest area will be infeasible
    landmark_area: top left and bottom right vertices of a rectangular area; all the rest area will be feasible
    """
    def __init__(self, grid_vertices_shape: tuple = (9,9),
                 cell_scale: int = 1,
                 robot_area: List[Tuple] = None,
                 landmark_area: List[Tuple] = None,
                 check_collision:bool = True,
                 rbt2pose: Dict[GridRobot, SE2Pose] = None,
                 lmk2point: Dict[GridBeacon, Point2] = None,
                 tol: float = 1e-5
                 ):
        self._x_pts, self._y_pts = grid_vertices_shape
        self._scale = cell_scale
        self._tol = tol
        self._check_collision = check_collision

        self._x_coords = np.arange(self._x_pts) * self._scale
        self._y_coords = np.arange(self._y_pts) * self._scale
        self._xv, self._yv = np.meshgrid(self._x_coords, self._y_coords, indexing='ij')

        # agents are added by vertices but stored with groundtruth poses or points
        if rbt2pose is not None:
            self._rbt2pose = rbt2pose
        else:
            self._rbt2pose = {}

        if rbt2pose is not None:
            self._lmk2point = lmk2point
        else:
            self._lmk2point = {}

        if robot_area is not None:
            # ensure a rectangular feasible area for robot
            bl, tr = robot_area
            self._robot_feasibility = np.zeros((self._x_pts, self._y_pts), dtype=bool)
            self._robot_feasibility[bl[0]:tr[0]+1,bl[1]:tr[1]+1] = True
            self._landmark_feasibility = np.ones((self._x_pts, self._y_pts), dtype=bool)
            self._landmark_feasibility[bl[0]:tr[0]+1,bl[1]:tr[1]+1] = False
        elif landmark_area is not None:
            # ensure a rectangular feasible area for landmarks
            bl, tr = landmark_area
            self._landmark_feasibility = np.zeros((self._x_pts, self._y_pts), dtype=bool)
            self._landmark_feasibility[bl[0]:tr[0]+1, bl[1]:tr[1]+1] = True
            self._robot_feasibility = np.ones((self._x_pts, self._y_pts), dtype=bool)
            self._robot_feasibility[bl[0]:tr[0]+1, bl[1]:tr[1]+1] = False
        else:
            self._landmark_feasibility = np.zeros((self._x_pts, self._y_pts), dtype=bool)
            self._robot_feasibility = np.ones((self._x_pts, self._y_pts), dtype=bool)

    def set_robot_area_feasibility(self,area: List[tuple], feasibility: Union[bool, int]):
        mask = np.zeros((self._x_pts, self._y_pts), dtype=bool)
        bl, tr = area
        mask[bl[0]:tr[0]+1, bl[1]:tr[1]+1] = True
        self._robot_feasibility[mask] = feasibility
        self._robot_feasibility[np.invert(mask)] = not feasibility

    def set_landmark_area_feasibility(self,area: List[tuple], feasibility: Union[bool, int]):
        mask = np.zeros((self._x_pts, self._y_pts), dtype=bool)
        bl, tr = area
        mask[bl[0]:tr[0]+1, bl[1]:tr[1]+1] = True
        self._landmark_feasibility[mask] = feasibility
        self._landmark_feasibility[np.invert(mask)] = not feasibility

    def get_neighboring_vertices(self, i: int, j: int)->List[tuple]:
        candidate_vertices = [(i+1,j), (i,j+1),  (i-1,j),  (i,j-1)]
        vertices_in_bound = []
        for vertex in candidate_vertices:
            if 0 <= vertex[0] < self._x_pts and 0 <= vertex[1] < self._y_pts:
                vertices_in_bound.append(vertex)
        return vertices_in_bound

    def get_neighboring_robot_vertices(self, i: int, j: int, feasibility = None)->List[tuple]:
        nb_pts = self.get_neighboring_vertices(i, j)
        return self.pick_robot_vertices(nb_pts, feasibility)

    def pick_robot_vertices(self, vertices, feasibility:np.ndarray = None)->List[tuple]:
        feasible_pts = []
        if feasibility is None:
            feasibility = self._robot_feasibility
        for vertex in vertices:
            if feasibility[vertex[0], vertex[1]]:
                feasible_pts.append(vertex)
        return feasible_pts

    def nearest_robot_vertex_coordinates(self, x: float, y: float, feasibility: np.ndarray = None)->List[tuple]:
        i, dx, x_close = find_nearest(self._x_coords, x)
        j, dy, y_close = find_nearest(self._y_coords, y)
        if abs(dx) <self._tol and abs(dy) <self._tol:
            goal_vertices = self.get_neighboring_vertices(i, j)
        elif abs(dy) <self._tol:
            if dx > 0:
                goal_vertices = [(i + 1, j), (i, j)]
            else:
                goal_vertices = [(i, j), (i - 1, j)]
        elif abs(dx) <self._tol:
            if dy > 0:
                goal_vertices = [(i, j + 1), (i, j)]
            else:
                goal_vertices = [(i, j), (i, j - 1)]
        else:
            raise ValueError("The robot with location ("+str(x),', '+str(y)+') falls off the grid.')
        goal_vertices = self.pick_robot_vertices(goal_vertices, feasibility)
        return [self.vertex2coordinate(*vertex) for vertex in goal_vertices]

    def coordinate2vertex(self,x, y)->tuple:
        i, dx, x_close = find_nearest(self._x_coords, x)
        j, dy, y_close = find_nearest(self._y_coords, y)
        if abs(dx) <self._tol and abs(dy) <self._tol:
            return (i, j)
        else:
            raise ValueError("The input ("+str(x)+", "+str(y)+") is off grid vertices.")

    def coordinates2vertices(self, coords: List[tuple])->List[tuple]:
        return [self.coordinate2vertex(*c) for c in coords]

    def agent_xy(self, agent2gt):
        return np.array([[pt.x, pt.y] for key, pt in agent2gt.items()])

    def no_collision(self, x, y, agent2gt):
        if self._check_collision:
            gt_xy = self.agent_xy(agent2gt)
            if gt_xy.shape[0] > 0:
                xy = np.array([x,y])
                min_dist = min(np.linalg.norm(gt_xy - xy, axis=1))
                if min_dist > self._tol:
                    return True
                else:
                    print('Collision: minimum distance to existing agents is '+str(min_dist))
                    return False
            else:
                return True
        else:
            return True

    def add_landmark(self, lmk: GridBeacon, i: int, j: int):
        if self._landmark_feasibility[i, j] and lmk not in self._lmk2point:
            x, y = self.vertex2coordinate(i, j)
            if not self._lmk2point or self.no_collision(x, y, self._lmk2point):
                self._lmk2point[lmk] = Point2(x,y)
                return True
            else:
                print('Add abort: landmark collision found.')
                return False
        elif lmk in self._lmk2point:
            print('Add abort: duplicated landmark.')
            return False
        else:
            print('Add abort: vertex ('+str(i)+', '+str(j)+') is infeasible for adding landmarks.')
            return False

    def add_robot(self, rbt: GridRobot, i = int, j = int, orientation = 0):
        if self._robot_feasibility[i, j] and rbt not in self._rbt2pose:
            x, y = self.vertex2coordinate(i, j)
            if not self._rbt2pose or self.no_collision(x, y, self._rbt2pose):
                self._rbt2pose[rbt] = SE2Pose(x,y, orientation)
                return True
            else:
                print('Add abort: robot collision found.')
                return False
        elif rbt in self._rbt2pose:
            print('Add abort: duplicated robot.')
            return False
        else:
            print('Add abort: vertex (' + str(i) + ', ' + str(j) + ') is infeasible for adding robots.')
            return False

    def remove_robot(self, agent):
        del self._rbt2pose[agent]
        return True

    def remove_landmark(self, agent):
        del self._lmk2point[agent]
        return True

    def reset_robot(self, agent, i, j, orientation=0):
        return self.remove_robot(agent) and self.add_robot(agent, i, j,orientation)

    def reset_landmark(self, agent, i, j):
        return self.remove_landmark(agent) and self.add_landmark(agent, i, j)

    def vertex2coordinate(self, i: int, j: int)->tuple:
        return (self._xv[i, j], self._yv[i, j])

    def vertices2coordinates(self, vs)->List[tuple]:
        return [self.vertex2coordinate(*v) for v in vs]

    def is_xy_on_robot_grid(self, x, y):
        # if x and y is on the grid and within the robot area
        # its nearest points in the area should be more than two.
        nearest_xy = self.nearest_robot_vertex_coordinates(x, y)
        if len(nearest_xy) >= 2:
            return True
        else:
            return False

    def update_robot_pose(self, agent, pose: SE2Pose):
        assert agent in self._rbt2pose
        x, y = pose.x, pose.y
        on_grid_in_bound = self.is_xy_on_robot_grid(x, y)
        no_collision = self.no_collision(x, y, self._rbt2pose)
        if on_grid_in_bound and no_collision:
            self._rbt2pose[agent] = pose
            return True
        if not no_collision:
            print("Update abort: found collision.")
        if not on_grid_in_bound:
            print("Update abort: pose off grid.")
        return False

    def is_robot_vertex(self, i, j):
        return self._robot_feasibility[i, j]

    def is_landmark_vertex(self, i, j):
        return self._landmark_feasibility[i, j]

    @property
    def vertices(self)->np.ndarray:
        mesh = np.array(np.meshgrid(np.arange(self._x_pts), np.arange(self._y_pts), indexing='ij'))
        combinations = mesh.T.reshape(-1, 2)
        return combinations

    @property
    def robot_feasible_vertices(self)->np.ndarray:
        res = []
        for pt in self.vertices:
            if self.is_robot_vertex(*pt):
                res.append(pt)
        return np.array(res)

    @property
    def landmark_feasible_vertices(self)->np.ndarray:
        res = []
        for pt in self.vertices:
            if self.is_landmark_vertex(*pt):
                res.append(pt)
        return np.array(res)

    @property
    def meshgrid(self)->tuple:
        return self._xv, self._yv

    @property
    def robot_feasibility(self)->np.ndarray:
        return self._robot_feasibility

    @property
    def landmark_feasibility(self)->np.ndarray:
        return self._landmark_feasibility

    @property
    def shape(self)->tuple:
        return (self._x_pts, self._y_pts)

    @property
    def scale(self)->float:
        return self._scale

    @property
    def robots(self)->List:
        return [agent for agent in self._rbt2pose]

    @property
    def landmarks(self)->List:
        return [agent for agent in self._lmk2point]

    def __str__(self):
        line = ''
        line += 'Shape: '+self.shape.__repr__() +'\n'
        line += 'Cell scale: '+self.scale.__repr__() +'\n'
        line += 'Robot feasible vertices: '+self._robot_feasibility.__repr__() +'\n'
        line += 'Landmark feasible vertices: '+self._landmark_feasibility.__repr__() +'\n'
        line += 'Robots: '+self._rbt2pose.__repr__() +'\n'
        line += 'Landmarks: '+self._lmk2point.__repr__() +'\n'
        return line

    def robot_edge_path(self, feasiblity = None, start_point: tuple = None)->List[tuple]:
        # the default direction is counter-clockwise
        next_wps = []
        # get a list of waypoints along the edge of feasible area
        if feasiblity is None:
            feasiblity = deepcopy(self.robot_feasibility)

        edge_pts = set()
        feasible_pts = np.array(np.where(feasiblity)).T
        # compute edge points first and then consider their order
        for pt in feasible_pts:
            nb_pts = self.get_neighboring_robot_vertices(*pt, feasibility=feasiblity)
            if len(nb_pts) < 4:
                edge_pts.add((pt[0], pt[1]))

        if start_point is None:
            # take the top left vertex as the start point
            for i in range(feasiblity.shape[0]):
                if start_point is not None:
                    break
                for j in range(feasiblity.shape[1]):
                    if feasiblity[i, j]:
                        start_point = (i, j)
                        break
        next_wps.append(start_point)

        counterclock_nb = [(1,0),(0,1),(-1,0),(0,-1)]

        while True:
            cur_point = next_wps[-1]
            i, j = cur_point
            feasiblity[i, j] = False
            feas_rbt_pts = self. \
                get_neighboring_robot_vertices(i, j, feasiblity)
            if len(feas_rbt_pts) > 0:
                pts_degree = np.array([len(self.get_neighboring_robot_vertices(*pt, feasibility=feasiblity))
                                       for pt in feas_rbt_pts])
                min_degree_idx = np.where(pts_degree == np.amin(pts_degree))[0]
                tmp = []
                for idx in min_degree_idx:
                    if feas_rbt_pts[idx] in edge_pts:
                        tmp.append(idx)
                min_degree_idx = np.array(tmp)
                next_pt_idx = 0
                least_order = np.inf
                for idx in min_degree_idx:
                    diff_vec = (feas_rbt_pts[idx][0]-i,feas_rbt_pts[idx][1]-j)
                    cur_order = counterclock_nb.index(diff_vec)
                    if cur_order < least_order:
                        least_order = cur_order
                        next_pt_idx = idx
                next_wps.append(feas_rbt_pts[next_pt_idx])
                if len(next_wps) == len(edge_pts):
                    if set(next_wps) == edge_pts:
                        if start_point in set(self.get_neighboring_vertices(*next_wps[-1])):
                            next_wps.append(start_point)
                            break
                        else:
                            raise ValueError("Edge points cannot form a loop.")
                    else:
                        raise ValueError("Non-edge vertices are added.")
            else:
                break
        return next_wps

    def robot_lawn_mower(self, feasiblity = None)->List[tuple]:
        # the default direction is counter-clockwise
        next_wps = []
        # get a list of waypoints along the edge of feasible area
        if feasiblity is None:
            feasiblity = deepcopy(self.robot_feasibility)

        inverse_i = False
        for j in range(feasiblity.shape[1]):
            if feasiblity[:,j].any():
                indices = np.where(feasiblity[:,j])[0]
                if not inverse_i:
                    for i in indices:
                        next_wps.append((i,j))
                else:
                    for i in indices[::-1]:
                        next_wps.append((i,j))
                inverse_i = not inverse_i
        return next_wps

    def plaza1_path(self)->List[tuple]:
        edge_path = self.robot_edge_path()
        lawn_mower = self.robot_lawn_mower()
        return edge_path[:-1] + lawn_mower

class ManhattanWorld(ManhattanWaterworld):
    """
    This class creates a simulated environment of Manhattan world.

    ---
    :param
    grid_vertices_shape: a tuple defining the shape of grid vertices
    cell_scale: width and length of a cell
    """
    def __init__(self, grid_vertices_shape: tuple = (9,9),
                 cell_scale: int = 1):
        super().__init__(grid_vertices_shape, cell_scale)
