import numpy as np
import platform
import math
import matplotlib.pyplot as plt
import os
import shutil

sysstr = platform.system()


class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, obj):
        return Vector3(self.x + obj.x, self.y + obj.y, self.z + obj.z)

    def __sub__(self, obj):
        return Vector3(self.x - obj.x, self.y - obj.y, self.z - obj.z)

    def __mul__(self, obj):
        if (type(obj) == Vector3):
            return Vector3(self.y * obj.z - self.z * obj.y, self.z * obj.x - self.x * obj.z,
                           self.x * obj.y - self.y * obj.x)
        if (type(obj) == float or type(obj) == int):
            return Vector3(self.x * obj, self.y * obj, self.z * obj)
        assert (False)

    def __str__(self):
        return str('(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')')

    def length2(self):
        return float(self.x * self.x + self.y * self.y + self.z * self.z)

    def length(self):
        return (self.x * self.x + self.y * self.y + self.z * self.z) ** .5

    def norm(self):
        l = self.length()
        return Vector3(self.x / l, self.y / l, self.z / l)

    def __eq__(self, other):
        assert (type(other) == Vector3)
        if (abs(self.x - other.x) < 1e-8 and abs(self.y - other.y) < 1e-8 and abs(self.z - other.z) < 1e-8):
            return True
        else:
            return False


class Point:
    def __init__(self, vec=Vector3(), supportX=0, supportY=0, supportZ=0, loadX=0.0, loadY=0.0, loadZ=0.0):
        self.vec = vec

        self.supportX = supportX
        self.supportY = supportY
        self.supportZ = supportZ
        self.isSupport = False
        if (supportX == 1 or supportY == 1 or supportZ == 1):
            self.isSupport = True

        self.loadX = loadX
        self.loadY = loadY
        self.loadZ = loadZ
        self.isLoad = False
        if (abs(loadX) > 1e-7 or abs(loadY) > 1e-7 or abs(loadZ) > 1e-7):
            self.isLoad = True


class Bar:
    def __init__(self, u=-1, v=-1, area=1.0, leng=0.0, inertia=1.0):
        self.u = int(u)
        self.v = int(v)
        self.area = float(area)
        self.force = float(0.0)
        self.len = leng
        self.stress = 0.0
        self.inertia = float(inertia)


def getlen(vec):
    return (vec.x * vec.x + vec.y * vec.y + vec.z * vec.z) ** .5


def getlen2(u, v):
    return getlen(u.vec - v.vec)


def getang(vec1, vec2):
    return (vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z) / (getlen(vec1) * getlen(vec2))


def is_edge_addable(u, v, points, edges, enabled=False):
    r'''
    Check if adding a bar between u and v is valid, only applied to 2-d case

    :param u: index of one end of the edge
    :param v: index of the other end of the edge
    :param points: nodes
    :param edges: edges
    :param enabled: Whether use this function to check edge constraint, if False, always return True
    :return: bool
    '''

    max_length = 18
    minang = 10
    cosminang = np.cos(minang / 180.0 * np.pi)
    max_edges = 10

    #判断杆件是否交叉
    def _intersect(point_u1,point_v1,point_u2,point_v2): #四个点对象，其中u1v1为一根杆，u2v2为一根杆
        intersected = False

        u1=np.array([point_u1.vec.x,point_u1.vec.y])
        v1=np.array([point_v1.vec.x,point_v1.vec.y])
        u2=np.array([point_u2.vec.x,point_u2.vec.y])
        v2=np.array([point_v2.vec.x,point_v2.vec.y])      #取得四个点坐标向量

        u1v1=v1-u1
        u2v2=v2-u2     #杆件向量

        u1u2=u2-u1
        u1v2=v2-u1

        u2u1=u1-u2
        u2v1=v1-u2

        def compare(a,b):
            if((a[0] < b[0]) or (a[0] == b[0] and a[1] < b[1])):
                return -1
            elif(a[0] == b[0] and a[1] == b[1]):
                return 0
            else:
                return 1
        #对一条线段的两端点进行排序，横坐标大的点更大，横坐标相同，纵坐标大的点更大，升序排序
        po=[u1,v1,u2,v2]
        if compare(po[0],po[1])>0:
            temp=po[0]
            po[0]=po[1]
            po[1]=temp
        if compare(po[2],po[3])>0:
            temp=po[2]
            po[2]=po[3]
            po[3]=temp

        #考虑一般情况
        if  ((np.cross(u1v1,u1u2)*np.cross(u1v1,u1v2)<0 and np.cross(u2v2,u2u1)*np.cross(u2v2,u2v1)<0) or    #叉积均小于0，跨越交叉
            (np.cross(u1v1,u1u2)*np.cross(u1v1,u1v2)==0 and np.cross(u2v2,u2u1)*np.cross(u2v2,u2v1)<0) or    #任意一方=0， 另一方<0，为一节点位于另一杆件上
            (np.cross(u1v1,u1u2)*np.cross(u1v1,u1v2)<0 and np.cross(u2v2,u2u1)*np.cross(u2v2,u2v1)==0)):     #顺便排除了有公共点的情况，有公共点两方均为0
            intersected = True

        #考虑如果两线段共线重叠
        if np.cross(u1v1,u2v2)==0 and np.cross(u1v1,u1v2)==0: #两线段共线
            if(compare(po[0],po[2]) <= 0 and compare(po[1],po[2]) > 0):     #第一条起点小于第二条起点，第一条终点大于第二条起点
                intersected = True
            elif(compare(po[2],po[0]) <= 0 and compare(po[3],po[0]) > 0):   #第二条起点小于第一条起点，第二条终点大于第一条起点
                intersected = True

        return intersected

    def _transintersect(
        u1,v1,u2,v2,
        points,
    ): # ?
        if (
            _intersect(
                points[u1], points[v1], points[u2], points[v2]
            )
        ):
            return True

        if (u1 == u2):
            if (
                getang(
                    points[v1].vec - points[u1].vec,
                    points[v2].vec - points[u2].vec,
                ) > cosminang
            ):
                return True
        if (u1 == v2):
            if (
                getang(
                    points[v1].vec - points[u1].vec,
                    points[u2].vec - points[v2].vec,
                ) > cosminang
            ):
                return True
        if (v1 == u2):
            if (
                getang(
                    points[u1].vec - points[v1].vec,
                    points[v2].vec - points[u2].vec,
                ) > cosminang
            ):
                return True
        if (v1 == v2):
            if (
                getang(
                    points[u1].vec - points[v1].vec,
                    points[u2].vec - points[v2].vec,
                ) > cosminang
            ):
                return True

        return False

    def _is_too_long(point_u, point_v):
        return getlen2(point_u, point_v) > max_length

    # MODIFICATION: not considering EDGE_CONFIG

    if not enabled:
        return True

    if _is_too_long(points[u], points[v]):
        return False

    if points[u].isSupport and points[v].isSupport:
        return False

    for edge in edges.values():
        if (
            _transintersect(
                u, v, edge.u, edge.v, points
            )
        ):
            return False

    return True


def getuv(x):
    x += 1
    v = math.ceil(
        (math.sqrt(1 + 8 * x) - 1) / 2.0
    )
    u = x - v * (v - 1) // 2 - 1
    return u, v


def readFile(input_file):
    r'''

    :param input_file: File name
    :return: point list, edge list
    '''

    p = []
    e = []

    with open(input_file, "r") as fle:
        lines = fle.readlines()
        for i in range(len(lines)):
            if len(lines[i]) < 2:
                continue
            line = lines[i]
            vec = line.strip().split(' ')
            if (i == 0):
                vn = int(vec[0])
                en = int(vec[1])
                continue

            if (1 <= i and i <= vn):
                p.append(Point(Vector3(float(vec[0]), float(vec[1]), float(vec[2])), int(vec[3]), int(vec[4]), int(vec[5]), float(vec[6]), float(vec[7]), float(vec[8])))
                continue

            if (vn + 1 <= i and i <= vn + en):
                e.append(Bar(vec[0], vec[1], float(vec[2]), getlen2(p[int(vec[0])], p[int(vec[1])])))
                continue
    return p, e


def save_file(initial_points, state, mass, path):
    r'''
    save state into txt
    :param initial_points: initial points, for support and load information
    :param state: truss state
    :param mass: mass of truss
    :param path: path to store
    :return: None
    '''
    fo = open(path + str(int(mass * 1000)) + ".txt", "w")
    n = state.num_points
    fo.write("{} {}\n".format(n, n * (n - 1) // 2))
    for i in range(n):
        x = state.nodes[i][0]
        y = state.nodes[i][1]
        if state.dimension == 2:
            z = 0.0
        else:
            z = state.nodes[i][2]
        fo.write("{} {} {} {} {} {} {} {} {}\n".format(x, y, z,
                                                       initial_points[i].supportX, initial_points[i].supportY, initial_points[i].supportZ,
                                                       initial_points[i].loadX, initial_points[i].loadY, initial_points[i].loadZ))
    for i in range(n):
        for j in range(i):
            fo.write("{} {} {}\n".format(j, i, state.edges[i][j]))
    fo.close()


def save_trajectory(initial_points, trajectory, mass, path):
    r'''
    save state into txt
    :param initial_points: initial points, for support and load information
    :param trajectory: history of truss states
    :param mass: mass of truss
    :param path: path to store
    :return: None
    '''
    current_dir = os.getcwd()
    dir = path + str(int(mass))
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    os.chdir(dir)
    for i in range(len(trajectory)):
        state = trajectory[i]

        def _save_file(initial_points, state, file_name):
            r'''
            save state into txt
            :param initial_points: initial points, for support and load information
            :param state: truss state
            :param mass: mass of truss
            :param path: path to store
            :return: None
            '''
            fo = open(file_name, "w")
            n = state.num_points
            fo.write("{} {}\n".format(n, n * (n - 1) // 2))
            for i in range(n):
                x = state.nodes[i][0]
                y = state.nodes[i][1]
                if state.dimension == 2:
                    z = 0.0
                else:
                    z = state.nodes[i][2]
                fo.write("{} {} {} {} {} {} {} {} {}\n".format(x, y, z,
                                                               initial_points[i].supportX, initial_points[i].supportY, initial_points[i].supportZ,
                                                               initial_points[i].loadX, initial_points[i].loadY, initial_points[i].loadZ))
            for i in range(n):
                for j in range(i):
                    fo.write("{} {} {}\n".format(j, i, state.edges[i][j]))
            fo.close()

        def _saveGraph(p, e, file):
            for i in range(len(p)):
                plt.scatter([p[i].vec.x], [p[i].vec.y], color='b')

            for i in range(len(e)):
                x0 = [p[e[i].u].vec.x, p[e[i].v].vec.x]
                y0 = [p[e[i].u].vec.y, p[e[i].v].vec.y]

                if e[i].area != -1:
                    plt.plot(x0, y0, color='b', linewidth=e[i].area / 0.01)

            plt.axis("equal")
            plt.savefig(file)
            plt.cla()

        _save_file(initial_points, state, str(i) + ".txt")

    os.chdir(current_dir)
