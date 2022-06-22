import time
import json
import numpy as np
import math
import random
import copy

import matplotlib.pyplot as plt
import warnings
import os, sys, contextlib
import openseespy.opensees as op

import heapq

FILENAME = "input-10bar-case1.txt"

time_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
LOGFOLDER = "./results/"
if not os.path.exists(LOGFOLDER):
    os.mkdir(LOGFOLDER)
LOGFILE = LOGFOLDER + time_str + ".log"
LOGFILE_Result = LOGFOLDER + time_str + "_decision.log"

LOG = open(LOGFILE, "w")
LOG_result = open(LOGFILE_Result, "w")

c = [16.0, 14.0, 9.0]
rate2 = 0.9
prob = 1.0
maxp = 6
print('maxp:', maxp, file=LOG)
alpha = 0.2
# rate = 0.005
rate = 0
pickalpha = 0.2
bestreward = 1000000
tmpbestreward = 1000000
bestMass = 1000000.0
pbest = []
ebest = []
mass_pbest = []
mass_ebest = []
baseiter = 1500
dislimit = 0.0508
useIntersect = True
useMaxlen = True
usePlist = True
useSelfWeight = False
useMaxlen = True
# constraints
CONSTRAINT_STRESS = 1
CONSTRAINT_DIS = 1
CONSTRAINT_BUCKLE = 0
print('CONSTRAINT_STRESS:', CONSTRAINT_STRESS, file=LOG)
print('CONSTRAINT_DIS:', CONSTRAINT_DIS, file=LOG)
print('CONSTRAINT_BUCKLE:', CONSTRAINT_BUCKLE, file=LOG)
USE_SOFT_PENALTY = False
USE_SOFT_DECAY = False
DECAY_RANGE = 20000
decay_iter = 0
print('USE_SOFT_PENALTY:', USE_SOFT_PENALTY, file=LOG)
print('USE_SOFT_DECAY:', USE_SOFT_DECAY, file=LOG)
print('DECAY_RANGE:', DECAY_RANGE, file=LOG)

reward_fun_count = 0

MULTIFOLDER = LOGFOLDER + 'Reward_MULTI_Result/'
if not os.path.exists(MULTIFOLDER):
    os.mkdir(MULTIFOLDER)
MultiList = []  # (- reward, p, e)
MultiNum = 100
print('MultiNum:', MultiNum, file=LOG)

MASS_MULTIFOLDER = LOGFOLDER + 'MASS_MULTI_Result/'
if not os.path.exists(MASS_MULTIFOLDER):
    os.mkdir(MASS_MULTIFOLDER)
Mass_MultiList = []  # (- reward, p, e)
Mass_MultiNum = 100
print('Mass_MultiNum:', Mass_MultiNum, file=LOG)

MASS_DIVFOLDER = LOGFOLDER + 'MASS_DIV_Result/'
if not os.path.exists(MASS_DIVFOLDER):
    os.mkdir(MASS_DIVFOLDER)
Mass_DIVList = []  # (- reward, p, e)
Mass_DIVNum = 1000
print('Mass_DIVNum:', Mass_DIVNum, file=LOG)

ALLFOLDER = LOGFOLDER + 'Reward_ALL_Result/'
if not os.path.exists(ALLFOLDER):
    os.mkdir(ALLFOLDER)
OUTPUT_ALL_MAX = 10000
OUTPUT_ALL_THRESHOLD = 3000
print('OUTPUT_ALL_THRESHOLD:', OUTPUT_ALL_THRESHOLD, file=LOG)

MASS_ALLFOLDER = LOGFOLDER + 'MASS_ALL_Result/'
if not os.path.exists(MASS_ALLFOLDER):
    os.mkdir(MASS_ALLFOLDER)
MASS_OUTPUT_ALL_MAX = 10000
MASS_OUTPUT_ALL_THRESHOLD = 3000
print('MASS_OUTPUT_ALL_THRESHOLD:', MASS_OUTPUT_ALL_THRESHOLD, file=LOG)

ADD_RANDOM_RATIO = 0.5
print('ADD_RANDOM_RATIO:', ADD_RANDOM_RATIO, file=LOG)

maxlen = 18.2
pstate = 0
estate = 0
qhead = 0
que = []
sgm1 = 0.0005
sgm2 = 0.3
minlen = 0.5
maxx = 0.0
maxy = 0.0
minx = 0.0
miny = 0.0
minarea = 0.0005
maxarea = 0.020
maxnum = 50
maxson = 200
minang = 10
initson = 20
cosminang = math.cos(minang / 180.0 * math.pi)

reward_lambda = 2000 * 2000 * 10
ratio_ring = 0.0

# Elasticity modulus
E = 6.895 * 10 ** 10
pho = 2.76799 * 10 ** 3
Sigma_T = 172.3 * 10 ** 6
Sigma_C = 172.3 * 10 ** 6


def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


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
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

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

    def __init__(self, vec=Vector3(), supportX=0, supportY=0, supportZ=1, loadX=0.0, loadY=0.0, loadZ=0.0):
        self.vec = vec

        self.supportX = supportX
        self.supportY = supportY
        self.supportZ = supportZ
        self.isSupport = False
        # 2D
        if (supportX == 1 or supportY == 1):
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


class Load:

    def __init__(self, u=-1, fx=0.0, fy=0.0, fz=0.0):
        self.u = int(u)
        self.fx = float(fx)
        self.fy = float(fy)
        self.fz = float(fz)


def drawGraph(p, e, canshow=1, reward=0.0):
    for i in range(len(p)):
        plt.scatter([p[i].vec.x], [p[i].vec.y], color='b')

    for i in range(len(e)):
        x0 = [p[e[i].u].vec.x, p[e[i].v].vec.x]
        y0 = [p[e[i].u].vec.y, p[e[i].v].vec.y]

        if (e[i].stress < 0):
            plt.plot(x0, y0, color='g', linewidth=e[i].area / 0.01)
        else:
            plt.plot(x0, y0, color='r', linewidth=e[i].area / 0.01)

    plt.axis("equal")
    plt.title(str(reward))
    if (canshow == 1):
        plt.show()


def saveGraph(p, e, bestreward=0.0):
    for i in range(len(p)):
        plt.scatter([p[i].vec.x], [p[i].vec.y], color='b')

    for i in range(len(e)):
        x0 = [p[e[i].u].vec.x, p[e[i].v].vec.x]
        y0 = [p[e[i].u].vec.y, p[e[i].v].vec.y]

        if (e[i].stress < 0):
            plt.plot(x0, y0, color='g', linewidth=e[i].area / 0.01)
        else:
            plt.plot(x0, y0, color='r', linewidth=e[i].area / 0.01)

    plt.axis("equal")
    plt.title(str(bestreward))
    inputname = FILENAME.replace(".txt", "_")
    FILENAME_jpg = LOGFOLDER + str(len(p)) + "p_" + inputname + str(round(bestreward, 2)) + ".jpg"

    plt.savefig(FILENAME_jpg, dpi=1000)


def draw2Graph(p1, e1, p2, e2, canshow=1):
    sub1 = plt.subplot(1, 2, 1)
    sub2 = plt.subplot(1, 2, 2)
    plt.sca(sub1)
    drawGraph(p1, e1, 0)
    plt.sca(sub2)
    drawGraph(p2, e2, 0)
    if (canshow == 1):
        plt.show()


def getlen2(u, v):
    return math.sqrt((u.vec.x - v.vec.x) ** 2 + (u.vec.y - v.vec.y) ** 2 + (u.vec.z - v.vec.z) ** 2)


def getlen(vec):
    return math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)


def intersect(N1, N2, N3, N4):
    game_stop = False
    X1 = N1.x
    Y1 = N1.y
    X2 = N2.x
    Y2 = N2.y
    X3 = N3.x
    Y3 = N3.y
    X4 = N4.x
    Y4 = N4.y

    if N1 != N3 and N1 != N4 and N2 != N3 and N2 != N4:
        SIN13_14 = (X3 - X1) * (Y4 - Y1) - (X4 - X1) * (Y3 - Y1)
        SIN23_24 = (X3 - X2) * (Y4 - Y2) - (X4 - X2) * (Y3 - Y2)
        SIN31_32 = (X1 - X3) * (Y2 - Y3) - (X2 - X3) * (Y1 - Y3)
        SIN41_42 = (X1 - X4) * (Y2 - Y4) - (X2 - X4) * (Y1 - Y4)

        if SIN13_14 * SIN23_24 <= 0 and SIN31_32 * SIN41_42 <= 0:
            SIN12_23 = (X2 - X1) * (Y3 - Y2) - (X3 - X2) * (Y2 - Y1)
            SIN12_24 = (X2 - X1) * (Y4 - Y2) - (X4 - X2) * (Y2 - Y1)
            SIN23_34 = (X3 - X2) * (Y4 - Y3) - (X4 - X3) * (Y3 - Y2)
            SIN13_34 = (X3 - X1) * (Y4 - Y3) - (X4 - X3) * (Y3 - Y1)

            if SIN12_23 != 0 and SIN12_24 != 0 and SIN23_34 != 0 and SIN13_34 != 0:
                game_stop = True

    SIN13_14 = (X3 - X1) * (Y4 - Y1) - (X4 - X1) * (Y3 - Y1)
    SIN23_24 = (X3 - X2) * (Y4 - Y2) - (X4 - X2) * (Y3 - Y2)
    if (abs(SIN13_14) < 1e-7 and abs(SIN23_24) < 1e-7 and useIntersect):
        D13 = math.sqrt((X3 - X1) * (X3 - X1) + (Y3 - Y1) * (Y3 - Y1))
        D14 = math.sqrt((X4 - X1) * (X4 - X1) + (Y4 - Y1) * (Y4 - Y1))
        D23 = math.sqrt((X3 - X2) * (X3 - X2) + (Y3 - Y2) * (Y3 - Y2))
        D24 = math.sqrt((X4 - X2) * (X4 - X2) + (Y4 - Y2) * (Y4 - Y2))
        D1 = D13 + D24
        D2 = D23 + D14
        if (abs(D1 - D2) > 1e-7):
            game_stop = True
    return game_stop


def getang(vec1, vec2):
    return (vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z) / (getlen(vec1) * getlen(vec2))


def transintersect(u1, v1, u2, v2, p):
    if (intersect(p[u1].vec, p[v1].vec, p[u2].vec, p[v2].vec)):
        return True
    if (u1 == u2):
        if (getang(p[v1].vec - p[u1].vec, p[v2].vec - p[u2].vec) > cosminang):
            return True
    if (u1 == v2):
        if (getang(p[v1].vec - p[u1].vec, p[u2].vec - p[v2].vec) > cosminang):
            return True
    if (v1 == u2):
        if (getang(p[u1].vec - p[v1].vec, p[v2].vec - p[u2].vec) > cosminang):
            return True
    if (v1 == v2):
        if (getang(p[u1].vec - p[v1].vec, p[u2].vec - p[v2].vec) > cosminang):
            return True

    return False


def readFile():
    p = []
    e = []
    pload = []
    # print(FILENAME)

    with open(FILENAME, "r") as fle:
        lines = fle.readlines()
        for i in range(len(lines)):
            line = lines[i]
            vec = line.strip().split(' ')
            if (i == 0):
                vn = int(vec[0])
                en = int(vec[1])
                continue

            if (1 <= i and i <= vn):
                p.append(
                    Point(Vector3(float(vec[0]), float(vec[1]), float(vec[2])), int(vec[3]), int(vec[4]), int(vec[5]),
                          float(vec[6]), float(vec[7]), float(vec[8])))
                pload.append(Load(i - 1, float(vec[6]), float(vec[7]), float(vec[8])))
                continue

            if (vn + 1 <= i and i <= vn + en):
                e.append(Bar(vec[0], vec[1], float(vec[2]), getlen2(p[int(vec[0])], p[int(vec[1])])))
                continue

    return p, e, pload


def similar(tup1, tup2):
    p1 = tup1[2]
    e1 = tup1[3]
    p2 = tup2[2]
    e2 = tup2[3]
    pts1 = []
    pts2 = []
    for p in p1:
        pts1.append([p.vec.x, p.vec.y, p.vec.z])
    for p in p2:
        pts2.append([p.vec.x, p.vec.y, p.vec.z])
    es1 = []
    es2 = []
    for e in e1:
        es1.append([pts1[e.u], pts1[e.v]])
        es1.append([pts1[e.v], pts1[e.u]])
    for e in e2:
        es2.append([pts2[e.u], pts2[e.v]])
        es2.append([pts2[e.v], pts2[e.u]])

    if sorted(pts1) != sorted(pts2):
        return False
    if sorted(es1) != sorted(es2):
        return False
    return True

    for p in pts1:
        if not (p in pts2):
            return False
    for p in pts2:
        if not (p in pts1):
            return False
    for e in es1:
        if not (e in es2):
            return False
    for e in es2:
        if not (e in es1):
            return False
    return True


def reward_fun(p, e, pload):
    global pbest
    global ebest
    global bestreward
    global tmpbestreward
    global bestMass
    global reward_fun_count
    global mass_pbest
    global mass_ebest

    reward_fun_count += 1

    # check_geometruc_stability
    Support_num = 0
    for i in range(len(p)):
        Support_num = Support_num + p[i].supportX + p[i].supportY + p[i].supportZ

    # print(Support_num)
    Freedom = len(p) * 3 - len(e) - Support_num
    # print(Freedom)
    if (Freedom > 0):
        return -1.0

    blockPrint()
    # print(123)

    # Units
    m = 1
    Pa = 1
    kg = 1

    # remove existing model
    op.wipe()

    # ndm表示：2维问题，ndf：每个节点有2个自由度，当其为3时，节点还能表示转动
    op.model('basic', '-ndm', 2, '-ndf', 2)

    # create nodes[节点编号，x坐标，y坐标]
    for i in range(len(p)):
        op.node(i, p[i].vec.x, p[i].vec.y, p[i].vec.z)

    # set boundary condition [节点编号，x方向是否约束（1为约束，0为不约束），y同理]
    for i in range(len(p)):
        if (p[i].isSupport):
            op.fix(i, p[i].supportX, p[i].supportY, p[i].supportZ)

    # define materials[定义材料本身的属性，弹性模量，matTag=1] 1代表材料编号
    op.uniaxialMaterial("Elastic", 1, E)

    # define elements
    # op.element('Truss', eleTag, *eleNodes, A, matTag[, '-rho', rho][, '-cMass', cFlag][, '-doRayleigh', rFlag])
    for i in range(len(e)):
        op.element("Truss", i, e[i].u, e[i].v, e[i].area, 1)

    # create TimeSeries
    op.timeSeries("Linear", 1)

    # create a plain load pattern
    op.pattern("Plain", 1, 1)

    # Create the nodal load - command: load nodeID xForce yForce
    # for i in range(len(pload)):
    # 	print(pload[i].u)
    # 	op.load(pload[i].u, pload[i].fx, pload[i].fy, pload[i].fz)
    for i in range(len(p)):
        if (p[i].isLoad):
            # print(i)
            op.load(i, pload[i].fx, pload[i].fy)

    # SelfWeight
    if useSelfWeight:
        gravity = 9.8
        Load_gravity = [0] * len(p)

        for i in range(len(e)):
            Mass_bar = e[i].len * e[i].area * pho
            Load_gravity[e[i].u] = Mass_bar * gravity * 0.5 + Load_gravity[e[i].u]
            Load_gravity[e[i].v] = Mass_bar * gravity * 0.5 + Load_gravity[e[i].v]

        for i in range(len(p)):
            op.load(i, 0.0, -1 * Load_gravity[i])

    ############################################
    # Start of analysis generation

    # create SOE
    op.system("BandSPD")

    # create DOF number
    op.numberer("RCM")

    # create constraint handler
    op.constraints("Plain")

    # create integrator
    op.integrator("LoadControl", 1.0)

    # create algorithm
    op.algorithm("Newton")

    # create analysis object
    op.analysis("Static")
    # blockPrint()
    # perform the analysis
    # drawGraph(p, e, canshow = 1, reward = 0.0)
    # print(len(p))
    # print(len(e))
    ok = op.analyze(1)

    enablePrint()
    # print(ok)
    # Perform the analysis. Return 0 if successful, <0 if NOT successful
    if ok == 0:

        # print(ok)
        Objective_weight = 1
        Constraint_stress_weight = np.zeros((len(e), 1))
        Constraint_displacement_weight = np.zeros((len(p), 1))
        Constraint_buckle_weight = np.zeros((len(e), 1))

        # get mass
        Mass = 0
        for i in range(len(e)):
            Mass += e[i].len * e[i].area * pho
        # print(Mass,'\n')

        # get dispalcement
        Dis_value = 0.0
        if CONSTRAINT_DIS == 1:
            limit_dis = dislimit
            p_u = []
            for i in range(len(p)):
                p_u.append(max(abs(op.nodeDisp(i, 1)), abs(op.nodeDisp(i, 2)), ))
                Constraint_displacement_weight[i] = max(p_u[i] / limit_dis - 1, 0)
            # Dis_max=max(p_u) #最大位移
            Dis_value = float(sum(Constraint_displacement_weight))
            # print(Dis_value)
            p_u_mm = []
            for i in range(len(p_u)):
                p_u_mm.append(round(p_u[i] * 1000, 3))
        # print("p_u_mm=",p_u_mm)

        # get stress
        Stress_value = 0.0
        if CONSTRAINT_STRESS == 1:
            stress = []
            for i in range(len(e)):
                e[i].force = op.basicForce(i)
                stress.append(e[i].force[0] / e[i].area)
                e[i].stress = stress[i]
                if stress[i] < 0:
                    Constraint_stress_weight[i] = max(abs(stress[i]) / Sigma_C - 1.0, 0)
                else:
                    Constraint_stress_weight[i] = max(abs(stress[i]) / Sigma_T - 1.0, 0)
            Stress_value = float(sum(Constraint_stress_weight))
            # print(Stress_value)
            stress_mpa = []
            for i in range(len(stress)):
                stress_mpa.append(round(stress[i] / 10 ** 6, 2))
        # print("stress_mpa= ", stress_mpa)

        # get buckle bound
        Buckle_value = 0.0
        if CONSTRAINT_BUCKLE == 1:
            miu_buckle = 1.0
            stress = []
            Buckle_stress_max = []
            bar_in_c = []
            bar_in_c_cr = []
            for i in range(len(e)):
                e[i].force = op.basicForce(i)
                stress.append(e[i].force[0] / e[i].area)
                if stress[i] < 0:
                    # ratio_ring=0
                    e[i].inertia = e[i].area ** 2 * (1 + ratio_ring ** 2) / (4 * math.pi * (1 - ratio_ring ** 2))
                    Force_cr = math.pi ** 2 * E * e[i].inertia / (miu_buckle * e[i].len) ** 2
                    Buckle_stress_max = Force_cr / e[i].area
                    bar_in_c_cr.append(round(Force_cr / e[i].area / 1000000, 2))
                    bar_in_c.append(round(abs(stress[i]) / 1000000, 2))
                    Constraint_buckle_weight[i] = max(abs(stress[i]) / abs(Buckle_stress_max) - 1.0, 0)

            Buckle_value = float(sum(Constraint_buckle_weight))
        # print(Constraint_stress_weight)
        # print(Buckle_value)

        if not USE_SOFT_PENALTY:
            if (Dis_value > 1e-7 or Stress_value > 1e-7 or Buckle_value > 1e-7):
                reward = 0.0
                return reward

        reward = Mass * (Objective_weight + Dis_value + Stress_value + Buckle_value)

        can_output = True
        for tup in MultiList:
            if round(- tup[0]) == round(reward):
                can_output = False
        if can_output:
            if len(MultiList) < MultiNum:
                heapq.heappush(MultiList, (- reward, reward_fun_count, copy.deepcopy(p), copy.deepcopy(e)))
            else:
                heapq.heappush(MultiList, (- reward, reward_fun_count, copy.deepcopy(p), copy.deepcopy(e)))
                _ = heapq.heappop(MultiList)

        can_output = True
        for tup in Mass_MultiList:
            if round(- tup[0]) == round(Mass):
                can_output = False
        if can_output:
            if not (Dis_value > 1e-7 or Stress_value > 1e-7 or Buckle_value > 1e-7):
                if len(Mass_MultiList) < Mass_MultiNum:
                    heapq.heappush(Mass_MultiList, (- Mass, reward_fun_count, copy.deepcopy(p), copy.deepcopy(e)))
                else:
                    heapq.heappush(Mass_MultiList, (- Mass, reward_fun_count, copy.deepcopy(p), copy.deepcopy(e)))
                    _ = heapq.heappop(Mass_MultiList)

        can_DIVoutput = True
        for tup in Mass_DIVList:
            if round(- tup[0]) == round(Mass):
                can_DIVoutput = False

        if can_DIVoutput:
            if not (Dis_value > 1e-7 or Stress_value > 1e-7 or Buckle_value > 1e-7):
                sim_list = []
                tup_now = (- Mass, reward_fun_count, copy.deepcopy(p), copy.deepcopy(e))
                for i in range(len(Mass_DIVList)):
                    if similar(Mass_DIVList[i], tup_now):
                        sim_list.append([i, Mass_DIVList[i][0]])
                if len(sim_list) < 50:
                    if len(Mass_DIVList) < Mass_DIVNum:
                        heapq.heappush(Mass_DIVList, (- Mass, reward_fun_count, copy.deepcopy(p), copy.deepcopy(e)))
                    else:
                        heapq.heappush(Mass_DIVList, (- Mass, reward_fun_count, copy.deepcopy(p), copy.deepcopy(e)))
                        _ = heapq.heappop(Mass_DIVList)
                else:
                    smallest = 1
                    s_id = -1
                    for sim in sim_list:
                        if sim[1] < smallest:
                            smallest = sim[1]
                            s_id = sim[0]
                    Mass_DIVList[s_id] = tup_now

        FILES = os.listdir(ALLFOLDER)
        if len(FILES) < OUTPUT_ALL_MAX and reward <= OUTPUT_ALL_THRESHOLD:
            OUTFILE = ALLFOLDER + str(reward_fun_count).zfill(len(str(OUTPUT_ALL_MAX))) + '_' + str(
                round(Mass)) + '_' + str(round(reward)) + '.txt'
            with open(OUTFILE, "w") as f:
                print(len(p), len(e), file=f)
                for i in range(len(p)):
                    print(p[i].vec.x, p[i].vec.y, p[i].vec.z, p[i].supportX, p[i].supportY,
                          p[i].supportZ, p[i].loadX, p[i].loadY, p[i].loadZ, file=f)

                for i in range(len(e)):
                    print(e[i].u, e[i].v, e[i].area, file=f)

        MASS_FILES = os.listdir(MASS_ALLFOLDER)
        if len(MASS_FILES) < MASS_OUTPUT_ALL_MAX and Mass <= MASS_OUTPUT_ALL_THRESHOLD:
            if not (Dis_value > 1e-7 or Stress_value > 1e-7 or Buckle_value > 1e-7):
                # MASS_OUTFILE = MASS_ALLFOLDER + str(reward_fun_count).zfill(
                #	len(str(MASS_OUTPUT_ALL_MAX))) + '_' + str(round(Mass)) + '_' + str(round(reward)) + '.txt'
                MASS_OUTFILE = MASS_ALLFOLDER + str(round(Mass)) + '.txt'
                with open(MASS_OUTFILE, "w") as f:
                    print(len(p), len(e), file=f)
                    for i in range(len(p)):
                        print(p[i].vec.x, p[i].vec.y, p[i].vec.z, p[i].supportX, p[i].supportY,
                              p[i].supportZ, p[i].loadX, p[i].loadY, p[i].loadZ, file=f)

                    for i in range(len(e)):
                        print(e[i].u, e[i].v, e[i].area, file=f)

        if (bestMass > Mass):
            if not (Dis_value > 1e-7 or Stress_value > 1e-7 or Buckle_value > 1e-7):
                bestMass = Mass
                mass_pbest = copy.deepcopy(p)
                mass_ebest = copy.deepcopy(e)

        if (bestreward > reward):
            bestreward = reward
            pbest = copy.deepcopy(p)
            ebest = copy.deepcopy(e)

        # drawGraph(p, e, reward = bestreward)
        if (tmpbestreward > reward):
            tmpbestreward = reward

        # reward=1.0/reward*1e4* (4000/reward)
        reward = reward_lambda / (reward * reward)
        # print(float(reward))
        if USE_SOFT_DECAY:
            if (Dis_value > 1e-7 or Stress_value > 1e-7 or Buckle_value > 1e-7):
                reward = reward * (max(0, DECAY_RANGE - decay_iter) / DECAY_RANGE)
        return reward

    else:
        # assert(False)
        reward = -1.0
        return reward


class Action():

    def __init__(self, opt, u=-1, v=-1, area=maxarea, vec=Vector3(), eid=0):
        self.opt = opt
        self.stateid = -1
        if (opt == 0):  # add node
            self.vec = vec
        # print(vec.x, vec.y)
        # assert(minx <= vec.x and vec.x <= maxx and miny <= vec.y and vec.y <= maxy)
        if (opt == 1):  # add edge
            self.opt = opt
            self.u = u
            self.v = v
            self.area = maxarea
        if (opt == 2):
            self.eid = eid
            self.area = area

    def __str__(self):
        if (self.opt == 0):
            return "add node at" + self.vec.__str__()
        if (self.opt == 1):
            if (self.area < 1e-8):
                return "do nothing"
            else:
                return "add edge between" + str(self.u) + "and" + str(self.v)
        if (self.opt == 2):
            return "modify area of " + str(self.eid) + "to " + str(self.area)


def inlen(u, v, p):
    if (not useMaxlen):
        return True

    dx = p[u].vec.x - p[v].vec.x
    dy = p[u].vec.y - p[v].vec.y
    if (dx * dx + dy * dy > maxlen * maxlen):
        return False
    return True


def canadd(N1, N2, p, e):
    for i in range(len(e)):
        N3 = e[i].u
        N4 = e[i].v
        if (transintersect(N1, N2, N3, N4, p)):
            return False

    return True


class State():

    def __init__(self, p, e, opt, fa=0, elist=set(), eid=0):
        # if (opt == 2):
        #	drawGraph(p, e, reward = reward_fun(p, e, gpload))
        self.opt = opt
        self.sons = []

        self.isEnd = False

        self.n = 0
        self.q = 0
        self.fa = fa
        self.allvisited = False
        self.mq = -1000.0
        self.w = 0.0
        self.sumq = 0.0
        self.sumw = 0.0
        if (opt == 0):
            for i in range(len(plist)):
                flag = 1
                for j in range(len(p)):
                    if (getlen(plist[i] - p[j].vec) < minlen):
                        flag = 0
                        break
                if (flag == 1):
                    self.sons.append(Action(0, vec=plist[i]))
        if (opt == 2):
            self.eid = eid
            if (eid >= len(e)):
                self.isEnd = True
                self.reward = reward_fun(p, e, gpload)
                return
            for i in range(len(arealist)):
                self.sons.append(Action(2, area=arealist[i], eid=eid))

        if (opt == 1):
            self.reward = reward_fun(p, e, gpload)
            if (self.reward > 1e-7):
                self.sons.append(Action(1))
            else:
                for i in elist:
                    self.sons.append(Action(1, u=i[0], v=i[1]))

            if (len(self.sons) == 0):
                self.isEnd = True

    def findunvis(self):
        ret = -1
        for i in range(len(self.sons)):
            if (self.sons[i].stateid == -1):
                ret = i
                break

        if (ret == -1):
            self.allvisited = True
        return ret


def bestchild(now, c, alpha, determine=False):
    global statelist
    sons_actid = []
    sons_state = []
    sons_value = []
    assert (len(statelist[now].sons) > 0)
    for i in range(len(statelist[now].sons)):
        v = statelist[now].sons[i].stateid
        if (abs(c) < 1e-7):
            tmp = alpha * statelist[v].q / statelist[v].n + (1 - alpha) * statelist[v].mq
        else:
            tmp = alpha * statelist[v].q / statelist[v].n + (1 - alpha) * statelist[v].mq \
                  + c * math.sqrt(2 * math.log(statelist[now].n) / statelist[v].n)
        sons_state.append(v)
        sons_value.append(tmp)
        sons_actid.append(i)
    sons_dist = np.array(sons_value)
    sons_dist = np.exp(sons_dist)
    sons_dist /= np.sum(sons_dist)

    random_idx = np.random.choice(len(sons_actid), p=sons_dist)
    max_idx = np.argmax(sons_dist)

    if determine:
        idx = max_idx
    else:
        idx = np.random.choice([random_idx, max_idx], p=[ADD_RANDOM_RATIO, 1 - ADD_RANDOM_RATIO])
    actid = sons_actid[idx]
    ret = sons_state[idx]
    return ret, statelist[now].sons[actid]


def take_action(p, e, elist, act):
    if (act.opt == 0):
        p.append(Point(act.vec))
        if (len(p) == maxp):
            for i in range(len(p)):
                for j in range(i + 1, len(p)):
                    if (p[i].isSupport and p[j].isSupport):
                        continue
                    if (not inlen(i, j, p)):
                        continue
                    elist.add((i, j))
    if (act.opt == 1):
        if (act.u != -1 and act.v != -1):
            e.append(Bar(act.u, act.v, act.area, getlen2(p[act.u], p[act.v])))
            dellist = []
            for i in elist:
                if (transintersect(act.u, act.v, i[0], i[1], p)):
                    dellist.append(i)
            for i in dellist:
                elist.remove(i)
        else:
            elist.clear()
    if (act.opt == 2):
        e[act.eid].area = act.area


def treepolicy(stateid, p_, e_, elist_):
    global qhead
    global que
    p = copy.deepcopy(p_)
    e = copy.deepcopy(e_)
    elist = copy.deepcopy(elist_)
    global statelist
    now = stateid
    sonid = -1
    while (True):
        if (statelist[now].isEnd):
            break
        opt = statelist[now].opt
        if (not statelist[now].allvisited):
            ret = statelist[now].findunvis()
            if (ret != -1):
                sonid = ret
                break
        now, act = bestchild(now, c[opt], alpha)
        take_action(p, e, elist, act)

    if (sonid >= 0):
        act = statelist[now].sons[sonid]
        take_action(p, e, elist, act)

        opt = statelist[now].opt
        if (opt == 0):
            if (len(p) == maxp):
                newstate = State(p, e, 1, fa=now, elist=elist)
            else:
                newstate = State(p, e, 0, fa=now)
        if (opt == 1):
            if (act.u < 0):
                newstate = State(p, e, 2, fa=now, eid=0)
            else:
                newstate = State(p, e, 1, fa=now, elist=elist)
        if (opt == 2):
            newstate = State(p, e, 2, fa=now, eid=act.eid + 1)
        if (True):
            statelist.append(newstate)
            statelist[now].sons[sonid].stateid = len(statelist) - 1
            now = len(statelist) - 1
    return now, p, e, elist


def defaultpolicy(stateid, p, e, elist):
    opt = statelist[stateid].opt
    if (opt == 0):
        pl = []
        for i in plist:
            flag = 1
            for j in p:
                if (i == j.vec):
                    flag = 0
                    break
            if (flag == 1):
                pl.append(i)

        random.shuffle(pl)
        for i in range(len(pl)):
            if (len(p) == maxp):
                break
            p.append(Point(pl[i]))
        opt = 1
    if (opt == 1):
        for i in range(len(e)):
            e[i].area = arealist[random.randint(0, len(arealist) - 1)]

        el = []
        for i in elist:
            el.append(i)
        if (len(el) == 0 and len(e) == 0):
            for i in range(len(p)):
                for j in range(i + 1, len(p)):
                    if (p[i].isSupport and p[j].isSupport):
                        continue
                    if (not inlen(i, j, p)):
                        continue
                    el.append((i, j))

        random.shuffle(el)
        for i in range(len(el)):
            probnow = random.random()
            if (probnow > prob):
                continue

            u = el[i][0]
            v = el[i][1]
            if (canadd(u, v, p, e)):
                e.append(Bar(u, v, arealist[random.randint(0, len(arealist) - 1)], getlen2(p[u], p[v])))

        ret = reward_fun(p, e, gpload)
        # drawGraph(p, e)
        ret2 = ret
        while (ret2 > 1e-7):
            ret = ret2
            e.pop()
            ret2 = reward_fun(p, e, gpload)
        return ret

    if (opt == 2):
        for i in range(statelist[stateid].eid, len(e)):
            e[i].area = arealist[random.randint(0, len(arealist) - 1)]

        ret = reward_fun(p, e, gpload)
        return ret

    assert (False)


def backup(now, delta, root):
    while (now != root):
        statelist[now].n = statelist[now].n + 1
        statelist[now].q = statelist[now].q + delta
        if (statelist[now].mq < delta):
            statelist[now].mq = delta
        now = statelist[now].fa

    statelist[now].n = statelist[now].n + 1
    statelist[now].q = statelist[now].q + delta
    if (statelist[now].mq < delta):
        statelist[now].mq = delta


def memoryopt(now):
    if (now < 0):
        return
    global que
    que.append(now)
    for i in range(len(statelist[now].sons)):
        memoryopt(statelist[now].sons[i].stateid)


def UCTSearch(p, e):
    global statelist
    global bestreward
    global tmpbestreward
    global alpha
    global c
    global elist
    global qhead
    global que
    global decay_iter
    elist = set()

    statelist = []

    t0 = 0
    t1 = 0
    t2 = 0

    dep = 30
    opt = 0
    eidnow = 0
    maxiter = 50000 + 10000
    while (True):
        statelist.clear()
        root = 0
        if (opt == 0):
            statelist.append(State(p, e, 0, -1))
        if (opt == 1):
            statelist.append(State(p, e, 1, -1, elist=elist))
        if (opt == 2):
            statelist.append(State(p, e, 2, -1, eid=eidnow))
        maxiter = maxiter - 1000
        maxiter2 = 0
        # if (dep > 0):
        #	maxiter = maxiter + dep * (baseiter * int(math.sqrt(len(statelist[root].sons))))
        if (dep == 30):
            maxiter2 = 50000
        if (dep == 29):
            maxiter2 = 25000
        dep = dep - 1
        # maxiter2 = 2000 * int(math.sqrt(len(statelist[root].sons)))
        # maxiter = maxiter - statelist[root].n // 4
        # if (maxiter2 > maxiter):
        #	maxiter = maxiter2
        print(maxiter)
        decay_iter = 0
        for iter in range((maxiter + maxiter2)):
            decay_iter += 1
            # print(iter)
            tt0 = time.time()
            tmp, ptmp, etmp, elisttmp = treepolicy(root, p, e, elist)
            tt1 = time.time()
            # print(tmp)

            delta = defaultpolicy(tmp, ptmp, etmp, elisttmp)
            # print(delta)
            tt2 = time.time()
            backup(tmp, delta, root)
            tt3 = time.time()
            t0 = t0 + tt1 - tt0
            t1 = t1 + tt2 - tt1
            t2 = t2 + tt3 - tt2

            if (iter % 3000 == 0):

                # print(iter, bestreward, bestMass, tmpbestreward, alpha, t0, t1, t2, len(statelist), len(statelist[root].sons))
                print(iter, bestreward, bestMass, tmpbestreward, alpha, t0, t1, t2, len(statelist),
                      len(statelist[root].sons), file=LOG)
                if (len(statelist[root].sons) > 500):
                    break
                tmpbestreward = 1000000
            # bestreward = 10000000
            # drawGraph(p, e)

        root2, act = bestchild(root, 0.0, pickalpha, determine=True)
        root = root2
        take_action(p, e, elist, act)
        print(act)
        print(act, file=LOG)
        print(act, file=LOG_result)
        print(bestreward, tmpbestreward, file=LOG_result)

        alpha = alpha + rate
        # c = c * rate2
        if (opt == 0):
            if (len(p) == maxp):
                opt = 1
            continue
        if (opt == 1):
            if (act.u == -1):
                opt = 2
            continue
        if (opt == 2):
            eidnow = eidnow + 1
            if (eidnow >= len(e)):
                break
    # drawGraph(p, e)


def main():
    global gpload
    global arealist
    global maxp
    global minx
    global maxx
    global miny
    global maxy
    global plist
    global arealist
    global minarea
    global maxarea
    p, e, pload = readFile()

    minx = 0.0
    miny = 0.0
    maxx = 18.288
    maxy = 9.144
    plist = []
    arealist = []

    if (usePlist):
        sepx = (maxx - minx) / 16
        sepy = (maxy - miny) / 8

        for i in range(17):
            for j in range(9):
                plist.append(Vector3(minx + i * sepx, miny + j * sepy, 0.0))

    separea = 0.0005
    for i in range(100):
        if (minarea + separea * i > maxarea + 1e-7):
            break
        arealist.append(minarea + separea * i)

    gpload = copy.deepcopy(pload)

    # reward_fun(p, e, gpload)
    UCTSearch(p, e)

    # print(reward_fun(pbest, ebest, gpload))
    print("bestreward=", bestreward)
    print(reward_fun(pbest, ebest, gpload))
    #
    print("bestmass=", bestMass)

    inputname = FILENAME.replace(".txt", "_")

    OUTFILE = LOGFOLDER + "reward_" + str(len(pbest)) + "p_" + inputname + str(round(bestreward, 2)) + ".txt"

    # OUTFILE = "./result/" + time_str + "_25bars_" + str(round(bestreward, 2)) + ".txt"
    # OUTFILE = ".\\" + str(len(pbest)) + "p_case1_" + str(round(bestreward, 2)) + ".txt"
    with open(OUTFILE, "w") as f:
        print(len(pbest), len(ebest), file=f)
        for i in range(len(pbest)):
            print(pbest[i].vec.x, pbest[i].vec.y, pbest[i].vec.z, pbest[i].supportX, pbest[i].supportY,
                  pbest[i].supportZ, pbest[i].loadX, pbest[i].loadY, pbest[i].loadZ, file=f)

        for i in range(len(ebest)):
            print(ebest[i].u, ebest[i].v, ebest[i].area, file=f)

        print("c = ", c, file=f)
        print("final alpha = ", alpha, file=f)
        print("rate = ", rate, file=f)
        print("pickalpha = ", pickalpha, file=f)
        if (useMaxlen):
            print("maxlen = ", maxlen, file=f)
        print("sgm1 = ", sgm1, file=f)
        print("sgm2 = ", sgm2, file=f)
        print("minarea = ", minarea, file=f)
        print("maxarea = ", maxarea, file=f)
        print("maxrandomnum = ", maxnum, file=f)
        print("minangle = ", minang, file=f)
        print("initson = ", initson, file=f)
        print("baseiter = ", baseiter, file=f)
        print("bestreward = ", bestreward, file=f)

    if USE_SOFT_PENALTY:
        MASS_OUTFILE = LOGFOLDER + "Mass_" + str(len(mass_pbest)) + "p_" + inputname + str(round(bestMass, 2)) + ".txt"

        # OUTFILE = "./result/" + time_str + "_25bars_" + str(round(bestreward, 2)) + ".txt"
        # OUTFILE = ".\\" + str(len(pbest)) + "p_case1_" + str(round(bestreward, 2)) + ".txt"
        with open(MASS_OUTFILE, "w") as mass_f:
            print(len(mass_pbest), len(mass_ebest), file=mass_f)
            for i in range(len(mass_pbest)):
                print(mass_pbest[i].vec.x, mass_pbest[i].vec.y, mass_pbest[i].vec.z, mass_pbest[i].supportX,
                      mass_pbest[i].supportY,
                      mass_pbest[i].supportZ, mass_pbest[i].loadX, mass_pbest[i].loadY, mass_pbest[i].loadZ,
                      file=mass_f)

            for i in range(len(mass_ebest)):
                print(mass_ebest[i].u, mass_ebest[i].v, mass_ebest[i].area, file=mass_f)

    while (len(MultiList) > 0):
        tup = heapq.heappop(MultiList)
        tup_OUTFILE = MULTIFOLDER + str(round(- tup[0])) + '.txt'
        p = tup[2]
        e = tup[3]
        with open(tup_OUTFILE, "w") as f:
            print(len(p), len(e), file=f)
            for i in range(len(p)):
                print(p[i].vec.x, p[i].vec.y, p[i].vec.z, p[i].supportX, p[i].supportY,
                      p[i].supportZ, p[i].loadX, p[i].loadY, p[i].loadZ, file=f)

            for i in range(len(e)):
                print(e[i].u, e[i].v, e[i].area, file=f)

    while (len(Mass_DIVList) > 0):
        tup = heapq.heappop(Mass_DIVList)
        tup_OUTFILE = MASS_DIVFOLDER + str(round(- tup[0])) + '.txt'
        p = tup[2]
        e = tup[3]
        with open(tup_OUTFILE, "w") as f:
            print(len(p), len(e), file=f)
            for i in range(len(p)):
                print(p[i].vec.x, p[i].vec.y, p[i].vec.z, p[i].supportX, p[i].supportY,
                      p[i].supportZ, p[i].loadX, p[i].loadY, p[i].loadZ, file=f)

            for i in range(len(e)):
                print(e[i].u, e[i].v, e[i].area, file=f)

    while (len(Mass_MultiList) > 0):
        tup = heapq.heappop(Mass_MultiList)
        tup_OUTFILE = MASS_MULTIFOLDER + str(round(- tup[0])) + '.txt'
        p = tup[2]
        e = tup[3]
        with open(tup_OUTFILE, "w") as f:
            print(len(p), len(e), file=f)
            for i in range(len(p)):
                print(p[i].vec.x, p[i].vec.y, p[i].vec.z, p[i].supportX, p[i].supportY,
                      p[i].supportZ, p[i].loadX, p[i].loadY, p[i].loadZ, file=f)

            for i in range(len(e)):
                print(e[i].u, e[i].v, e[i].area, file=f)

    # print("*****************")
    # for i in range(len(pbest)):
    #	print(i, pbest[i].vec, pbest[i].isSupport)
    # for i in range(len(ebest)):
    #	print(i, ebest[i].u, ebest[i].v, ebest[i].area)

    saveGraph(pbest, ebest, bestreward=bestreward)
    saveGraph(mass_pbest, mass_ebest, bestreward=bestMass)


# drawGraph(pbest, ebest, reward = bestreward)
# print(reward_fun(pbest, ebest, gpload))


if __name__ == '__main__':
    main()


