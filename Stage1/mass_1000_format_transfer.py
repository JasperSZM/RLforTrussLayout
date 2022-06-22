import time
import json
import numpy as np
import math
import random
import copy
import shutil

import matplotlib.pyplot as plt
import warnings
import os, sys, contextlib
import openseespy.opensees as op

import heapq

for maxp in range(6, 10):
    print(maxp)

    OrgFolder = './PostResults/max' + str(maxp) + 'p/'
    TarFolder = './PostResults/max' + str(maxp) + 'p_Mass1000/'
    if not os.path.exists(TarFolder):
        os.mkdir(TarFolder)

    permutation = [0, 1, 2, 3]

    #if not os.path.exists(TarFolder):
    #    os.mkdir(TarFolder)

    files = os.listdir(OrgFolder)

    #print(len(selected_files))
    for file in files:
        FILENAME = OrgFolder + file
        #SAVENAME = TarFolder + file
        #SAVENAME = SAVENAME[:-4]
        #SAVENAME += folder_name
        #SAVENAME += '.txt'
        if file[-4:] != '.txt':
            continue
        with open(FILENAME, "r") as fle:
            lines = fle.readlines()
            for i in range(len(lines)):
                line = lines[i]
                vec = line.strip().split(' ')

                if (i == 0):
                    vn = int(vec[0])
                    en = int(vec[1])
                    Edges = [[- 1.0 for _ in range(vn)] for _ in range(vn)]
                    Nodes = []
                    nodes_position = []
                    continue

                if (1 <= i and i <= vn):
                    Nodes.append(line)
                    nodes_position.append([vec[0], vec[1]])
                    continue

                if (vn + 1 <= i and i <= vn + en):
                    node1 = int(vec[0])
                    node2 = int(vec[1])
                    Edges[node1][node2] = vec[2]
                    Edges[node2][node1] = vec[2]

        mass = 0
        pho = 2.76799 * 10 ** 3
        for v_i in range(vn):
            for v_j in range(vn):
                if v_i < v_j:
                    i_x = float(nodes_position[v_i][0])
                    i_y = float(nodes_position[v_i][1])
                    j_x = float(nodes_position[v_j][0])
                    j_y = float(nodes_position[v_j][1])
                    area = float(Edges[v_i][v_j])
                    if area == -1:
                        continue
                    mass += math.sqrt((i_x - j_x) * (i_x - j_x) + (i_y - j_y) * (i_y - j_y)) * area * pho


        SAVENAME = TarFolder + str(round(mass * 1000)).zfill(7) + '.txt'
        #print(mass, SAVENAME)


        shutil.copyfile(FILENAME, SAVENAME)