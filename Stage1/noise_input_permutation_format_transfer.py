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

OrgF = './results/9pointsNoise/'
TarFolder = './PostResults/NewMixedPoints/New9pointsNoise/'
if not os.path.exists(TarFolder):
    os.mkdir(TarFolder)
folderlist = os.listdir(OrgF)
folder_count = 0
for folder_name in folderlist:
    if len(folder_name) != 3:
        continue
    folder_count += 1
    print(folder_count)
    #OrgFolder = './MaxPoints/MctsStage1_MaxPoints_' + str(int(maxp)) + '/'
    #TarFolder = './PermutationMaxPoints/MctsStage1_MaxPoints_' + str(int(maxp)) + '/'
    OrgFolder = OrgF + folder_name + '/MASS_MULTI_Result/'

    permutation = [3, 0, 2, 1]

    #if not os.path.exists(TarFolder):
    #    os.mkdir(TarFolder)

    files = os.listdir(OrgFolder)
    files.sort()
    selected_files = files[:5]
    max_idx = len(files)
    if max_idx <= 15:
        continue
    #print('max_idx', max_idx)
    for _ in range(10):
        print(max_idx)
        _idx = random.randint(5, max_idx-1)
        selected_files.append(files[_idx])
        #print(_idx)

    #print(len(selected_files))
    for file in selected_files:
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

        PermutationEdges = [[- 1.0 for _ in range(vn)] for _ in range(vn)]
        for i in range(vn):
            for j in range(vn):
                new_i = i
                new_j = j
                if i < len(permutation):
                    new_i = permutation[i]
                if j < len(permutation):
                    new_j = permutation[j]
                PermutationEdges[i][j] = Edges[new_i][new_j]
                PermutationEdges[j][i] = Edges[new_j][new_i]

        with open(SAVENAME, "w") as f:
            print(int(vn), int(vn * (vn - 1) / 2), file=f)
            for i in range(len(Nodes)):
                new_i = i
                if i < len(permutation):
                    new_i = permutation[i]
                print(Nodes[new_i], file=f, end='')
            for j in range(vn):
                for i in range(vn):
                    if i < j:
                        print(int(i), int(j), PermutationEdges[i][j], file=f)