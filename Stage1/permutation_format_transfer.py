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

for maxp in range(6, 10):
    print(maxp)
    #OrgFolder = './MaxPoints/MctsStage1_MaxPoints_' + str(int(maxp)) + '/'
    #TarFolder = './PermutationMaxPoints/MctsStage1_MaxPoints_' + str(int(maxp)) + '/'
    OrgFolder = './results/max' + str(maxp) + 'p/MASS_MULTI_Result/'
    TarFolder = './PostResults/max' + str(maxp) + 'p/'

    permutation = [3, 0, 2, 1]

    if not os.path.exists(TarFolder):
        os.mkdir(TarFolder)

    files = os.listdir(OrgFolder)

    for file in files:
        FILENAME = OrgFolder + file
        SAVENAME = TarFolder + file
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
                    continue

                if (1 <= i and i <= vn):
                    Nodes.append(line)
                    continue

                if (vn + 1 <= i and i <= vn + en):
                    node1 = int(vec[0])
                    node2 = int(vec[1])
                    Edges[node1][node2] = vec[2]
                    Edges[node2][node1] = vec[2]

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