import random

import numpy as np


class Line:
    """Define line"""

    def __init__(self, p0, p1):
        self.p = np.array(p0)
        self.dir = np.array(p1) - np.array(p0)
        self.dist = np.linalg.norm(self.dir)
        self.dirn = self.dir / self.dist  # normalize

    def path(self, t):
        return self.p + t * self.dirn


def Intersection(line, center, radius):
    """Check line-sphere (circle) intersection"""
    a = np.dot(line.dirn, line.dirn)
    b = 2 * np.dot(line.dirn, line.p - center)
    c = np.dot(line.p - center, line.p - center) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
        return False

    return True


def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def isInObstacle(vex, obstacles, radius):
    for obs in obstacles:
        if distance(obs, vex) < radius:
            return True
    return False


def isThruObstacle(line, obstacles, radius):
    for obs in obstacles:
        if Intersection(line, obs, radius):
            return True
    return False


def nearest(G, vex, obstacles, radius):
    Nvex = None
    Nidx = None
    minDist = float("inf")

    for idx, v in enumerate(G.vertices):
        line = Line(v, vex)
        if isThruObstacle(line, obstacles, radius):
            continue

        dist = distance(v, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx


def newVertex(randvex, nearvex, stepSize):
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(stepSize, length)

    newvex = (int(nearvex[0] + dirn[0]), int(nearvex[1] + dirn[1]))
    return newvex


class Graph:
    """Define graph"""

    def __init__(self, startpos, endpos=None):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos: 0}
        self.neighbors = {0: []}
        self.distances = {0: 0.0}

    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except KeyError:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))


def RRT(startpos, obstacles, n_iter, radius, stepSize, area_list, endpos=None):
    """RRT algorithm"""
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = random.choice(area_list)
        if isInObstacle(randvex, obstacles, radius):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)

        if G.endpos:
            dist = distance(newvex, G.endpos)
            if dist < 2 * radius:
                endidx = G.add_vex(G.endpos)
                G.add_edge(newidx, endidx, dist)
                G.success = True

    print("RRT graph generated")
    return G
