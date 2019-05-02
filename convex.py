import numpy as np

# Using convex hull to extract the face
# References is this page https://algorithmtutor.com/Computational-Geometry/Convex-Hull-Algorithms-Jarvis-s-March/
# We are using the jarvis's march as described in this page.

# 0 --> p, q and r are colinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def get_orientation(origin, p1, p2):
    val = (p1[1] - origin[1]) * (p2[0] - p1[0]) - (p1[0] - origin[0]) * (p2[1] - p1[1])

    if val == 0:
        return 0
    return 1 if val > 0 else 2

def get_dist(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def get_hull(points):
    points = points.tolist()
    hull = []
    # We find the leftmost points by doing this.
    leftmost = points[0]
    for point in points:
        if point[0] < leftmost[0]:
            leftmost = point

    hull.append(leftmost)

    # 1 --> Clockwise
    # 2 --> Counterclockwise
    q = None
    l = leftmost
    while q is not leftmost:

        for point in points:
            if point is l:
                continue
            q = point
            break

        for i in points:
            ori = get_orientation(l, i, q)
            if ori == 1:
                q = i

            if ori == 0 and (get_dist(l, i) > get_dist(q, l)):
                q = i
        hull.append(q)
        l = q

    return np.asarray(hull).astype(int)

