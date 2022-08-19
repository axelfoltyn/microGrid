import numpy as np

def creat_sub_dim(map, cut):
    """
    old function (not use)
    :param map:
    :param cut:
    :return:
    """
    if map is None:
        if isinstance(cut, list):
            return [None for _ in cut]
        return [None for _ in range(cut)]
    for i in range(len(map)):
        map[i] = creat_sub_dim(map[i], cut)
    return map

def creat_map(lcut,  min, max):
    """
    old function (not use)
    :param lcut:
    :param min:
    :param max:
    :return:
    """
    res = None # la carte
    res_score = None
    res_cut = [] # comment est decoupe les coordonees
    for cut in lcut:
        res = creat_sub_dim(res, cut)
        res_score = creat_sub_dim(res_score, cut)
        if isinstance(cut, list):
            res_cut.append(cut)
        else:
            res_cut.append([min + i * (max-min)/cut for i in range(cut)])

    return res, res_score, res_cut

def coor_map(map_cut, score):
    """
    get the coordinate of the map from its score and the cut
    :param map_cut: cutting value lists of our map
    :param score: score for each dimension
    :return: the coordinate of the map
    """
    res = []
    for i in range(len(score)):
        coor = 1
        while coor < len(map_cut[i]) and score[i] > map_cut[i][coor]:
            coor += 1
        res.append(coor-1)
    return res


def insert_map(dict_map, dict_map_score, coor, ind, score, i=0):
    """
    old function (not use)
    :param dict_map:
    :param dict_map_score:
    :param coor:
    :param ind:
    :param score:
    :param i:
    :return:
    """
    tuple_coor = tuple(coor)
    diff = sum(score)
    if tuple_coor in dict_map:
        diff -= sum(dict_map_score[tuple_coor])
        if diff > 0:
            dict_map[tuple_coor] = ind.copy()
            dict_map_score[tuple_coor] = score.copy()
    else:
        dict_map[tuple_coor] = ind.copy()
        dict_map_score[tuple_coor] = score.copy()
    return diff

def get_d(kd_tree, coor, k=3):
    """
    :param kd_tree: kd_tree = KDTree(np.array(coordinate))
    :param k: k nearest neighbors
    :param coor: the coordinate of the map
    :return: distance
    """
    if k <= 1:
        return 0
    d, ind = kd_tree.query(np.array(coor), k)
    d = list(d)
    d.sort()
    return sum(d[:k+1])/k


if __name__ == "__main__":
    _, _, r2 = creat_map([2, 100, [.2, .5, .60, .80]], 0, 1)
    dict_map = dict()
    dict_map_s = dict()
    print(r2)
    s = [.265, .42, .82]
    print(s)
    coor = coor_map(r2, s)
    print(coor)
    print([r2[i][coor[i]] for i in range(len(coor))])
    print(insert_map(dict_map, dict_map_s, coor, s, s))

    print(r1)
    print(insert_map(dict_map, dict_map_s, coor, s, s))

    s2 = [.265, .42, .83]
    print(s2)
    coor2 = coor_map(r2, s2)
    print(coor2)
    print([r2[i][coor2[i]] for i in range(len(coor))])
    print(insert_map(dict_map, dict_map_s, coor2, s2, s2))
    print(r1)
    print(insert_map(dict_map, dict_map_s, coor2, s2, s2))