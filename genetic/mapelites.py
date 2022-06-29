


def creat_sub_dim(map, cut):
    if map is None:
        if isinstance(cut, list):
            return [None for _ in cut]
        return [None for _ in range(cut)]
    for i in range(len(map)):
        map[i] = creat_sub_dim(map[i], cut)
    return map

def creat_map(lcut,  min, max):
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
    res = []
    for i in range(len(score)):
        coor = 1
        while coor < len(map_cut[i]) and score[i] > map_cut[i][coor]:
            coor += 1
        res.append(coor-1)
    return res



if __name__ == "__main__":
    r1,r_score, r2 = creat_map([2, 100, [.2, .5, .60, .80]], 0, 1)
    print(r1)
    print([len(r) for r in r2])
    print(r2)
    s = [.265, .42, .82]
    print(s)
    coor = coor_map(r2, s)
    print(coor)
    print([r2[i][coor[i]] for i in range(len(coor))])