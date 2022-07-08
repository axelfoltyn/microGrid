
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

"""def insert_map(map, map_score, coor, ind, score, i=0):
    if i == len(coor)-1:
        if map[coor[i]] is None:
            diff = sum(score)
            map[coor[i]] = ind
            map_score[coor[i]] = score
        elif sum(map_score[coor[i]]) < sum(score):
            diff = sum(score) - sum(map_score[coor[i]])
            map[coor[i]] = ind
            map_score[coor[i]] = score
        else:
            diff = sum(score) - sum(map_score[coor[i]])
        return diff
    return insert_map(map[coor[i]], map_score[coor[i]], coor, ind, score, i+1)"""

def insert_map(dict_map, dict_map_score, coor, ind, score, i=0):
    tuple_coor = tuple(coor)
    diff = sum(score)
    if tuple_coor in dict_map:
        diff -= sum(dict_map_score[tuple_coor])
        if sum(score) > sum(dict_map_score[tuple_coor]):
            dict_map[tuple_coor] = ind
            dict_map_score[tuple_coor] = score
    else:
        dict_map[tuple_coor] = ind
        dict_map_score[tuple_coor] = score
    return diff



if __name__ == "__main__":
    r1,r_score, r2 = creat_map([2, 100, [.2, .5, .60, .80]], 0, 1)
    dict_map = dict()
    dict_map_s = dict()
    #print(r1)
    #print([len(r) for r in r2])
    print(r2)
    s = [.265, .42, .82]
    print(s)
    coor = coor_map(r2, s)
    print(coor)
    print([r2[i][coor[i]] for i in range(len(coor))])
    print(insert_map(r1, r_score, coor, s, s))
    print(insert_map2(dict_map, dict_map_s, coor, s, s))

    print(r1)
    print(insert_map(r1, r_score, coor, s, s))
    print(insert_map2(dict_map, dict_map_s, coor, s, s))

    s2 = [.265, .42, .83]
    print(s2)
    coor2 = coor_map(r2, s2)
    print(coor2)
    print([r2[i][coor2[i]] for i in range(len(coor))])
    print(insert_map(r1, r_score, coor2, s2, s2))
    print(insert_map2(dict_map, dict_map_s, coor2, s2, s2))
    print(r1)
    print(insert_map(r1, r_score, coor, s2, s2))
    print(insert_map2(dict_map, dict_map_s, coor2, s2, s2))