from scipy.spatial import distance

class rainflow:
    def __init__(self):
        self.reset()

    def add(self, val):
        if isinstance(val, tuple):
            self.lpoint.append(val)
        elif isinstance(val, int) or \
                isinstance(val, float) or \
                isinstance(val, list):
            self.lpoint.append(tuple(val))
        else:
            raise "need tuple, int float or list value"
        while len(self.lpoint) >= 3:
            X = distance.euclidean(self.lpoint[-3], self.lpoint[-2])
            Y = distance.euclidean(self.lpoint[-2], self.lpoint[-1])
            if X >= Y:
                x = self.lpoint.pop(-1)
                y = self.lpoint.pop(-1)
                self.lcycle.append([x, y])
                continue
            break
        return self.lcycle

    def reset(self):
        self.lpoint = []
        self.lcycle = []
