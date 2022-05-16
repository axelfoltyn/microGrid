
class Reward:
    def reset(self):
        pass

class ClientReward(Reward):
    def __init__(self):
        self.reset()
        self.f_insatifaction = lambda x: -x

    def fn(self, param):
        if param["lack_energy"] > 0:
            self.nb_blackout += 1
        else:
            self.nb_blackout = 0
        return self.f_insatifaction(self.nb_blackout)

    def set_fn(self, f):
        self.f_insatifaction = f

    def reset(self):
        self.nb_blackout = 0

class BlackoutReward(Reward):
    def __init__(self):
        self.reset()

    def fn(self, param):
        if param["lack_energy"] > 0:
            self.nb_blackout += 1
        return self.nb_blackout

    def reset(self):
        self.nb_blackout = 0

class DODReward(Reward):
    def __init__(self, rainflow):
        self.rainflow = rainflow
        self.reset()
        self.f_dod = lambda x: -x

    def fn(self, param):
        pass

    def set_fn(self, f):
        self.f_dod = f

    def reset(self):
        self.rainflow.reset()
