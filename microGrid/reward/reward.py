class ClientReward:
    def __init__(self):
        self.nb_blackout = 0
        self.f_insatifaction = lambda x: -x

    def fn(self, param):
        if param["lack_energy"] > 0:
            self.nb_blackout += 1
        else:
            self.nb_blackout = 0
        return self.f_insatifaction(self.nb_blackout)

    def set_fn(self, f):
        self.f_insatifaction = f