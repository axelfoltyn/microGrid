
class Reward:
    def fn(self, param):
        """
        the reward function
        :param param: a dict with each value used for reward (see _init_dict in final_env)
        :return:
        """
        pass

    def set_fn(self, f):
        """
        set the new reward function
        :param f: the new reward function
        :return: NoneType
        """
        print("not need function")

    def reset(self):
        """
        reset value for the reward function
        :return: NoneType
        """
        pass

class ClientReward(Reward):
    """
    reward related to the dissatisfaction of the customer.
    It's calculated by function  as parameter,
    the counting the number of blackout since the last time there was none.
    """
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
    """

    """
    def __init__(self):
        self.reset()

    def fn(self, param):
        if param["lack_energy"] > 0:
            self.nb_blackout += 1
        return -self.nb_blackout

    def reset(self):
        self.nb_blackout = 0

class CountBuyReward(Reward):
    def __init__(self):
        self.reset()

    def fn(self, param):
        if param["buy_energy"] > 0:
            self.nb_blackout += param["buy_energy"]
        return -self.nb_blackout

    def reset(self):
        self.nb_blackout = 0

class DODReward(Reward):
    def __init__(self, rainflow):
        self.rainflow = rainflow
        self.reset()
        self.f_dod = lambda x: -x

    def fn(self, param):
        return self.f_dod(len(self.rainflow.add(param["soc"])))

    def set_fn(self, f):
        self.f_dod = f

    def reset(self):
        self.rainflow.reset()

class DOD2Reward(Reward):

    def fn(self, param):
        if param["flow_lithium"] != 0:
            self.count += 1
        else:
            self.count = 0
        return self.count

    def reset(self):
        self.count = 0

class Client2Reward(Reward):

    def fn(self, param):
        if param["lack_energy"] > 0:
            return (1 - param["soc"] + param["sell_energy"]) ** 2
        return -(1 - param["soc"] + param["buy_energy"]) ** 2
