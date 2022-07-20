from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from collections import defaultdict
import os

class ResetCallback(BaseCallback):
    def __init__(self, lreset):
        super().__init__()
        self.lreset = lreset

    def _on_step(self) -> bool:
        if all(self.locals["dones"]):
            for elt in self.lreset:
                elt.reset()
        return True

class BestCallback(BaseCallback):
    def __init__(self, valid_env, dict_env, patience, dirname, parent_path, save_all=True, verbose=True):
        super().__init__()
        self.bestScores = defaultdict(list)
        self.allScores = defaultdict(list)
        self.dict_env = dict_env
        self.env_valid = valid_env
        self.bestValidationScoreSoFar = None
        self.cycle = 0
        self.patience = patience
        self.dirname = dirname
        self.parent_path = parent_path
        self.data = None
        self.name = ""
        self.save_all = save_all
        self.verbose = verbose
    
        
    def _on_step(self) -> bool:
        if not all(self.locals["dones"]):
            return True
        self.cycle += 1
        
        """for k in self.dict_env.keys():
            mean_reward, std_reward = evaluate_policy(self.model, self.dict_env[k], n_eval_episodes=1)
            self.validationScores[k].append(mean_reward)
        mean_reward, std_reward = evaluate_policy(self.model, self.training_env, n_eval_episodes=1)
        self.trainScores.append(mean_reward)
        mean_reward, std_reward = evaluate_policy(self.model, self.env_valid, n_eval_episodes=1)
        self.validationScores["default"].append(mean_reward)
        """

        for k in self.dict_env.keys():
            mean_reward, std_reward = evaluate_policy(self.model, self.dict_env[k], n_eval_episodes=1)
            self.allScores[k].append(mean_reward)

        mean_reward, std_reward = evaluate_policy(self.model, self.training_env, n_eval_episodes=1)
        self.allScores["train"].append(mean_reward)
        mean_reward, std_reward = evaluate_policy(self.model, self.env_valid, n_eval_episodes=1)
        self.allScores["validation"].append(mean_reward)
        if self.verbose:
            print("score", self.dirname + "_score:" + str(self.allScores["validation"][-1]))
            print("train score:" + str(self.allScores["train"][-1]))

        # part best
        if self.bestValidationScoreSoFar is None or mean_reward > self.bestValidationScoreSoFar:
            if not self.save_all and self.name != "":
                os.remove(self.name)

            self.bestValidationScoreSoFar = mean_reward
            self.cycle = 0

            for k in self.dict_env.keys():
                self.bestScores[k].append(self.allScores[k][-1])

            self.bestScores["validation"].append(self.allScores["validation"][-1])
            self.bestScores["train"].append(self.allScores["train"][-1])

            if self.verbose:
                print("new best", self.dirname + "_score:" + str(self.bestScores["validation"][-1]))
                print("train score:" + str(self.bestScores["train"][-1]))
            self.name = self.parent_path + "/" + self.dirname + "/" + \
                        self.dirname + "_" + self.model.__class__.__name__ + \
                        "_score" + str(self.bestScores["validation"][-1])
            self.model.save(self.name)
            self.data = self.env_valid.get_data()[-2] #-1 empty because env is reset at the end



        if self.patience is not None and self.cycle >= self.patience:
            return False
        return True

    def get_best_name(self):
        return self.name

    def get_data(self):
        return self.data

    def get_score(self):
        return self.bestScores, self.allScores
    