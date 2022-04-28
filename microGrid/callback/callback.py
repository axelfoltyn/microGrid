from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from collections import defaultdict
    

class BestCallback(BaseCallback):
    def __init__(self, valid_env, dict_env, patience, dirname, parent_path):
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

        # part best
        if self.bestValidationScoreSoFar is None or mean_reward > self.bestValidationScoreSoFar:
            self.cycle = 0

            for k in self.dict_env.keys():
                mean_reward, std_reward = evaluate_policy(self.model, self.dict_env[k], n_eval_episodes=1)
                self.bestScores[k].append(mean_reward)

            mean_reward, std_reward = evaluate_policy(self.model, self.env_valid, n_eval_episodes=1)
            self.bestScores["validation"].append(mean_reward)
            mean_reward, std_reward = evaluate_policy(self.model, self.training_env, n_eval_episodes=1)
            self.bestScores["train"].append(mean_reward)

            self.bestValidationScoreSoFar = self.bestScores["validation"][-1]
            print("new best", self.dirname + "score:" + str(self.bestScores["validation"][-1]))
            print("train score:" + str(self.bestScores["train"][-1]))
            self.model.save(self.parent_path + "/" + self.dirname + "/" + self.dirname + "-" + self.model.__class__.__name__ +
                            "-score" + str(self.bestScores["validation"][-1]))
            self.data = self.env_valid.get_data()[-2] #-1 empty because env is reset at the end
        if self.cycle >= self.patience:
            return False
        return True

    def get_data(self):
        return self.data

    def get_score(self):
        return self.bestScores, self.allScores
    