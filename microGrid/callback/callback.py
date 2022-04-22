from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class BestCallback(BaseCallback):
    def __init__(self, env_valid, patience, dirname, parent_path):
        super().__init__()
        self.validationScores = []
        self.trainScores = []
        self.env_valid = env_valid
        self.bestValidationScoreSoFar = None
        self.cycle = 0
        self.patience = patience
        self.dirname = dirname
        self.data = None
    
        
    def _on_step(self) -> bool:
        if not all(self.locals["dones"]):
            return True
        self.cycle += 1
        
        mean_reward, std_reward = evaluate_policy(self.model, self.env_valid, n_eval_episodes=1)
        self.validationScores.append(mean_reward)
        
        mean_reward, std_reward = evaluate_policy(self.model, self.training_env, n_eval_episodes=1)
        self.trainScores.append(mean_reward)
        
        # part best
        if self.bestValidationScoreSoFar is None or self.validationScores[-1] > self.bestValidationScoreSoFar:
            self.cycle = 0
            self.bestValidationScoreSoFar = self.validationScores[-1]
            print("new best", self.dirname + "score:"+ str(self.validationScores[-1]))
            self.model.save(self.dirname + "/" + self.dirname + 
                            "-score-" + str(self.validationScores[-1]))
            self.data = self.env_valid.get_data()[-2] #-1 empty because env is reset at the end
            #agent.dumpNetwork(self.dirname + "/" + self.dirname + "-score-" + str(validationScores[-1]), epoch)
        if self.cycle >= self.patience:
            return False
        return True
    def get_data(self):
        return self.data
    def get_score(self):
        return self.validationScores, self.trainScores