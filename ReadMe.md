# Micro grid
This git is related to a study on reward function for reinforcement learning for a micro-grid.

# Files
## env
### final_env
The last environment used. it inherits from the gym Env class.
Our environment contains information stored in a dictionary. It contains everything that is in _init_dict:
- "flow_H2"  
- "flow_lithium" 
- "lack_energy" 
- "waste_energy"  
- "soc"  
- "buy_energy"  
- "sell_energy"

As you can see, all parameter names used must be in lower case and with a unique name if you want to modify the environment.
You will also have to normalize your values in _normalize_value method.

To add reward functions to your environment, you can use 

    add_reward(key, fn, coeff=1.)
The data is stored and retrievable from 
        
    get_data()

## genetic
This folder contains the useful files for the genetic algorithm and the Map-Elites.
### ga
Contains the functions for the genetic algorithm.
### mapelites
Contains the funtions for Map-Elites.

### test_me
This is the place where you can run reward function tests.
The places you have to modify are : 
- function in creat_lfn (Not _set_fn and _get_mean but below)
	 - exemple (lfn and lcut is created before)
	 
	  isNeg = True
	  min_val= -1
	  max_val = 0
	  
	  # function name must start with a capital letter and be unique
      lname.append("Name_exemple")  
      fn, cut = _set_fn(lambda x: -x["buy_energy"], lres_reset, isNeg, min_val, max_val, nb_cut)  
      #or 
      fn, cut = _set_fn(lambda x, f=my_fn: f(x), lres_reset, isNeg, min_val, max_val, nb_cut)
      lfn.append(fn)  
      lcut.append(cut)
- "values has initialized" part in  main loop
