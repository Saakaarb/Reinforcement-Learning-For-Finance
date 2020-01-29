# Reinforcement-Learning-For-Finance
Student Repository for CME241 at Stanford University

The repository contains HW codes and Latex files(to be added), corresponding to assigned homeworks. Shown below is a list of files, with their purpose

* mdp.py:
  * Contains functions pertaining to creating and solving an mdp based on user input. User input is in dictionary format.
  * Functions with their description:
    * evaluate_policy(): For any policy, evaluate the value function
    * improve_policy(): Find the immediate greedy policy, leading to policy improvement
    * policy iteration(): Combines the above two to form a policy iteration algorithm (Currently prints best policy for a state)
    * create_mdp(): Given lists and dictionaries of relevant parameters of a sparse problem, formulates the MDP matrices 
* mrp.py:
  * Contains functions pertaining to creating and solving an mrp based on user input. User input is in dictionary format.
    * create_MRP_rs_sd(): Creates relevant MRP matrices, with reward in R(s,s') form
    * create_mrp_rs(): Creates relevant MRP matrices, with reward in R(s) form
    * change_reward_def(): Changes from R(s,s') to R(s) form of MRP
    * mdp_to_mrp(): Converts an MDP to an MRP given a policy(Outputs the transformed matrices)
    * solve_mrp(): Solves an MRP using the matrix method
* simulate_gridworld.py:
  * The file has the input dictionaries for the gridworl with random policy problem discussed in the class. MDP functions are tested on this.
* simulate_studentmrp.py:
  * The file has inputs for the student MDP problem dicussed in class with gamm=0.9. MRP functions are tested on this.

Written below is a corrpespondance of date of assigned homeworks, and the files it is contained in:

Jan 10: simulate_mrp.py, mrp.py,mdp.py  
Jan 15,17: simulate_mdp.py, mdp.py,

simulate_* files are test cases created to test the scripts. Most are from classroom examples.


