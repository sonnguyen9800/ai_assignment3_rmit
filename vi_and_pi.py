### MDP Value Iteration and Policy Iteration
### Acknowledgement: start-up codes were adapted with permission from Prof. Emma Brunskill of Stanford University
import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    #Initialize Uo(s_i) = 0 for all s_i
    value_function = np.zeros(nS)

    #Main Loop - Loop through all k
    while True:

        #Making a copy of the former U ( Uk(s_i-1) )
        prev_value_function = np.copy(value_function)

        #Loop through all possible state
        for state in range(nS):
            #get the policy of that state
            policy_initialize = policy[state]
            tem = 0
            #Calculate the value of policy to deep k

            for value in P[state][policy_initialize]:
                prob = value[0]
                next_state = value[1]
                reward = value[2]
                tem += prob * (reward + gamma * prev_value_function[next_state])
            value_function[state] = tem
        # Keep evaluating until stop_flag = False
        stop_flag = True
        for state in range(nS):
            #Check until all value_function[state] do not change much, then break the loop and
            # return the value
            if (abs(value_function[state] - prev_value_function[state]) > tol):
                stop_flag = False
                break
        if (stop_flag):
            break

    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):

    #ini new_policy to old policy (policy)
    new_policy = np.copy(policy)

    #loop though all state to find each optimal policy per each state
    for state in range(nS):
        q_star_action = np.zeros(nA)

        #loop through all possible action of each state
        for action in range(nA):
            q_star_action_tem = 0
            for value in P[state][action]:
                prob = value[0]
                next_state = value[1]
                reward = value[2]
                q_star_action_tem += prob * (reward + gamma * value_from_policy[next_state])
            #calculate q_star
            q_star_action[action] = q_star_action_tem
        #after each loop - get the optimal policy of each state
        new_policy[state] = np.argmax(q_star_action)

    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    policy = np.zeros(nS, dtype=int)

    #Main Loop
    while True:
        #Evaluation the policy first, - return an array of value_function that match to each state
        old_policy_value = policy_evaluation(P, nS, nA, policy, gamma, tol)
        #Extracted the optimal policy
        extracted_policy = policy_improvement(P, nS, nA, old_policy_value, policy, gamma)

        flag = True

        #Stop the loop if the policy is not changed
        for state in range(nS):
            if policy[state] != extracted_policy[state]:
                flag = False
                break

        if (flag):
            break
        #Update new Policy
        policy = extracted_policy

    value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
	Learn value function and policy by using value iteration method for a given
	gamma and environment.
	"""

    # ini values to 0
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    # Main Loop - keep interating until converge
    while True:
        #Make a copy of old value
        prev_value_function = np.copy(value_function)

        #Loop through all state
        for state in range(nS):
            #initialize q_star
            q_sa = np.zeros(nA)

            #Loop through all action, calculate q_star for each action
            for action in range(nA):
                q_sa_tem = 0
                for value in P[state][action]:
                    prob = value[0]
                    next_state = value[1]
                    reward = value[2]
                    q_sa_tem += prob*(reward + gamma*prev_value_function[next_state])
                q_sa[action] = q_sa_tem

            #Find the v* based on q*
            value_function[state] = max(q_sa)
            #Get the optimal policy
            policy[state] = np.argmax(q_sa)


        # Break the loop when all states are converged
        flag = True
        for state in range(nS):
            # print "Result: {0:.3f}".format(abs(value_function[state] - prev_value_function[state]))
            if (abs(value_function[state] - prev_value_function[state]) > tol):
                flag = False
                break

        if (flag):
            break

    return value_function, policy


def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render();
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below

#1 2 1 0 1 0 1 0 2 1 1 0 0 2 2 0
if __name__ == "__main__":
    # comment/uncomment these lines to switch between deterministic/stochastic environments
    #env = gym.make("Deterministic-4x4-FrozenLake-v0")
    env = gym.make("Stochastic-4x4-FrozenLake-v0")
    #
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)
