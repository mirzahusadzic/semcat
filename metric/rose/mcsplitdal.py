# Algorithm 1: MCS(Ev, πv, πw, curSol, MaxSol)
# Input: a domain set Ev; policies πv and πw for selecting
# the matching pair (v, w); the current solution curSol and
# the best solution found so far MaxSol
# Output: MaxSol

def MCS(Ev, πv, πw, curSol, MaxSol):
  # Calculate the upper bound of the current solution
  UB = len(curSol) + sum(min(len(V_ip), len(V_it)) for (V_ip, V_it) in Ev)
  # If the upper bound is not larger than the best solution, prune and backtrack
  if UB <= len(MaxSol):
    return MaxSol
  # Select a domain (Vip, Vit) from Ev using selectD function
  (V_ip, V_it) = selectD(Ev)
  # Select a vertex v from Vip using policy πv
  v = selectV(V_ip, πv)
  # For each vertex w in Vit
  for w in V_it:
    # Remove w from Vit
    V_it.remove(w)
    # Add (v, w) to curSol
    curSol.append((v, w))
    # Update MaxSol if curSol is better
    if len(curSol) > len(MaxSol):
      MaxSol = curSol.copy()
    # Split the domains in Ev according to (v, w) and get a new domain set Ev0
    Ev0 = split(Ev, v, w)
    # Recursively call MCS on Ev0
    MaxSol = MCS(Ev0, πv, πw, curSol, MaxSol)
    # Remove (v, w) from curSol
    curSol.pop()
  # Remove v from Ev and get a new domain set Ev0
  Ev0 = remove(Ev, v)
  # Recursively call MCS on Ev0
  MaxSol = MCS(Ev0, πv, πw, curSol, MaxSol)
  # Return the optimal solution
  return MaxSol

# selectD function: Select a domain (Vip, Vit) from Ev
def selectD(Ev):
  # Define the size of a domain (Vip, Vit) to be max(|Vip|, |Vit|)[^1^][1]
  # Return the domain with the smallest size from Ev
  # Ties are broken by the largest vertex degree in Vip
  smallest_size = float('inf')
  selected_domain = None
  for (Vip, Vit) in Ev:
    size = max(len(Vip), len(Vit))
    if size < smallest_size:
      smallest_size = size
      selected_domain = (Vip, Vit)
    elif size == smallest_size:
      if max_degree(Vip) > max_degree(selected_domain[0]):
        selected_domain = (Vip, Vit)
  return selected_domain

# Helper function: Calculate the maximum degree of a vertex set
def max_degree(vertex_set):
  # Here we assume that the degree of a vertex can be obtained by calling degree(v)
  # In practice, you might need to implement this function based on your specific graph structure
  return max(degree(v) for v in vertex_set)

# Equation 1: R(v, w) = ∑(Vip,Vit)∈Ev min(|Vip|, |Vit|) - ∑(V0ip,V0it)∈Ev0 min(|V0ip|, |V0it|) + |Ev0|
# Input: a domain set Ev; a new domain set Ev0 obtained by matching (v, w)
# Output: the reward of matching (v, w)

def reward(v, w, Ev, Ev0):
  # Initialize the reward to zero
  reward = 0
  # For each domain (Vip, Vit) in Ev
  for (Vip, Vit) in Ev:
    # Add the minimum size of Vip and Vit to the reward
    reward += min(len(Vip), len(Vit))
  # For each domain (V0ip, V0it) in Ev0
  for (V0ip, V0it) in Ev0:
    # Subtract the minimum size of V0ip and V0it from the reward
    reward -= min(len(V0ip), len(V0it))
  # Add the number of domains in Ev0 to the reward
  reward += len(Ev0)
  # Return the reward
  return reward

# DAL(v) and DAL(v, w) are initially zero
DAL_v = {}
DAL_v_w = {}

# Update DAL(v) and DAL(v, w) after matching (v, w)[^1^][1]
def update_DAL(v, w, Ev, Ev0):
  # Calculate the reward of matching (v, w)
  rwd = reward(v, w, Ev, Ev0)
  # Update DAL(v)
  if v not in DAL_v:
    DAL_v[v] = 0
  DAL_v[v] += rwd
  # Update DAL(v, w)
  if (v, w) not in DAL_v_w:
    DAL_v_w[(v, w)] = 0
  DAL_v_w[(v, w)] += rwd

# Get DAL(v)
def get_DAL_v(v):
  if v not in DAL_v:
    # If v is not in DAL_v, recursively calculate it
    for (Vip, Vit) in Ev:
      if v in Vip:
        for w in Vit:
          Ev0 = split(Ev, v, w)
          update_DAL(v, w, Ev, Ev0)
          get_DAL_v(v)
  return DAL_v.get(v, 0)

# Get DAL(v, w)
def get_DAL_v_w(v, w):
  if (v, w) not in DAL_v_w:
    # If (v, w) is not in DAL_v_w, recursively calculate it
    Ev0 = split(Ev, v, w)
    update_DAL(v, w, Ev, Ev0)
    get_DAL_v_w(v, w)
  return DAL_v_w.get((v, w), 0)

# RL policy for vertex selection
# Input: a domain set Ev; a value function V; a reward function R
# Output: a vertex v from Ev

def RL_policy(Ev, V, R):
  # Initialize the best value and the best vertex to None
  best_value = None
  best_vertex = None
  # For each domain (Vip, Vit) in Ev
  for (Vip, Vit) in Ev:
    # For each vertex v in Vip
    for v in Vip:
      # Initialize the expected value of v to zero
      expected_value = 0
      # For each vertex w in Vit
      for w in Vit:
        # Calculate the reward of matching (v, w)
        reward = R(v, w, Ev)
        # Calculate the new domain set after matching (v, w)
        Ev0 = split(Ev, v, w)
        # Calculate the value of the new domain set using V
        value = V(Ev0)
        # Update the expected value of v by adding the reward and the value
        expected_value += reward + value
      # If the expected value of v is better than the best value so far
      if best_value is None or expected_value > best_value:
        # Update the best value and the best vertex to v
        best_value = expected_value
        best_vertex = v
  # Return the best vertex
  return best_vertex

# Initialize the number of applications of the current branching strategy
NbApp = 0
# Set the maximum number of applications of a branching strategy
MaxNbApp = 10
# Choose the initial branching strategy (RL or DAL)
current_strategy = RL_policy

def hybrid_branching(Ev, V, R):
  global NbApp, current_strategy
  # If NbApp reaches MaxNbApp, switch the branching strategy
  if NbApp >= MaxNbApp:
    if current_strategy == RL_policy:
      current_strategy = DAL_policy
    else:
      current_strategy = RL_policy
    NbApp = 0
  # Select a vertex using the current branching strategy
  v = current_strategy(Ev, V, R)
  # Increase NbApp
  NbApp += 1
  return v

# RL policy for vertex selection
# Input: a domain set Ev; a value function V; a reward function R
# Output: a vertex v from Ev

# RL policy for vertex selection
# Input: a domain set Ev; a value function V; a reward function R
# Output: a vertex v from Ev

def RL_policy(Ev, V, R):
  # Initialize the best value and the best vertex to None
  best_value = None
  best_vertex = None
  # For each domain (Vip, Vit) in Ev
  for (Vip, Vit) in Ev:
    # For each vertex v in Vip
    for v in Vip:
      # Initialize the expected value of v to zero
      expected_value = 0
      # For each vertex w in Vit
      for w in Vit:
        # Calculate the reward of matching (v, w)
        reward = R(v, w, Ev)
        # Calculate the new domain set after matching (v, w)
        Ev0 = split(Ev, v, w)
        # Calculate the value of the new domain set using V
        value = V(Ev0)
        # Update the expected value of v by adding the reward and the value
        expected_value += reward + value
      # If the expected value of v is better than the best value so far
      if best_value is None or expected_value > best_value:
        # Update the best value and the best vertex to v
        best_value = expected_value
        best_vertex = v
  # Return the best vertex
  return best_vertex

# selectD function: Select a domain (Vip, Vit) from Ev
def selectD(Ev):
  # Define the size of a domain (Vip, Vit) to be max(|Vip|, |Vit|)[^1^][1]
  # Return the domain with the smallest size from Ev
  # Ties are broken by the largest vertex degree in Vip
  smallest_size = float('inf')
  selected_domain = None
  for (Vip, Vit) in Ev:
    size = max(len(Vip), len(Vit))
    if size < smallest_size:
      smallest_size = size
      selected_domain = (Vip, Vit)
    elif size == smallest_size:
      if max_degree(Vip) > max_degree(selected_domain[0]):
        selected_domain = (Vip, Vit)
  return selected_domain

# Helper function: Calculate the maximum degree of a vertex set
def max_degree(vertex_set):
  # Here we assume that the degree of a vertex can be obtained by calling degree(v)
  # In practice, you might need to implement this function based on your specific graph structure
  return max(degree(v) for v in vertex_set)

# selectW function: Select a vertex w from Vit using policy πw
def selectW(Vit, πw):[^2^][2]
  # Here we simply select the first vertex in Vit
  # In practice, you might want to use a more sophisticated policy
  return Vit[0]

# DAL policy for vertex selection
# Input: a domain set Ev; a value function V; a reward function R
# Output: a vertex v from Ev

def DAL_policy_v(Ev, V, R):
  # Initialize the best value and the best vertex to None
  best_value = None
  best_vertex = None
  # For each domain (Vip, Vit) in Ev
  for (Vip, Vit) in Ev:
    # For each vertex v in Vip
    for v in Vip:
      # Initialize the expected value of v to zero
      expected_value = 0
      # For each vertex w in Vit
      for w in Vit:
        # Calculate the reward of matching (v, w)
        reward = R(v, w, Ev)
        # Calculate the new domain set after matching (v, w)
        Ev0 = split(Ev, v, w)
        # Calculate the value of the new domain set using V
        value = V(Ev0)
        # Update the expected value of v by adding the reward and the value
        expected_value += reward + value
      # If the expected value of v is better than the best value so far
      if best_value is None or expected_value > best_value:
        # Update the best value and the best vertex to v
        best_value = expected_value
        best_vertex = v
  # Return the best vertex
  return best_vertex

# DAL policy for vertex selection
# Input: a domain set Vit; a value function V; a reward function R; selected vertex v from Vip.
# Output: a vertex w from Vit

def DAL_policy_w(Vit, V, R, v):
  # Initialize the best value and the best vertex to None
  best_value = None
  best_vertex = None
  # For each vertex w in Vit
  for w in Vit:
    # Calculate the reward of matching (v, w)
    reward = R(v, w, Ev)
    # Calculate the new domain set after matching (v, w)
    Ev0 = split(Ev, v, w)
    # Calculate the value of the new domain set using V
    value = V(Ev0)
    # Update the expected value of w by adding the reward and the value
    expected_value = reward + value
    # If the expected value of w is better than the best value so far
    if best_value is None or expected_value > best_value:
      # Update the best value and the best vertex to w
      best_value = expected_value
      best_vertex = w
  # Return the best vertex
  return best_vertex

