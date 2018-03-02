#Choose the action based on the policy and belief
#Performs search of Tree returning best action from top node
#Can return tree info as second paramter
function action_info(p::POMCPPlanner, b; tree_in_info=false)
    local a::action_type(p.problem)
    info = Dict{Symbol, Any}()
    try
        tree = POMCPTree(p.problem, p.solver.tree_queries)
        a = search(p, b, tree, info)
        p._tree = Nullable(tree)
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
    catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        a = convert(action_type(p.problem), default_action(p.solver.default_action, p.problem, b, ex))
        info[:exception] = ex
    end
    return a, info
end

#Generic action selection based on policy and current belief
#returns best action, after doing search of tree (first is because could return tree info as second parameter)
action(p::POMCPPlanner, b) = first(action_info(p, b))

#Takes a policy and searches the tree based on initial belief
#Returns best action
#This calls simulate() for each node open to search in the tree.
function search(p::POMCPPlanner, b, t::POMCPTree, info::Dict)
    all_terminal = true
    i = 0
    start_us = CPUtime_us()
    #Cycle for defined number of loops of sample+simulate to update policy
    for i in 1:p.solver.tree_queries
        #Check if search time has expired
        if CPUtime_us() - start_us >= 1e6*p.solver.max_time
            break
        end
        s = rand(p.rng, b) #SAMPLE State from belief <-----------!!!!!!!!!!!!!!!!!!!!!
        #Check for terminal state in the problem;
        if !POMDPs.isterminal(p.problem, s)
            #Not terminal, so simulate the random state selected, and add to tree;
            simulate(p, s, POMCPObsNode(t, 1), p.solver.max_depth)
            all_terminal = false
        end
    end
    #Record search information
    info[:search_time_us] = CPUtime_us() - start_us
    info[:tree_queries] = i

    #Account for nothing to search.
    if all_terminal
        throw(AllSamplesTerminal(b))
    end

    #Search the roots children nodes (actions) for the best value
    h = 1
    best_node = first(t.children[h])
    best_v = t.v[best_node]
    @assert !isnan(best_v)
    for node in t.children[h][2:end]
        if t.v[node] >= best_v
            best_v = t.v[node]
            best_node = node
        end
    end

    #Return best value's action
    return t.a_labels[best_node]
end

solve(solver::POMCPSolver, pomdp::POMDP) = POMCPPlanner(solver, pomdp)

#I ADDED
#struct Obs_Counts{S,A,O}
#    M::Dict{Tuple{S,A,O}, Int}
#end

#struct Trans_Counts{S,A}
#    M::Dict{Tuple{S,A,S}, Int}
#end

#Get the observation probability
#Base BAPOMDP state
mutable struct BAPOMDPState{S}
    s::S                        #State
    oc::Vector{Int}             #Multi dimensional array: Array{Int}(S,A,O)
    tc::Vector{Int}             #Multi dimensional array: Array{Int}(S,A,S)
end
#TODO: Improve to handle continuous, or infinite number of states
#    oc::Dict{Tuple{S,A,O},Int}  #Observation counts
#    tc::Dict{Tuple{S,A,S},Int}  #Transition counts


function obs_count_prob(p::POMDP,s::BAPOMDPState,a,o)
    s_idx = find(x -> x==s.s,states(p))
    a_idx = find(x -> x==a,actions(p))
    o_idx = find(x -> x==o,observations(p))
    return first(s.oc[s_idx,a_idx,o_idx]/sum(s.oc[s_idx,a_idx,:])) #convert to scalar
end

#Get the transition probability
function trans_count_prob(p::POMDP,s::BAPOMDPState,a,sp::BAPOMDPState)
    s_idx = find(x -> x==s.s,states(p))
    a_idx = find(x -> x==a,actions(p))
    sp_idx = find(x -> x==sp.s,states(p))
    return first(s.tc[s_idx,a_idx,sp_idx]/sum(s.tc[s_idx,a_idx,:])) #convert to scalar
end

#Get the distribution of transition probabilities
function trans_count_dist(p::POMDP,s::BAPOMDPState,a)
    s_idx = find(x -> x==s.s,states(p))
    a_idx = find(x -> x==a,actions(p))
    return reshape(s.tc[s_idx,a_idx,:]/sum(s.tc[s_idx,a_idx,:]),(n_states(p)))
end

#Initialize the BAPOMDP state
function initiate_state(p::POMDP,s)
    tc = ones(Int,POMDPs.n_states(p),POMDPs.n_actions(p),POMDPs.n_states(p))
    oc = ones(Int,POMDPs.n_states(p),POMDPs.n_actions(p),POMDPs.n_observations(p))
    return BAPOMDPState{state_type(p)}(s,oc,tc)
end

#Copy BAPOMDP state
function copy(s::BAPOMDPState)
    return BAPOMDPState(s.s,Base.copy(s.oc),Base.copy(s.tc))
end

#Increment the counts for the new sp
function increment_trans_obs_counts(p::POMDP,s::BAPOMDPState,a,o,sp::BAPOMDPState)
    s_idx = find(x -> x==s.s,states(p))
    a_idx = find(x -> x==a,actions(p))
    sp_idx = find(x -> x==sp.s,states(p))
    o_idx = find(x -> x==o,observations(p))
    sp.tc[s_idx,a_idx,sp_idx] += 1 #only can do transition at this time, observation will be updated later
    sp.oc[s_idx,a_idx,o_idx] += 1 #only can do transition at this time, observation will be updated later
end


#Rollout probably needs the current counts to do the updater()
#Actually do a generate sp from s and action for particle filter
# See https://github.com/JuliaPOMDP/ParticleFilters.jl
#See the SimpleParticleFilter update function calls these two functions
#https://github.com/JuliaPOMDP/ParticleFilters.jl/blob/master/src/ParticleFilters.jl
#Add all probability to the obs_weight, and the generate_s should just be a random state
function ParticleFilters.generate_s(model::POMDP,s::BAPOMDPState,a,rng::AbstractRNG)
    sp = copy(s)
    s_index = rand(Categorical(trans_count_dist(model,s,a))) #Handle states that are not integers
    sp.s = states(model)[s_index]
    return sp
end

#OUtput of this is put into a WeightedParticleBelief, paired with the output of the generate_s() above
#only just the probability of sp + o, given a + s; P(s',o | a,s)
#NOTE: Modifying the sp counts.
function ParticleFilters.obs_weight(model::POMDP,a,s::BAPOMDPState,sp::BAPOMDPState,o)
    increment_trans_obs_counts(model,s,a,o,sp)
    return obs_count_prob(model,s,a,o) * trans_count_prob(model,s,a,sp) #merge the observation and transition probabilities together
end


#Also do a generate observation probability distribution from new state sp and action taken.
#Don't need, just need obs_weight; unless it is unweighted
#function ParticleFilters.observation(model::,a,sp)
#    obsDist = Array{Float}(n_obs(O)) #?????
#    return obsDist
#end
#Where is the transition and observation probabilities used??

#struct SA_Model{S}
#    s::Vector{Tuple{S,Trans_Counts,Obs_Counts}}
#end

#function generate_sa_model(t::Trans_Counts,o::Obs_Counts)
#end
#END ADDED

#Called per node visited, and with a specific state
#Return the roll-up value of the an observation reward....., plus node at each
function simulate(p::POMCPPlanner, s, hnode::POMCPObsNode, steps::Int)
    #Checks for final step depth to expand, or if state is terminal for the problem
    #Then end value accumulation
    if steps == 0 || isterminal(p.problem, s)
        return 0.0
    end

    t = hnode.tree
    h = hnode.node

    #Get log of total number of times visited this observation node (h)
    ltn = log(t.total_n[h])

    #Keep track of the equal value best nodes.
    best_nodes = empty!(p._best_node_mem)
    best_criterion_val = -Inf
    #Search each action node of this observation node for best criterion value
    for node in t.children[h]
        #get total number of times visited this action node
        n = t.n[node]

        #Decide if should absolutely pick this node, or pick it based on its value, or pick it based on exploration
        if n == 0 && ltn <= 0.0
            criterion_value = t.v[node]
        elseif n == 0 && t.v[node] == -Inf
            criterion_value = Inf
        else
            criterion_value = t.v[node] + p.solver.c*sqrt(ltn/n)
        end

        #See if better node value, or same node value
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion_value == best_criterion_val
            push!(best_nodes, node)
        end
    end
    #Sample the best node (if multiple of them with same value) randomly
    ha = rand(p.rng, best_nodes)
    a = t.a_labels[ha]

    #Given the best action a and the sampled state in the belief state, then get the s', o, r
    sp, o, r = generate_sor(p.problem, s, a, p.rng)

    #Look-up the observation node under the current action node, else return 0
    hao = get(t.o_lookup, (ha, o), 0)
    if hao == 0
        #No observation node under the action node, so add new observation node
        hao = insert_obs_node!(t, p.problem, ha, o)
        #Estimate the value based on a state and observation node.
        #This is where a random rollout is performed to determine value, with the current policy
        #Dont think I need to add BA information in the rollout estimation
        v = estimate_value(p.solved_estimator,
                           p.problem,
                           sp,
                           POMCPObsNode(t, hao),
                           steps-1)
        R = r + discount(p.problem)*v
    else
        #Has observation node under the action node, so
        R = r + discount(p.problem)*simulate(p, sp, POMCPObsNode(t, hao), steps-1)
    end

    #Increment number of times visited for observation and best action node picked.
    t.total_n[h] += 1
    t.n[ha] += 1

    #Add the current reward, plus rollout of a new observation node, or the value of the next best action after this observation node
    # minus the current value, divided by total times visited.
    t.v[ha] += (R-t.v[ha])/t.n[ha]

    #Return future reward + current reward
    return R
end
