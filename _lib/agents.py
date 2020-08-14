import numpy as np
import _lib.pr_func as pr

class Agent(object):

    def __init__(self,actions,states,transition,V=np.array([])):
        # defined props
        self.actions = actions
        self.states = states
        self.transition = transition
        self.V = V                                                            # V = [V[S'=0],V[S'=1],...]
        # placeholder props
        self.observations = []
        self.Q = np.array([])

    def _model(self,s,a):
        s1,r = self.transition(s,a)
        return s1

    def _updateQ(self,s):
        next_states = [self._model(s,a) for a in self.actions]      # [S'(a=0),S'(a=1),...] (using model)
        self.Q = np.array([self.V[s1] for s1 in next_states])       # Q = [V[S'(a=0)],V[S'(a=1)], ...]

    def _policy(self,s):
        """
        s => a
        """
        self._updateQ(s)
        return np.argmax(self.Q) 

    def proc(self,obs): 
        s,r = obs
        self.observations.append(obs)
        return self._policy(s)

    def evaluate_episode(self):
        self.observations = []

    def get_Vmat(self,dim_states):
        Vmat = np.zeros(dim_states)
        for i in self.states:
            j,k = np.unravel_index(i,dim_states)
            Vmat[j,k] = self.V[i]
        return Vmat


class RandomAgent(Agent):

    def proc(self,observation):
        return np.random.choice(self.actions)


class FirstVisitMC(Agent):

    def __init__(self,actions,states,transition,gamma=1):
        Agent.__init__(self, actions, states, transition)
        self.gamma = gamma
        self.N = np.zeros(len(self.states))
        self.R = np.zeros(len(self.states))

    def _updateV(self):
        self.V = self.R/(self.N+1e-10)

    def _policy(self,s):
        self._updateV()
        self._updateQ(s)
        # need to randomize action selection because of unfinished value functions
        praw = np.exp(self.gamma*self.Q)                                   
        return np.random.choice(self.actions, p=praw/np.sum(praw))         

    def evaluate_episode(self):
        visited_states = [o[0] for o in self.observations]
        num_states_visited = len(visited_states)
        G = 0 # cumulative reward 
        for k in range(len(self.observations)):
            o = self.observations[::-1][k]
            G += o[1]
            s = o[0]
            if not(s in visited_states[0:num_states_visited-k-1]):
                self.R[s] += G
                self.N[s] += 1       
        self.observations = []


class ValueIteration(Agent):
    
    def __init__(self,actions,states,transition,iterations=100,gamma=.9):
        Agent.__init__(self, actions, states, transition)
        self.iterations = iterations
        self.gamma = gamma
        self.V = self._value_iteration()

    def _value_iteration(self):
        v0 = np.zeros(len(self.states))
        v1 = np.zeros(len(self.states))
        for iter in range(self.iterations):
            for s in self.states:
                vec = []
                for a in self.actions:
                    s1,r = self.transition(s,a)
                    vec.append(r+self.gamma*v0[s1])
                v1[s] = np.max(vec)
            v0 = v1
        return v1


class FreeEnergyIteration(Agent):

    def __init__(self,actions,states,transition,iterations=100,alpha=1.0,gamma=0.9):
        Agent.__init__(self, actions, states, transition)
        self.iterations = iterations
        self.alpha = alpha
        self.gamma = gamma
        self.prior = np.ones(len(self.actions))/len(self.actions)
        self.V = self._free_energy_iteration()

    def _free_energy_iteration(self):
        v0 = np.zeros(len(self.states))
        v1 = np.zeros(len(self.states))
        for iter in range(self.iterations):
            for s in self.states:
                vec = []
                for a in self.actions:
                    s1,r = self.transition(s,a)
                    vec.append(r+self.gamma*v0[s1])
                e_vec = np.exp(self.alpha*np.array(vec))
                v1[s] = np.log(np.sum(self.prior*e_vec))/self.alpha
            v0 = v1
        return v1

    def _policy(self,s):
        praw = []
        for a in self.actions:
            s1,r = self.transition(s,a)
            praw.append(np.exp(self.alpha*(r+self.gamma*self.V[s1])))
        praw = np.array(praw)
        return np.random.choice(self.actions, p = praw/np.sum(praw))


class ProbAgent(Agent):

    def __init__(self,env,vision=.1,maxSteps=6):
        Agent.__init__(self, env.actions, env.states, env.transition)
        self.env = env
        self.K, self.N = len(env.actions), len(env.states)
        self.T = maxSteps
        self.sigmas = self.get_sigmas(vision)
        ## transition 
        self.pTrans = self.get_transition(self.N,self.K,env.transition)   # s,a,s1 = ikj
        ### likelihood 
        self.likelihood = self.get_likelihood(env)   # s,x = jl  
        ## desired distribution 
        self.pDesired = self.get_pdes(self.N,env.goal_s)   # x = l
        ## variables that change in each timestep:
        self.x, self.a = [], []
        self.actionDists = []
        self.S,self.A,self.X,self.Tr,self.L,self.D = {},{},{},{},{},{}
        self.pS0, self.pA0, self.t = 0,0,0
        self._set_variables()

    def evaluate_episode(self):
        self.x = []
        self.a = []
        self.t = 0

    def normalize(self,p):
        r = range(0,len(np.shape(p)))
        r0 = range(0,len(np.shape(p))-1)
        return np.einsum(p,r,1.0/np.einsum(p,r,r0),r0,r)

    def get_sigmas(self,vision):
        if isinstance(vision,str):
            vision_str = vision[1:-1].split('\n')
            vis = [line.split(' ') for line in vision_str]
            sigmas = np.array([[float(el) for el in line] for line in vis])
        elif isinstance(vision,np.ndarray):
            sigmas = vision
        else:
            sigmas = vision*np.ones(np.shape(self.env.gridmap))    
        if not(np.shape(self.env.gridmap)==np.shape(sigmas)):
            print('Sigmas must have the same dimensions as the gridworld!')
        return sigmas

    def get_transition(self,N,K,transition):
        T = np.zeros((N,K,N)) # s,a,s1
        for s in range(N):
            for a in range(K):
                s1,r = transition(s,a)
                T[s,a,s1] = 1
        return T + 1e-30

    def get_likelihood(self,env):
        sDims = np.shape(env.gridmap)
        N = len(env.states)
        def gaussian2d(xx,yy,mus,sigma):
            vec0 = np.exp(-(xx-mus[0])**2/(2*sigma**2))
            vec1 = np.exp(-(yy-mus[1])**2/(2*sigma**2))    
            mat = np.zeros((len(xx),len(yy)))
            for i in range(len(xx)):
                for j in range(len(yy)):
                    mat[i,j] = vec0[i]*vec1[j]
            return mat/np.sum(mat)
        def flatten(B):
            Bflat = [B[tuple(env.state_num_to_tuple(num))] for num in range(N)]
            return np.array(Bflat)
        xx = np.arange(0,sDims[0])
        yy = np.arange(0,sDims[1])
        A = np.zeros((N,N))
        for s in range(N):
            i,j = env.state_num_to_tuple(s)
            A[s,:] = flatten(gaussian2d(xx,yy,[i,j],self.sigmas[i,j]))
        return A + 1e-30

    def get_pdes(self,N,goal):
        tmp = np.zeros(N) + 1e-30
        for g in goal:
            tmp[g] = 1
        return tmp/np.sum(tmp)

    def proc(self,obs): 
        """
        1. pass state through likelihood to create an observation and add to list
        2. get a new action from the policy and add to list
        3. increment time counter and check whether max steps are not exceeded
        """
        s,r = obs
        distorted_s = np.random.choice(self.env.states, p=self.likelihood[s,:])
        self.x.append(distorted_s)
        action = self._policy()
        self.a.append(action)
        self.t += 1
        if self.t >= self.T:        
            self.env.done = True
        return action

    def _policy(self):
        """
        1. calculate the current action distribution
        2. return a sample as new action 
        """
        qAt = self._get_action_dist()
        self.actionDists.append(qAt)
        return np.random.choice(self.env.actions,p=qAt)

    def _set_variables(self):
        """
        set up variables
        """
        S = {i : 'S{}'.format(i) for i in range(0,self.T+1)}
        A = {i : 'A{}'.format(i) for i in range(0,self.T)}
        X = {i : 'X{}'.format(i) for i in range(1,self.T+1)}
        pr.set_dims([(s,self.N) for s in S.values()]+[(a,self.K) for a in A.values()]+[(x,self.N) for x in X.values()])
        self.S,self.A,self.X = S,A,X

    def _set_fixed(self):  
        """
        for current time step t,
        put fixed quantities (likelihood, transition,...) 
        in pr_func instances    
        """
        S,A,X = self.S,self.A,self.X
        self.pS0 = pr.func(vars=['S0'],val='unif').normalize()            # unbiased belief of starting state
        self.pA0 = pr.func(vars=list(A.values()),val='unif').normalize()  # and action, i.e. these priors have no influence       
        self.Tr = {**{i : pr.func(vars=[S[i],S[i+1]],val=self.pTrans[:,self.a[i],:]) for i in range(0,self.t)},  # p0(S'|S,a) 
               **{i : pr.func(vars=[S[i],A[i],S[i+1]],val=self.pTrans) for i in range(self.t,self.T)}}           # p0(S'|S,A)  
        self.L = {**{i : pr.func(vars=[S[i]],val=self.likelihood[:,self.x[i]]) for i in range(0,self.t+1)},      # p0(x|S)
             **{i : pr.func(vars=[S[i],X[i]],val=self.likelihood) for i in range(self.t+1,self.T+1)}}            # p0(X|S)
        self.D = {i : pr.func(vars=[X[i]],val=self.pDesired) for i in range(self.t+1,self.T+1)}                  # pdes(X)
       
    def _get_action_dist(self):
        """
        dummy function: must be defined by each model
        """
        return np.ones(self.K)/self.K


class ActiveInference2016(ProbAgent):

    def __init__(self,env,vision=.1,maxSteps=6,iterations=100,alpha=1.0):
        ProbAgent.__init__(self, env, vision, maxSteps)
        self.iterations = iterations
        self.alpha = alpha

    def meanField(self):
        precision=1e-4
        N,K,T,S,A,X,t = self.N,self.K,self.T,self.S,self.A,self.X,self.t
        Tr,L,D,pS0,pA0 = self.Tr,self.L,self.D,self.pS0,self.pA0 
        ### initialize
        qA = pr.func(vars=list(A.values()), val='unif').normalize()
        qS = {**{i: pr.func(vars=[S[i]], val='unif').normalize([S[i]]) for i in range(0,t+1)},
              **{i: pr.func(vars=[S[i],*[A[j] for j in range(t,i)]],val='unif').normalize([S[i]]) for i in range(t+1,T+1)}}
        ## repeat:
        q0 = 0
        for iter in range(self.iterations):
            if t > 0:
                qS[0] = (L[0]*pS0*pr.exp_tr(pr.sum([S[1]],pr.log(Tr[0])*qS[1]))).normalize([S[0]])
                for tau in range(1,t):
                    qS[tau] = (L[tau]*pr.exp_tr(pr.sum([S[tau-1]],pr.log(Tr[tau-1])*qS[tau-1])) 
                                        +pr.sum([S[tau+1]],pr.log(Tr[tau])*qS[tau+1])).normalize([S[tau]])
                qS[t] = (L[t]*pr.exp_tr(pr.sum([S[t-1]],pr.log(Tr[t-1])*qS[t-1])) 
                                        +pr.sum([S[t+1],A[t]],pr.log(Tr[t])*qS[t+1]*pr.sum(qA,[A[t]]))).normalize([S[t]])
            else: 
                qS[0] = (L[0]*pS0*pr.exp_tr(pr.sum([S[1],A[0]],pr.log(Tr[0])*qS[1]*pr.sum(qA,[A[0]])))).normalize([S[0]])

            for tau in range(t+1,T):
                qS[tau] = (pr.exp_tr(pr.sum([S[tau-1]],pr.log(Tr[tau-1])*qS[tau-1]) 
                        + pr.sum([S[tau+1],A[tau]],pr.log(Tr[tau])*qS[tau+1]*pr.sum(qA,[A[tau]])))).normalize([S[tau]])
            qS[T] = pr.exp_tr(pr.sum([S[T-1]],pr.log(Tr[T-1])*qS[T-1])).normalize([S[T]])
            G = 0
            for tau in range(t+1,T+1):
                G = G + ( pr.sum([S[tau-1],S[tau]],pr.log(Tr[tau-1])*qS[tau]*qS[tau-1])
                    - pr.sum([S[tau]],pr.log(qS[tau])*qS[tau]) 
                    + pr.sum([X[tau],S[tau]],(pr.log(L[tau])+pr.log(D[tau]))*L[tau]*qS[tau])
                    - pr.sum([X[tau],S[tau]],pr.log(pr.sum([S[tau]],L[tau]*qS[tau]))*L[tau]*qS[tau]))

            qA = (pA0*pr.exp_tr(self.alpha*G)).normalize()

            qAmarg = pr.sum(qA,[A[t]]).val
            if np.linalg.norm(qAmarg-q0) < precision:
                break
            q0 =  qAmarg
        return qAmarg

    def _get_action_dist(self):
        self._set_fixed()
        return self.meanField()




class ExactQ201516(ProbAgent):

    def __init__(self,env,vision=.1,maxSteps=6,alpha=1.0):
        ProbAgent.__init__(self, env, vision, maxSteps)
        self.memory = 0
        self.alpha = alpha

    def _get_action_dist(self):
        self._set_fixed()
        N,K,T,S,A,X,t = self.N,self.K,self.T,self.S,self.A,self.X,self.t
        Tr,L,D,pS0,pA0 = self.Tr,self.L,self.D,self.pS0,self.pA0
        ## inference
        if t==0:
            pSt = (L[0]*pS0).normalize([S[0]])
        else:
            pSt0 = pr.sum([S[t-1]],Tr[t-1]*self.memory)
            pSt = (L[t]*pSt0).normalize([S[t]])
        self.memory = pSt
        ## action distribution
        pred = pSt # placeholder for previous predictive state distributions
        Q = 0      # placeholder for value function
        for tau in range(t+1,T+1):
            pred = pr.sum([S[tau-1]],Tr[tau-1]*pred)
            Q = Q + ( pr.sum([X[tau],S[tau]],(pr.log(L[tau])+pr.log(D[tau]))*L[tau]*pred) 
                - pr.sum([X[tau],S[tau]],pr.log(pr.sum([S[tau]],L[tau]*pred))*L[tau]*pred))
        pA = (pA0*pr.exp_tr(self.alpha*Q)).normalize()

        return pr.sum(pA,[A[t]]).val # return marginal of pA with respect to At


class ExactInference(ProbAgent):

    def __init__(self,env,vision=.1,maxSteps=6,alpha=1.0):
        ProbAgent.__init__(self, env, vision, maxSteps)
        self.memory = 0
        self.alpha = alpha

    def _get_action_dist(self):
        self._set_fixed()
        N,K,T,S,A,X,t = self.N,self.K,self.T,self.S,self.A,self.X,self.t
        Tr,L,D,pS0,pA0 = self.Tr,self.L,self.D,self.pS0,self.pA0
        ## inference
        if t==0:
            pSt = (L[0]*pS0).normalize([S[0]])
        else:
            pSt0 = pr.sum([S[t-1]],Tr[t-1]*self.memory)
            pSt = (L[t]*pSt0).normalize([S[t]])
        self.memory = pSt
        ## action distribution
        pred = pSt # placeholder for previous predictive state distributions
        Q = 0      # placeholder for value function
        for tau in range(t+1,T+1):
            pred = pr.sum([S[tau-1]],Tr[tau-1]*pred)
            Q = Q + pr.sum([X[tau],S[tau]],pr.log(D[tau])*L[tau]*pred) 
        pA = (pA0*pr.exp_tr(self.alpha*Q)).normalize()
        return pr.sum(pA,[A[t]]).val # return marginal of pA with respect to At


class Jeffrey(ProbAgent):

    def __init__(self,env,vision=.1,maxSteps=6):
        ProbAgent.__init__(self, env, vision, maxSteps)
        self.memory = 0

    def _get_action_dist(self):
        self._set_fixed()
        N,K,T,S,A,X,t = self.N,self.K,self.T,self.S,self.A,self.X,self.t
        Tr,L,D,pS0,pA0 = self.Tr,self.L,self.D,self.pS0,self.pA0
        xdes = np.random.choice(range(N),p=D[T].val)
        ## inference over states
        if t==0:
            pSt = (L[0]*pS0).normalize([S[0]])
        else:
            pSt0 = pr.sum([S[t-1]],Tr[t-1]*self.memory)
            pSt = (L[t]*pSt0).normalize([S[t]])
        self.memory = pSt
        pred = pSt # placeholder for previous predictive state distributions
        ## inference over actions
        for tau in range(t+1,T+1):
            pred = pr.sum([S[tau-1]],Tr[tau-1]*pred)
        pA = pr.sum([S[T]],L[T].eval(X[T],xdes)*pred).normalize()
        return pr.sum(pA,[A[t]]).val # return marginal of pA with respect to At



class GeneralisedFreeEnergy(ProbAgent):

    def __init__(self,env,vision=.1,maxSteps=6,iterations=10,alpha=1.0):
        ProbAgent.__init__(self, env, vision, maxSteps)
        self.iterations = iterations
        self.alpha = alpha

    def meanField(self):
        N,K,T,S,A,X,t = self.N,self.K,self.T,self.S,self.A,self.X,self.t
        Tr,L,D,pS0,pA0 = self.Tr,self.L,self.D,self.pS0,self.pA0 
        ### initialize
        qA = pr.func(vars=list(A.values()), val='unif').normalize()
        qS = {**{i: pr.func(vars=[S[i]], val='unif').normalize([S[i]]) for i in range(0,t+1)},
              **{i: pr.func(vars=[S[i],*[A[j] for j in range(t,i)]],val='unif').normalize([S[i]]) for i in range(t+1,T+1)}}
        ## repeat:
        for iter in range(self.iterations):
            if t > 0:
                qS[0] = (L[0]*pS0*pr.exp_tr(pr.sum([S[1]],pr.log(Tr[0])*qS[1]))).normalize([S[0]])
                for tau in range(1,t):
                    qS[tau] = (L[tau]*pr.exp_tr(pr.sum([S[tau-1]],pr.log(Tr[tau-1])*qS[tau-1])) 
                                        +pr.sum([S[tau+1]],pr.log(Tr[tau])*qS[tau+1])).normalize([S[tau]])
                qS[t] = (L[t]*pr.exp_tr(pr.sum([S[t-1]],pr.log(Tr[t-1])*qS[t-1])) 
                                        +pr.sum([S[t+1],A[t]],pr.log(Tr[t])*qS[t+1]*pr.sum(qA,[A[t]]))).normalize([S[t]])
            else: 
                qS[0] = (L[0]*pS0*pr.exp_tr(pr.sum([S[1],A[0]],pr.log(Tr[0])*qS[1]*pr.sum(qA,[A[0]])))).normalize([S[0]])

            for tau in range(t+1,T):
                qS[tau] = (pr.exp_tr(pr.sum([S[tau-1]],pr.log(Tr[tau-1])*qS[tau-1]) 
                        + pr.sum([S[tau+1],A[tau]],pr.log(Tr[tau])*qS[tau+1]*pr.sum(qA,[A[tau]]))
                        + pr.sum([X[tau]],pr.log(D[tau])*L[tau]) 
                        )/(pr.sum([X[tau]],L[tau]*D[tau])+1e-30) ).normalize([S[tau]])
            qS[T] = (pr.exp_tr(pr.sum([S[T-1]],pr.log(Tr[T-1])*qS[T-1])
                        + pr.sum([X[T]],pr.log(D[T])*L[T])
                        )/(pr.sum([X[T]],L[T]*D[T])+1e-30) ).normalize([S[T]])
            G = 0
            for tau in range(t+1,T+1):
                G = G + ( pr.sum([S[tau-1],S[tau]],pr.log(Tr[tau-1])*qS[tau]*qS[tau-1])
                    - pr.sum([S[tau]],pr.log(qS[tau])*qS[tau]) 
                    - pr.sum([S[tau]], pr.log(pr.sum([X[tau]],L[tau]*D[tau]))*qS[tau])
                    + pr.sum([X[tau],S[tau]],pr.log(D[tau])) )
            qA = (pA0*pr.exp_tr(self.alpha*G)).normalize()
        return pr.sum(qA,[A[t]]).val

    def _get_action_dist(self):
        self._set_fixed()
        return self.meanField()


class Toussaint(ProbAgent):

    def __init__(self,env,vision=.1,maxSteps=6,gamma=.99):
        ProbAgent.__init__(self, env, vision, maxSteps)
        self.memory = 0
        self.gamma = gamma

    def _get_action_dist(self):
        self._set_fixed()
        N,K,T,S,A,X,t = self.N,self.K,self.T,self.S,self.A,self.X,self.t
        Tr,L,D,pS0,pA0 = self.Tr,self.L,self.D,self.pS0,self.pA0
        pA = 0   
        psuccess=D[T]
        ## inference over states
        if t==0:
            pSt = (L[0]*pS0).normalize([S[0]])
        else:
            pSt0 = pr.sum([S[t-1]],Tr[t-1]*self.memory)
            pSt = (L[t]*pSt0).normalize([S[t]])
        self.memory = pSt
        pred = pSt # placeholder for previous predictive state distributions
        ## inference over actions
        for tau0 in range(t+1,T+1):
            pred = pr.sum([S[tau0-1]],Tr[tau0-1]*pred)
        pA = pA + self.gamma**T * pr.sum([X[T],S[T]],psuccess*L[T]*pred).normalize()
        return pr.sum(pA.normalize(),[A[t]]).val # return marginal of pA with respect to At
    
    def _policy(self):
        qAt = self._get_action_dist()
        self.actionDists.append(qAt)
        return np.argmax(qAt)


class Toussaint_mf(ProbAgent):

    def __init__(self,env,vision=.1,maxSteps=6,iterations=10,alpha=1.0):
        ProbAgent.__init__(self, env, vision, maxSteps)
        self.iterations = iterations
        self.alpha = alpha

    def meanField(self):
        N,K,T,S,A,X,t = self.N,self.K,self.T,self.S,self.A,self.X,self.t
        Tr,L,D,pS0,pA0 = self.Tr,self.L,self.D,self.pS0,self.pA0 
        ### initialize
        qA = pr.func(vars=list(A.values()), val='unif').normalize()
        qS = {**{i: pr.func(vars=[S[i]], val='unif').normalize([S[i]]) for i in range(0,t+1)},
              **{i: pr.func(vars=[S[i],*[A[j] for j in range(t,i)]],val='unif').normalize([S[i]]) for i in range(t+1,T+1)}}
        ## repeat:
        for iter in range(self.iterations):
            if t > 0:
                qS[0] = (L[0]*pS0*pr.exp_tr(pr.sum([S[1]],pr.log(Tr[0])*qS[1]))).normalize([S[0]])
                for tau in range(1,t):
                    qS[tau] = (L[tau]*pr.exp_tr(pr.sum([S[tau-1]],pr.log(Tr[tau-1])*qS[tau-1])) 
                                        +pr.sum([S[tau+1]],pr.log(Tr[tau])*qS[tau+1])).normalize([S[tau]])
                qS[t] = (L[t]*pr.exp_tr(pr.sum([S[t-1]],pr.log(Tr[t-1])*qS[t-1])) 
                                        +pr.sum([S[t+1],A[t]],pr.log(Tr[t])*qS[t+1]*pr.sum(qA,[A[t]]))).normalize([S[t]])
            else: 
                qS[0] = (L[0]*pS0*pr.exp_tr(pr.sum([S[1],A[0]],pr.log(Tr[0])*qS[1]*pr.sum(qA,[A[0]])))).normalize([S[0]])

            for tau in range(t+1,T):
                qS[tau] = (pr.exp_tr(pr.sum([S[tau-1]],pr.log(Tr[tau-1])*qS[tau-1]) 
                        + pr.sum([S[tau+1],A[tau]],pr.log(Tr[tau])*qS[tau+1]*pr.sum(qA,[A[tau]]))
                        + pr.sum([X[tau]],pr.log(D[tau])*L[tau]) 
                        )).normalize([S[tau]])
            qS[T] = (pr.exp_tr(pr.sum([S[T-1]],pr.log(Tr[T-1])*qS[T-1])
                        + pr.sum([X[T]],pr.log(D[T])*L[T])
                        )).normalize([S[T]])
            G = 0
            for tau in range(t+1,T+1):
                G = G + ( pr.sum([S[tau-1],S[tau]],pr.log(Tr[tau-1])*qS[tau]*qS[tau-1])
                    - pr.sum([S[tau]],pr.log(qS[tau])*qS[tau]) 
                    + pr.sum([X[tau],S[tau]],pr.log(D[tau])) )
            qA = (pA0*pr.exp_tr(self.alpha*G)).normalize()
        return pr.sum(qA,[A[t]]).val

    def _get_action_dist(self):
        self._set_fixed()
        return self.meanField()





def get_qS_GD(t,tau,T,N,K,trans,likelihood,pdes,qsm,qsp,qact):
    import tensorflow as tf
    import os,time
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    qa = tf.constant(qact)
    lh = tf.constant(likelihood)

    epochs = 1000
    precision = 1e-4
    optimizer = tf.keras.optimizers.Adam(lr=.1)
    
    def sh_1s(shp,n):
        return shp+[1 for i in range(n)]


    def softmax(q):
        Z = tf.expand_dims(tf.reduce_sum(tf.exp(q),axis=0),0)
        return tf.exp(q)/Z


    # tf Variable # tau = t+2 -> q = q(S_t+2|At,At+1)
    q = tf.Variable(np.random.uniform(0,1,size=[N]+[4 for i in range(tau-t)]))

    # optimize free energy directly
    @tf.function 
    def opt(q):
        with tf.GradientTape() as g:
            qs = softmax(q)

            sh_S111 = sh_1s([N],tau-t) # = (S,1,...,1) with tau-t "1"s
            sh_SX111 = sh_1s([N,N],tau-t)    # = (S,X,1,...,1) with tau-t "1"s
            sh_A111 = sh_1s([K]*(tau-t),T-tau) # = (At,...,Atau-1,1,...,1) with T-tau "1"s
            sh_SSp111A = sh_1s([N,N],tau-t)+[K]
            sh_SSm111A = sh_1s([N,N],tau-t-1)+[K]

            obj = 0
            # t1 = -<log L pdes>_qS(S|A)
            v1 = tf.reduce_sum(lh*tf.math.log(lh*tf.expand_dims(pdes,0)),axis=1)
            t1 = -tf.reduce_sum(qs*tf.reshape(v1,sh_S111),axis=0)
            # t2 = <log qS>_qS|A
            t2 = tf.reduce_sum(qs*tf.math.log(qs),axis=0)
            # t3 = <log qX>_qX|A
            qx = tf.reduce_sum( tf.expand_dims(qs,1) * tf.reshape(lh,sh_SX111) ,axis=0)
            t3 = tf.reduce_sum(qx*tf.math.log(qx),axis=0)
            # add t1,t2,t3 and take expectation wrt qa
            obj += tf.reduce_sum(qa*tf.reshape(t1+t2+t3,sh_A111))
            # t4 = -<log T(Stau+1|Stau,A_tau)>
            logTrans1 = tf.transpose(tf.math.log(trans),[0,2,1])     # Stau,Stau+1,A
            if tau<T:
                v4 = tf.expand_dims(qsp,0) * tf.reshape(logTrans1,sh_SSp111A)  # Stau,Stau+1,At,...,Atau 
                t4 = -tf.reduce_sum(v4,axis=1)
                t4_A = tf.reduce_sum(t4*tf.expand_dims(qs,-1),axis=0)
                obj += tf.reduce_sum(qa*tf.reshape(t4_A,sh_1s(np.shape(t4_A),T-tau-1)))
            # t5 = <log T(Stau|Stau-1,A_tau-1)> 
            logTrans2 = tf.transpose(tf.math.log(trans),[2,0,1])     # Stau-1,A,Stau -> Stau,Stau-1,A
            v5 = tf.expand_dims(tf.expand_dims(qsm,0),-1) * tf.reshape(logTrans2,sh_SSm111A)  # Stau,Stau-1,At,...,Atau-1
            t5 = -tf.reduce_sum(v5,axis=1)
            t5_A = tf.reduce_sum(t5*qs,axis=0)
            obj += tf.reduce_sum(qa*tf.reshape(t5_A,sh_A111))        
        gradients = g.gradient(obj, [q])
        optimizer.apply_gradients(zip(gradients, [q]))

    t0 = time.time()
    for i in range(epochs):
        q0 = softmax(q).numpy()
        opt(q)
        if (np.linalg.norm(softmax(q).numpy()-q0) < precision):
            break
    t1 = time.time()

    result = {
        'q': softmax(q).numpy(), 
        'iterations': i, 
        'elapsed': t1-t0,
        }
    
    # print(result['iterations'],'epochs')
    return result['q']


class ActiveInference2016_GD(ProbAgent):

    def __init__(self,env,vision=.1,maxSteps=6,iterations=200,alpha=1.0):
        ProbAgent.__init__(self, env, vision, maxSteps)
        self.iterations = iterations
        self.alpha = alpha

    def meanField(self):
        precision=1e-3
        N,K,T,S,A,X,t = self.N,self.K,self.T,self.S,self.A,self.X,self.t
        Tr,L,D,pS0,pA0 = self.Tr,self.L,self.D,self.pS0,self.pA0 
        ### initialize
        qA = pr.func(vars=list(A.values()), val='unif').normalize()
        qS = {**{i: pr.func(vars=[S[i]], val='unif').normalize([S[i]]) for i in range(0,t+1)},
              **{i: pr.func(vars=[S[i],*[A[j] for j in range(t,i)]],val='unif').normalize([S[i]]) for i in range(t+1,T+1)}}
        ## repeat:
        q0 = 0
        for iter in range(self.iterations):
            # PAST AND CURRENT STATES
            if t > 0:
                qS[0] = (L[0]*pS0*pr.exp_tr(pr.sum([S[1]],pr.log(Tr[0])*qS[1]))).normalize([S[0]])
                for tau in range(1,t):
                    qS[tau] = (L[tau]*pr.exp_tr(pr.sum([S[tau-1]],pr.log(Tr[tau-1])*qS[tau-1])) 
                                        +pr.sum([S[tau+1]],pr.log(Tr[tau])*qS[tau+1])).normalize([S[tau]])
                qS[t] = (L[t]*pr.exp_tr(pr.sum([S[t-1]],pr.log(Tr[t-1])*qS[t-1])) 
                                        +pr.sum([S[t+1],A[t]],pr.log(Tr[t])*qS[t+1]*pr.sum(qA,[A[t]]))).normalize([S[t]])
            else: 
                qS[0] = (L[0]*pS0*pr.exp_tr(pr.sum([S[1],A[0]],pr.log(Tr[0])*qS[1]*pr.sum(qA,[A[0]])))).normalize([S[0]])

            # FUTURE STATES --- replaced by gradient descent ---------
            for tau in range(t+1,T):
                qp = qS[tau+1].val
                qm = qS[tau-1].val
                qS[tau].val = get_qS_GD(t,tau,T,N,K,self.pTrans,self.likelihood,self.pDesired,qm,qp,qA.val)            
            qS[T].val = get_qS_GD(t,T,T,N,K,self.pTrans,self.likelihood,self.pDesired,qS[T-1].val,[],qA.val)
            #----------------------------------------------------------
            # ACTION
            G = 0
            for tau in range(t+1,T+1):
                G = G + ( pr.sum([S[tau-1],S[tau]],pr.log(Tr[tau-1])*qS[tau]*qS[tau-1])
                    - pr.sum([S[tau]],pr.log(qS[tau])*qS[tau]) 
                    + pr.sum([X[tau],S[tau]],(pr.log(L[tau])+pr.log(D[tau]))*L[tau]*qS[tau])
                    - pr.sum([X[tau],S[tau]],pr.log(pr.sum([S[tau]],L[tau]*qS[tau]))*L[tau]*qS[tau]))
            qA = (pA0*pr.exp_tr(self.alpha*G)).normalize()

            qAmarg = pr.sum(qA,[A[t]]).val
            if np.linalg.norm(qAmarg-q0) < precision:
                break
            q0 =  qAmarg
        return qAmarg

    def _get_action_dist(self):
        self._set_fixed()
        return self.meanField()



class ActiveInference2016_fullpolicy(ProbAgent):

    def __init__(self,env,vision=.1,maxSteps=6,iterations=100,alpha=1.0):
        ProbAgent.__init__(self, env, vision, maxSteps)
        self.iterations = iterations
        self.alpha = alpha

    def meanField(self):
        precision=1e-4
        N,K,T,S,A,X,t = self.N,self.K,self.T,self.S,self.A,self.X,self.t
        Tr,L,D,pS0,pA0 = self.Tr,self.L,self.D,self.pS0,self.pA0 
        ### initialize
        qA = pr.func(vars=list(A.values()), val='unif').normalize()
        qS = {**{i: pr.func(vars=[S[i],*[A[j] for j in range(t,T)]], val='unif').normalize([S[i]]) for i in range(0,t+1)},
              **{i: pr.func(vars=[S[i],*[A[j] for j in range(t,T)]],val='unif').normalize([S[i]]) for i in range(t+1,T+1)}}

        ## repeat:
        q0 = 0
        for iter in range(self.iterations):
            qS[0] = (L[0]*pS0*pr.exp_tr(pr.sum([S[1]],pr.log(Tr[0])*qS[1]))).normalize([S[0]])

            if t > 0:
                for tau in range(1,t):
                    qS[tau] = (L[tau]*pr.exp_tr(pr.sum([S[tau-1]],pr.log(Tr[tau-1])*qS[tau-1])) 
                                        +pr.sum([S[tau+1]],pr.log(Tr[tau])*qS[tau+1])).normalize([S[tau]])
                qS[t] = (L[t]*pr.exp_tr(pr.sum([S[t-1]],pr.log(Tr[t-1])*qS[t-1])) 
                                        +pr.sum([S[t+1]],pr.log(Tr[t])*qS[t+1])).normalize([S[t]])

            for tau in range(t+1,T):
                qS[tau] = (pr.exp_tr(pr.sum([S[tau-1]],pr.log(Tr[tau-1])*qS[tau-1]) 
                        + pr.sum([S[tau+1]],pr.log(Tr[tau])*qS[tau+1]))).normalize([S[tau]])

            qS[T] = pr.exp_tr(pr.sum([S[T-1]],pr.log(Tr[T-1])*qS[T-1])).normalize([S[T]])
            G = 0
            for tau in range(t+1,T+1):
                G = G + ( pr.sum([S[tau-1],S[tau]],pr.log(Tr[tau-1])*qS[tau]*qS[tau-1])
                    - pr.sum([S[tau]],pr.log(qS[tau])*qS[tau]) 
                    + pr.sum([X[tau],S[tau]],(pr.log(L[tau])+pr.log(D[tau]))*L[tau]*qS[tau])
                    - pr.sum([X[tau],S[tau]],pr.log(pr.sum([S[tau]],L[tau]*qS[tau]))*L[tau]*qS[tau]))

            G = G + ( pr.sum([S[0]],pr.log(pS0)*qS[0]) 
                    - pr.sum([S[0]],pr.log(qS[0])*qS[0]) 
                    + pr.sum([S[0]],pr.log(L[0])*qS[0]))

            for tau in range(1,t+1):
                G = G + ( pr.sum([S[tau-1],S[tau]],pr.log(Tr[tau-1])*qS[tau]*qS[tau-1]) 
                    - pr.sum([S[tau]],pr.log(qS[tau])*qS[tau]) 
                    + pr.sum([S[tau]],pr.log(L[tau])*qS[tau]) )

            qA = (pA0*pr.exp_tr(self.alpha*G)).normalize()

            qAmarg = pr.sum(qA,[A[t]]).val
            if np.linalg.norm(qAmarg-q0) < precision:
                break
            q0 =  qAmarg
        return qAmarg

    def _get_action_dist(self):
        self._set_fixed()
        return self.meanField()






