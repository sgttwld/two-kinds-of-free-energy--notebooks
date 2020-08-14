import matplotlib.pyplot as plt
import numpy as np
import os, sys

class GridEnv(object):

    def __init__(self,render=True,planfile='', plan='',speed='fast',verbose=True,figsize=(3,3)):

        self.gridmap, self.initial_agent_state, self.goal_states = self._read_gridmap(planfile,plan)
        self.dim_states = np.shape(self.gridmap)
        self.initial_agent_s = self.state_tuple_to_num(self.initial_agent_state)
        self.agent_s = self.initial_agent_s
        self.goal_s = [self.state_tuple_to_num(state) for state in self.goal_states]
        self.states = list(range(np.prod(self.dim_states)))
        self.action_dict = {
            0: [-1,0],
            1: [1,0],
            2: [0,-1],
            3: [0,1],
            }
        self.actions = list(self.action_dict.keys())
        self.done = False
        self.verbose = verbose
        self.shall_render = render
        self.speed = speed
        self.current_obs = [0,0]
        self.trajectory = [self.initial_agent_s]
        if self.shall_render:
            cmap=plt.get_cmap('Greys')
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(111)
            plt.axis('off')
            self.ax.imshow(self.gridmap,cmap=cmap)
            for goal_state in self.goal_states:
                self.ax.scatter(goal_state[1],goal_state[0],s=100,marker='x',color='black')
            initial_state = self.state_num_to_tuple(self.initial_agent_s)
            self.ax.scatter(initial_state[1],initial_state[0],s=50,marker='s',color='gray')
            agent_state = self.state_num_to_tuple(self.agent_s)
            self.scat = self.ax.scatter(agent_state[1],agent_state[0],s=50,marker='s',color='r')

    def _read_gridmap(self,planfile,plan):
        if len(planfile) > 0:
            this_file_path = os.path.dirname(os.path.realpath(__file__))
            gridmap_path = os.path.join(this_file_path, planfile) 
            with open(gridmap_path, 'r') as f:
                gridmap_str = f.readlines()
        elif len(plan) > 0:
            gridmap_str = plan[1:-1].split('\n')
        gm = [line.split(' ') for line in gridmap_str]
        agent_pos, goal_pos = (0,0), []
        for i in range(len(gm)):
            for j in range(len(gm[0])):
                if gm[i][j] == 'S': 
                    starting_pos = (i,j)
                    gm[i][j] = '0'
                elif gm[i][j] == 'G':
                    goal_pos.append((i,j))
                    gm[i][j] = '0'
                gm[i][j] = int(gm[i][j])
        return np.array(gm), starting_pos, goal_pos

    def state_tuple_to_num(self,tpl):
        k,l = tpl
        return k*self.dim_states[1]+l

    def state_num_to_tuple(self,num):
        return np.unravel_index(num,self.dim_states)

    def render(self):
        if self.shall_render:
            agent_state = self.state_num_to_tuple(self.agent_s)
            self.scat.set_offsets([[agent_state[1],agent_state[0]]])
            self.fig.canvas.draw()
            if self.speed == 'slow':
                plt.pause(.10) # ~ 10-15 steps/s
            if self.speed == 'medium':
                plt.pause(.005)
            if self.speed == 'fast':
                plt.pause(.000001) # ~ 200-350 steps/s (no-render: 180.000 steps/s with random agent)

    def transition(self,s,a):
        """
        state, action => next state, reward
        """
        state = self.state_num_to_tuple(s)
        if self.gridmap[state[0],state[1]] == 1:
            # on the wall
            return s, -1
        elif  (s in self.goal_s):
            # on the goal
            return s, 1.5
        elif self.gridmap[state[0],state[1]] == 2:
            # in a hole
            return self.initial_agent_s, -100
        next_state = (state[0]+self.action_dict[a][0], state[1]+self.action_dict[a][1])
        s1 = self.state_tuple_to_num(next_state)
        if (s1 in self.goal_s):
            # hit goal
            return s1, 1
        elif self.gridmap[next_state[0],next_state[1]] == 1:
            # hit wall
            return s, -1
        elif self.gridmap[next_state[0],next_state[1]] == 2:
            # fell into a hole
            return self.initial_agent_s, -3
        else:
            # normal step
            return s1, 0

    def proc(self,action):
        """
        current agent state, action => next state, reward (for next state)
        """   
        next_s, reward = self.transition(self.agent_s,action)
        if (reward == 1):
            # hit goal
            self.done = True
        self.agent_s = next_s
        self.trajectory.append(next_s)
        return next_s, reward

    def reset(self):
        self.done = False
        self.trajectory = [self.initial_agent_s]
        self.agent_s= self.initial_agent_s
        agent_state = self.state_num_to_tuple(self.agent_s)
        self.current_obs = [self.initial_agent_s,0]
        if self.shall_render:
            self.scat.set_offsets([[agent_state[1],agent_state[0]]])
            self.fig.canvas.draw()


    def run(self,agent,steps=50,episodes=25):

        for i in range(episodes):
            sys.stdout.write("\rNavigating...")
            self.reset()
            agent.actionDists = []

            
            for j in range(steps):

                action = agent.proc(self.current_obs)
                self.current_obs = self.proc(action)
                self.render()

                if self.done:  
                    if self.verbose:
                        if self.agent_s in self.goal_s:
                            print('Done: Success after {} steps!'.format(j+1))
                            # print('success after {} steps in the episode!'.format(j+1))
                        else:
                            print('Fail: Episode stopped after {} steps'.format(j+1))
                    break
            agent.evaluate_episode()
        
