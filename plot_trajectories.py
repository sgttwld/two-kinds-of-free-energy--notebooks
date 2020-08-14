import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import _lib.pr_func as pr
from _lib.environment import * 
# from _lib.agents import *
from _lib.agent_config import *


def load_trajectories(select,walls=True):
    ## load trajectories from files
    trajectories = []
    if walls:
        for slug in select:
            f = np.load('_lib/_trajectories/traj_{}.npz'.format(slug))
            trajectories.append(f['arr_0'])
    else: 
        for slug in select:
            f = np.load('_lib/_trajectories/traj_{}_nowalls.npz'.format(slug))
            trajectories.append(f['arr_0'])
    return trajectories
      

def show_paths(env,sigmas,steps,trajectories,select,spec,num=8):
    agents = get_agent_config(env,sigmas,steps)
    cmap=plt.get_cmap('Greys')
    linestyles = [(0,(1,i/2)) for i in range(num)]
    offset = np.append(np.array([0]),np.linspace(-.1,.1,num-1))
    subplotspec = spec
    fig,ax = plt.subplots(spec[0],spec[1], figsize=(11/3*spec[1],4*spec[0]))


    for j in range(len(select)):

        trajects = trajectories[j]
        agent = agents[select[j]]

        eps = offset


        if subplotspec[0]>1:
            plotCoords = np.unravel_index(j,subplotspec)
        else: 
            plotCoords = j

        ## convert to unique paths with probabilities
        trajs,counts = np.unique(trajects,axis=0,return_counts=True)
        paths = []
        for i in range(len(trajs)):
            paths.append({'path': [env.state_num_to_tuple(s) for s in trajs[i]], 'prob': counts[i]/np.sum(counts)})

        ## get most probable paths
        pathprobs = np.array([p['prob'] for p in paths])
        
        inds = pathprobs.argsort()[::-1][:num]

        ax[plotCoords].axis('off')
        ax[plotCoords].imshow(env.gridmap,cmap=cmap)
        for goal_state in env.goal_states:
            ax[plotCoords].scatter(goal_state[1],goal_state[0],s=150,marker='x',color='black')
        initial_state = env.state_num_to_tuple(env.initial_agent_s)
        ax[plotCoords].scatter(initial_state[1],initial_state[0],s=100,marker='s',color='grey',alpha=1)
        ax[plotCoords].imshow(sigmas,alpha=.3,cmap=cmap)
        ax[plotCoords].set_title('{}: '.format(agent['section']) + agent['name'],fontsize=12)

        for k in range(len(inds)):
            pth = paths[inds[k]]
            statePath = pth['path']
            prob = pth['prob']

            invStatePath = [[s[1],s[0]] for s in statePath]

            Path = mpath.Path
            path_data = [(Path.MOVETO, invStatePath[0])]
            for state in invStatePath:
                path_data.append((Path.LINETO, state))

            ax[plotCoords].scatter(invStatePath[-1][0],invStatePath[-1][1],s=50,marker='s',color='red',alpha=.5)

            codes, verts = zip(*path_data)
            path = mpath.Path(verts, codes)

            x, y = zip(*path.vertices)
            line, = ax[plotCoords].plot(np.array(x)+eps[k], np.array(y)+eps[k], 'r' , ls=linestyles[k],alpha=1)
            # line, = ax[plotCoords].plot(np.array(x), np.array(y), 'r' , ls=':',alpha=.8)

    plt.show()
                   

if __name__ == "__main__":

    c = 100.0
    sigmas = np.array([
        [.1,.1,.1,.1,.1,.1],
        [.1,.1,.1, c, c,.1],
        [.1,.1,.1, c, c,.1],
        [.1,.1,.1,.1,.1,.1],
        [.1,.1,.1,.1,.1,.1],
        [.1,.1,.1,.1,.1,.1]])
    steps = 5
    # env = GridEnv(planfile='_plans/planK.txt',render=False)     # without walls
    env = GridEnv(planfile='_plans/planL.txt',render=False)   # with walls

    select = ['qmf','qmff']
    # select = ['jeff','touss','util','qexact']
    # select = ['qmf','toussmf','toussbethe','genfe']
    trajectories = load_trajectories(select,walls=True)
    show_paths(env,sigmas,steps,trajectories,select,spec=[1,2],num=8)

    # merge GD trajectories:
    # traj = np.load('_lib/_trajectories/traj_qgd.npz')
    # traj_nowalls = np.load('_lib/_trajectories/traj_qgd_nowalls.npz')
    # print(np.shape(traj['arr_0']),np.shape(traj_nowalls['arr_0']))
    # oldGD = np.load('_lib/_trajectories/traj_agent6.npz')
    # newGD = np.load('traj_agent_GD_walls.npz')
    # print(np.shape(oldGD['arr_0']))
    # combGD = np.append(oldGD['arr_0'],newGD['arr_0'],axis=0)
    # np.savez('traj_agent6.npz',combGD)


