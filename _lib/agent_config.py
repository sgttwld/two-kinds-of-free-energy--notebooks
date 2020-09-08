from _lib.agents import *

def get_agent_config(env,sigmas,steps): 
    return {
        'jeff': {
            'name': r'Conditioning on $p_\mathrm{des}$' + '\n (Jeffrey conditioning)',
            'sname': r'Conditioning on $p_\mathrm{des}$',
            'section': 'I',
            'slug': 'jeff',
            'agent': Jeffrey(env,vision=sigmas,maxSteps=steps),
        },
        'touss': {
            'name': 'Conditioning on success \n (Control as Inference)',
            'sname': 'Conditioning on success',
            'section': 'II',
            'slug': 'touss',
            'agent': Toussaint(env,vision=sigmas,maxSteps=steps),
        },
        'util': {
            'name': 'Soft-maximizing \n expected utility',
            'sname': 'Expected utility',
            'section': 'III',
            'slug': 'util',
            'agent': ExactInference(env, vision=sigmas, maxSteps=steps),
        },
        'qmf': {
            'name': 'Q-value Active Inference\n (mean-field; Friston et al. 2016/17)',
            'sname': 'Q-value Active Inference',
            'section': 'IV.a',
            'slug': 'qmf',
            'agent': ActiveInference2016(env,vision=sigmas,maxSteps=steps),
        },
        'qmff': {
            'name': 'Q-value Act. Inference \n (dependency on full path)',
            'sname': 'Q-value Act. Inf. full-A',
            'section': '',
            'slug': 'qmff',
            'agent': ActiveInference2016_fullpolicy(env,vision=sigmas,maxSteps=steps),
        },
        'qgd': {
            'name': "Q-value Active Inference \n incl. $q$-dependency of $Q$ (mf)",
            'sname': 'incl. $q$-dependency of Q-value',
            'section': 'IV.b',
            'slug': 'qgd',
            'agent': ActiveInference2016_GD(env,vision=sigmas,maxSteps=steps)
        },
        'qexact': {
            'name': r"Soft-maximizing $Q_{exact}$",
            'sname': r"Soft-maximizing $Q_{exact}$",
            'section': 'IV.c',
            'slug': 'qexact',
            'agent': ExactQ201516(env,vision=sigmas,maxSteps=steps),
        },
        'toussmf': {
            'name': 'Conditioning on success\n (mean-field; Schwöbel et al. 2018)',
            'sname': 'Conditioning on success (mf)',
            # 'name': 'direct Active Inference, \n  conditioning on success (mf)',
            'section': 'V.a',
            'slug': 'toussmf',
            'agent': Toussaint_mf(env,vision=sigmas,maxSteps=steps),
        },
        'toussbethe': {
            'name': 'Conditioning on success\n (Bethe ass.; Schwöbel et al. 2018)',
            'sname': 'Cond. on success (Bethe)',
            # 'name': 'direct Active Inference, \n conditioning on success (Bethe)',
            'section': 'V.b',
            'slug': 'toussbethe',
            'agent': Toussaint(env,vision=sigmas,maxSteps=steps),
        },
    }
