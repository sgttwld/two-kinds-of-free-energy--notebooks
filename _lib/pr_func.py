# -*- coding: utf-8 -*-
# pr_func.py
# implementation of a 'function' container class containing a numpy
# array and knowledge about its dependent variables.
# 2018 Sebastian Gottwald <sebastian.gottwald@uni-ulm.de>

import numpy as np

global dims

eps = 1e-25

def set_dims(d=[]):
    global dims
    dims = d

def get_r(variables,dims):
    r = []
    for item in variables:
        for i in range(0,len(dims)):
            if dims[i][0] == item:
                r.append(i)
    return r

def exp(fnc):
    return func(val=np.exp(fnc.val), r=fnc.r, parse_name=False)

def exp_tr(fnc):
    return func(val=np.exp(fnc.val-np.max(fnc.val)), r=fnc.r, parse_name=False)

def exp_tr_min(fnc):
    return func(val=np.exp(fnc.val-np.min(fnc.val)), r=fnc.r, parse_name=False)

def log(fnc):
    return func(val=np.log(fnc.val+eps), r=fnc.r, parse_name=False)

def sum(*args):
    """
    sum over a func object, args can consist of one or two arguments;
    if only one argument is given, the sum is over all variables;
    if two args are given: args = [vars,func_obj] or args = [func_obj, vars],
    where vars is a list of variable strings, i.e. vars = ['x','w',...]
    * if args[0] = vars, then vars specifies the variables to sum over
    * if args[1] = vars, then vars specifies the variables of the result
    """
    if len(args) == 1:
        r_target = []
        val = np.einsum(args[0].val,args[0].r,r_target)
    else:
        if type(args[0]) == list:
            r_over = get_r(args[0],dims)
            r_target = [item for item in args[1].r if not(item in r_over)]
            val = np.einsum(args[1].val,args[1].r,r_target)
        elif type(args[1]) == list:
            r_target = get_r(args[1],dims)
            val = np.einsum(args[0].val,args[0].r,r_target)

    return func(val=val,r=r_target,parse_name=False)


class func(object):
    """function class that knows about its variables"""

    def __init__(self,name='',val='',vars=[],r=[],parse_name=False):
        # set vars:
        if len(vars)>0:
            self.vars = vars
        elif parse_name:
            self.vars = self.parse_name(name)
        elif len(r)>0:
            self.vars = [dims[ind][0] for ind in r]
        else:
            self.vars = []
        # set r:
        if len(r) >0:
            self.r = r
        else:
            self.r = get_r(self.vars,dims)
        # set dims:
        self.dims = [dims[ind][1] for ind in self.r]
        # set val:
        if type(val) == str:
            if val == 'rnd':
                self.val = np.random.rand(*self.dims)
            elif val == 'unif':
                self.val = np.ones(self.dims)
            else:
                self.val = 0
        else:
            self.val = val

    def parse_name(self,name):
        if name.count('f') == 1:
            vars = name[2:-1].split(',')
        return vars

    def shape(self):
        return np.shape(self.val)

    def get_name(self):
        return "f({})".format(','.join(self.vars))

    def __str__(self):
        return self.get_name()

    def __mul__(self,other):
        if isinstance(other, (int,float)):
            return func(val=other*self.val,vars=self.vars,r=self.r,parse_name=False)
        else:
            vars = list(set(self.vars + other.vars))
            r = sorted(list(set(self.r+other.r)))
            result = np.einsum(other.val,other.r,self.val,self.r,r)
            return func(val=result,vars=vars,r=r,parse_name=False)

    def __rmul__(self,other):
        if isinstance(other, (int,float)):
            return func(val=other*self.val,vars=self.vars,r=self.r,parse_name=False)

    def __div__(self,other):
        if isinstance(other, (int,float)):
            return func(val=self.val/float(other),vars=self.vars,r=self.r,parse_name=False)
        else:
            r = sorted(list(set(self.r+other.r)))
            vars = list(set(self.vars + other.vars))
            result = np.einsum(1.0/(other.val+eps),other.r,self.val,self.r,r)
            return func(val=result,vars=vars,r=r,parse_name=False)

    def __rdiv__(self,other):
        if isinstance(other, (int,float)):
            return func(val=float(other)/self.val,vars=self.vars,r=self.r,parse_name=False)

    def __pos__(self):
        return self

    def __neg__(self):
        return (-1)*self

    def __add__(self,other):
        if isinstance(other, (int,float)):
            return func(val=self.val+other,vars=self.vars,r=self.r,parse_name=False)
        else:
            vars = list(set(self.vars+other.vars))
            r = sorted(list(set(self.r+other.r)))
            I = np.ones([dims[ind][1] for ind in r])
            s0 = np.einsum(I,r,self.val,self.r,r)
            s1 = np.einsum(I,r,other.val,other.r,r)
            return func(val=s0+s1,vars=vars,r=r,parse_name=False)

    def __radd__(self,other):
        if isinstance(other, (int,float)):
            return func(val=self.val+other,vars=self.vars,r=self.r,parse_name=False)

    def __sub__(self,other):
        return self+(-other)

    def __rsub__(self,other):
        return other+(-self)

    def __pow__(self,y):
        return func(val=self.val**y,vars=self.vars,r=self.r,parse_name=False)

    # for python3:
    def __truediv__(self,other):
        return self.__div__(other)
    def __rtruediv__(self,other):
        return self.__rdiv__(other)

    def normalize(self,vrs=[]):
        if len(vrs) == 0:
            vrs = self.vars
        r_vrs = get_r(vrs,dims)
        r_Z = [rval for rval in self.r if not(rval in r_vrs)]
        Z = np.einsum(self.val,self.r,r_Z)
        self.val = np.einsum(self.val,self.r,1.0/(Z+eps),r_Z,self.r)
        return self

    def eval(self,variable,value):
        vars = [var for var in self.vars if not(var==variable)]
        r_var = get_r([variable],dims)[0]
        r_new = [r for r in self.r if not(r==r_var)]
        t = [slice(0,dims[ind][1]) for ind in range(0,len(dims))]
        t[r_var] = value
        t = [t[ind] for ind in self.r]
        t = tuple(t)
        return func(val=self.val[t],vars=vars,r=r_new,parse_name=False)
