# Import package and module
import numpy as np
import utility as util
import tools


def EGM(sol,t,par):
    #sol = EGM_loop(sol,t,par) 
    sol = EGM_vec(sol,t,par) 
    return sol

def EGM_loop (sol,t,par):
    for i_a,a in enumerate(par.grid_a[t,:]):

        # Fill in:
        # Hint: Same procedure as in 02_EGM.ipynb
        interp = lambda m: tools.interp_linear_1d(sol.m[t+1,:], sol.c[t+1,:], m)
        
        # Future m and c
        if t+1<= par.Tr: # No pension in the next period
            fac = par.G*par.L[t]*par.psi_vec # Trick to ease notation and calculations

        else:
            fac = par.G*par.L[t]

        m_next = fac*par.R*a + par.xi_vec
        
        # Future expected marginal utility
        mu_next = util.marg_util(m_next)

        # Current C and m
        #sol.c[t,i_a+1]=
        #sol.m[t,i_a+1]=

    return sol

def EGM_vec (sol,t,par): 
    
    interp = lambda m: tools.interp_linear_1d(sol.m[t+1,:], sol.c[t+1,:], m)

    
    if t+1<= par.Tr: # No pension in the next period
        fac = par.G*par.L[t]*par.psi_vec[:,np.newaxis] # Trick to ease notation and calculations

    else:
        fac = par.G*par.L[t]


    m_next = ((1/fac)*par.R*par.grid_a[t, np.newaxis] + par.xi_vec[:,np.newaxis]).ravel()
    c_next = interp(m_next) # something is wrong here :)
    c_next_adj = c_next/(np.tile(fac, par.Na).ravel())

    ws = (np.tile(par.w, par.Na))
    mu_next = util.marg_util(c_next_adj, par)
    Emu_next = (ws*mu_next).reshape(par.Nshocks,par.Na).sum(axis=0)
    c = util.inv_marg_util(Emu_next, par)
    

    sol.c[t, 1:] = c
    sol.m[t, 1:] = par.grid_a[t] - c

    return sol
