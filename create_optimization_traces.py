# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:35:34 2021

@author: Nils
"""

print("what?")

from multiprocessing import Pool


import os, sys, shutil
from optparse import OptionParser
sys.path.append(os.environ['SU2_RUN'])
import SU2

from numpy import random
import numpy as np

from torch.quasirandom import SobolEngine

import copy

import pickle

#import cProfile

# -------------------------------------------------------------------
#  Main 
# -------------------------------------------------------------------

def main():
    
    print("hello0")
    
    parser=OptionParser()
    parser.add_option("-f", "--file", dest="filename",
                      help="read config from FILE", metavar="FILE",
                      default="turb_SA_RAE2822.cfg")
    parser.add_option("-r", "--name", dest="projectname", default='',
                      help="try to restart from project file NAME", metavar="NAME")
    parser.add_option("-n", "--partitions", dest="partitions", default=1,
                      help="number of PARTITIONS", metavar="PARTITIONS")
    parser.add_option("-g", "--gradient", dest="gradient", #default="DISCRETE_ADJOINT",
                      help="Method for computing the GRADIENT (CONTINUOUS_ADJOINT, DISCRETE_ADJOINT, FINDIFF, NONE)", metavar="GRADIENT",
                      default="CONTINUOUS_ADJOINT")
    parser.add_option("-o", "--optimization", dest="optimization", default="SLSQP",
                      help="OPTIMIZATION techique (SLSQP, CG, BFGS, POWELL)", metavar="OPTIMIZATION")
    parser.add_option("-q", "--quiet", dest="quiet", default="True",
                      help="True/False Quiet all SU2 output (optimizer output only)", metavar="QUIET")
    parser.add_option("-z", "--zones", dest="nzones", default="1",
                      help="Number of Zones", metavar="ZONES")


    (options, args)=parser.parse_args()
    
    # process inputs
    options.partitions  = int( options.partitions )
    options.quiet       = options.quiet.upper() == 'TRUE'
    options.gradient    = options.gradient.upper()
    options.nzones      = int( options.nzones )
    
    
    print("hello")
    print(f"filename: {options.filename}")
    
    
    fname_mesh = "mesh_RAE2822_turb.su2"
    fname_cfg = "turb_SA_RAE2822.cfg"
    
    s_cfg = ""
    s_mesh = ""
     
    with open(fname_cfg, "rb") as f_cfg:
        s_cfg = f_cfg.read()
    
    with open(fname_mesh, "rb") as f_mesh:
        s_mesh = f_mesh.read()
    
    
    config = SU2.io.Config(options.filename)
    config.NUMBER_PART = options.partitions
    config.NZONES      = int( options.nzones )
    if options.quiet: config.CONSOLE = 'CONCISE'
    config.GRADIENT_METHOD = options.gradient
    
    #state = SU2.io.State()
    #state.find_files(config)
    
    
    #project_name = "opt_*"
    
    #project = SU2.opt.Project(config, state, folder="opt_*")
    
    
    ######
    
    #n_dv = len( project.config['DEFINITION_DV']['KIND'] )
    #project.n_dv = n_dv
    
    soboleng = SobolEngine(41, True)

    
    n_samples = 1000
    
    scale = 1.e-04
    X_draw = soboleng.draw(n_samples).numpy()
    
    
    #We can consider 3sigma around the mean, so that the ranges are:
    #Mach* 0.734+-0.0015 [-]
    #AoA: 2.79+-0.3 [deg.] 
    
    Ma_upper = 0.734+0.0015
    Ma_lower = 0.734-0.0015
    
    alpha_upper = 2.79+0.3
    alpha_lower = 2.79-0.3
    
    Re_upper = 7_000_000
    Re_lower = 6_000_000
    
    Ma_mat =  X_draw[:,0] * (Ma_upper-Ma_lower) + Ma_lower
    AOA_mat = X_draw[:,1] * (alpha_upper-alpha_lower) + alpha_lower
    Re_mat =  X_draw[:,2] * (Re_upper-Re_lower) + Re_lower
    
    # X_draw = random.rand(n_samples, n_dv+2)
    dv_mat = scale * (X_draw[:,3:] - 0.5)
    
    
    
    inputs = []
    dv_list = []
    # Ma_list = []
    # alpha_list
    for i in range(n_samples):
        dvs = dv_mat[i].tolist()
        dv_list.append(dvs)
        # alpha = double(X[i,1])
        # Ma = double(X[i,0])
        iconfig = copy.deepcopy(config)
        iconfig.MACH_NUMBER = Ma_mat[i]
        iconfig.AOA = AOA_mat[i]
        iconfig.REYNOLDS_NUMBER = Re_mat[i]
        iconfig.DV_VALUE_OLD = dvs
        
        inputs.append({"i": i, "dvs": dvs, "config": iconfig, "its": 3, "accu": 1.e-10, "bound_upper": 0.5*scale, "bound_lower": -0.5*scale, "project_name": "opt_{:04d}".format(i)})
    
    print(dv_list[:3])
    pickle.dump(dv_list, open("dv_list.p", "wb"))
    pickle.dump(Ma_mat, open("Ma_mat.p", "wb"))
    pickle.dump(AOA_mat, open("AOA_mat.p", "wb"))
    pickle.dump(Re_mat, open("Re_mat.p", "wb"))
    
    
    
    
    
    with Pool(processes=6) as pool:
        #outputs = pool.map(run_one_design, inputs)
        outputs = pool.map(start_opt, inputs, chunksize=1)
    
    print(outputs)
    
    
    
    
    return

def start_opt(inp):
    
    x0 = inp["dvs"]
    config = inp["config"]
    project_name = inp["project_name"]
    
    state = SU2.io.State()
    state.find_files(config)
    
    
    
    project = SU2.opt.Project(config, state, folder=project_name)
    #project.n_dv = n_dv
    
    
    
    its = inp["its"]
    accu = inp["accu"]
    bound_upper = inp["bound_upper"]
    bound_lower = inp["bound_lower"]
    
    relax_factor = 1
    n_dv = len(x0)
    
    xb_low            = [float(bound_lower)/float(relax_factor)]*n_dv      # lower dv bound it includes the line search acceleration factor
    xb_up             = [float(bound_upper)/float(relax_factor)]*n_dv      # upper dv bound it includes the line search acceleration fa
    xb                = list(zip(xb_low, xb_up))
    
    SU2.opt.BFGS(project,x0,xb,its,accu)
    
    
def shape_optimization( filename                           ,
                        projectname = ''                   ,
                        partitions  = 0                    ,
                        gradient    = 'CONTINUOUS_ADJOINT' ,
                        optimization = 'SLSQP'             ,
                        quiet       = False                ,
                        nzones      = 1                    ):
    # Config
    config = SU2.io.Config(filename)
    config.NUMBER_PART = partitions
    config.NZONES      = int( nzones )
    if quiet: config.CONSOLE = 'CONCISE'
    config.GRADIENT_METHOD = gradient
    
    its               = int ( config.OPT_ITERATIONS )                      # number of opt iterations
    bound_upper       = float ( config.OPT_BOUND_UPPER )                   # variable bound to be scaled by the line search
    bound_lower       = float ( config.OPT_BOUND_LOWER )                   # variable bound to be scaled by the line search
    relax_factor      = float ( config.OPT_RELAX_FACTOR )                  # line search scale
    gradient_factor   = float ( config.OPT_GRADIENT_FACTOR )               # objective function and gradient scale
    def_dv            = config.DEFINITION_DV                               # complete definition of the desing variable
    n_dv              = sum(def_dv['SIZE'])                                # number of design variables
    accu              = float ( config.OPT_ACCURACY ) * gradient_factor    # optimizer accuracy
    x0                = [0.0]*n_dv # initial design
    xb_low            = [float(bound_lower)/float(relax_factor)]*n_dv      # lower dv bound it includes the line search acceleration factor
    xb_up             = [float(bound_upper)/float(relax_factor)]*n_dv      # upper dv bound it includes the line search acceleration fa
    xb                = list(zip(xb_low, xb_up)) # design bounds
    
    # State
    state = SU2.io.State()
    state.find_files(config)

    # add restart files to state.FILES
    if config.get('TIME_DOMAIN', 'NO') == 'YES' and config.get('RESTART_SOL', 'NO') == 'YES' and gradient != 'CONTINUOUS_ADJOINT':
        restart_name = config['RESTART_FILENAME'].split('.')[0]
        restart_filename = restart_name + '_' + str(int(config['RESTART_ITER'])-1).zfill(5) + '.dat'
        if not os.path.isfile(restart_filename): # throw, if restart files does not exist
            sys.exit("Error: Restart file <" + restart_filename + "> not found.")
        state['FILES']['RESTART_FILE_1'] = restart_filename

        # use only, if time integration is second order
        if config.get('TIME_MARCHING', 'NO') == 'DUAL_TIME_STEPPING-2ND_ORDER':
            restart_filename = restart_name + '_' + str(int(config['RESTART_ITER'])-2).zfill(5) + '.dat'
            if not os.path.isfile(restart_filename): # throw, if restart files does not exist
                sys.exit("Error: Restart file <" + restart_filename + "> not found.")
            state['FILES']['RESTART_FILE_2'] =restart_filename


    # Project

    if os.path.exists(projectname):
        project = SU2.io.load_data(projectname)
        project.config = config
    else:
        project = SU2.opt.Project(config,state)

    # Optimize
    if optimization == 'SLSQP':
      SU2.opt.SLSQP(project,x0,xb,its,accu)
    if optimization == 'CG':
      SU2.opt.CG(project,x0,xb,its,accu)
    if optimization == 'BFGS':
      SU2.opt.BFGS(project,x0,xb,its,accu)
    if optimization == 'POWELL':
      SU2.opt.POWELL(project,x0,xb,its,accu)


    # rename project file
    if projectname:
        shutil.move('project.pkl',projectname)
    
    return project
    
    
    
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    