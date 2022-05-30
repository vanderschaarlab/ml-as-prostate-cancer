_EPSILON = 1e-08

import numpy as np
import pandas as pd
import random
import os,sys

def predict_cpg(psa, grade, stage, is_external=False):
    cpg   = -1. * np.ones_like(psa)
    '''
    Notes:
        - There were some cases that are predicted as cpg 3. This should not happen because based on our dataset extraction process.
        - So, here we only focus on cgp 1 and cpg 2. Anything that does not belong to both will be cpg 3 (which should not happen here)

    '''#
    #there were some cases that are predicted as cpg3 -- which should not happend based on the extraction criterion of the dataset. So, here we remove all the cases

    if is_external: #remove grade information -- not available for external 
        idx1  = (psa < 10.)&(stage <=1.)
        idx2  = (((psa >= 10.)&(psa <= 20.)))&(stage <= 1.)
        idx3  = (~idx1)&(~idx2)
    #     idx3  = (((psa >= 10.)&(psa <= 20.))&(stage <= 1.)) #|((grade == 3.)&(stage <= 1.))
    #     idx4  = (psa > 20.)|(stage == 2.)
    else: 
        idx1  = (grade == 1.)&(psa < 10.)&(stage <=1.)
        idx2  = ((grade == 2.)|((psa >= 10.)&(psa <= 20.)))&(stage <= 1.)
        idx3  = (~idx1)&(~idx2)
    #     idx3  = ((grade == 2.)&((psa >= 10.)&(psa <= 20.))&(stage <= 1.))|((grade == 3.)&(stage <= 1.))
    #     idx4  = (grade == 3.)|(psa > 20.)|(stage == 2.)    

    cpg[idx1] = 1.
    cpg[idx2] = 2.
    cpg[idx3] = 3.
    
    return cpg