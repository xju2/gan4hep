#Code to add an extra invariant mumu column to the muon output file, to make this code run you need to have installed pylorentz via pip install and to enter a terminal input similar to: ln -s /eos/user/p/pfitzhug/GANWork/gan4hep/gan4hep/mumu_Invariant_Mass.py to create a copy of the .py file in gan_work


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pylorentz import Momentum4


#Reads dataset
def read_dataframe(filename, sep=",", engine=None):
    if type(filename) == list:
        print(filename)
        df_list = [
            pd.read_csv(f, sep=sep, header=None, names=None, engine=engine)
                for f in filename
        ]
        df = pd.concat(df_list, ignore_index=True)
        filename = filename[0]
    else:
        df = pd.read_csv(filename, sep=sep, 
                    header=None, names=None, engine=engine)
    return df

#Reads and shuffles dataset
def run_file(filename):
    df = read_dataframe(filename, " ", None)
    truth_data = df.to_numpy()

    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    np_rs = RandomState(MT19937(SeedSequence(123456789)))
    np_rs.shuffle(truth_data)

    #Calculates invariant mass
    truth_data=mumu_invariant_mass(truth_data)

    
def mumu_invariant_mass(truths):
    
    #print(truths)
    #Define muon mass and create two arrays filled with this value
    m_u=1.883531627e-28
    masses_lead = np.full((len(truths[:,0]), 1), m_u)
    masses_sub = np.full((len(truths[:,0]), 1), m_u)
    
    #Take each column from the tru and generated data and rename to their parameter type
    pts_lead_true = np.array(truths[:, 0]).flatten()
    etas_lead_true = np.array(truths[:, 1]).flatten()
    phis_lead_true = np.array(truths[:, 2]).flatten()
    pts_sub_true = np.array(truths[:, 3]).flatten()
    etas_sub_true =np.array(truths[:, 4]).flatten()
    phis_sub_true = np.array(truths[:, 5]).flatten()

    #Create lists for 4 vector values
    muon_lead_true=[]
    muon_sub_true=[]
    parent_true=[]
    
    #Create lists for invarient mass values
    mass_true=[]

    for i in range(len(truths)):
        #Use pylorentz to define 4 momentum arrays for each event
        muon_lead_true.append(Momentum4.m_eta_phi_pt(masses_lead[i], etas_lead_true[i], phis_lead_true[i], pts_lead_true[i]))
        muon_sub_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], etas_sub_true[i], phis_sub_true[i], pts_sub_true[i]))

        #Calculate the Higgs boson 4 vector
        parent_true.append(muon_lead_true[i] + muon_sub_true[i])

        #Retrieve the Higgs Mass
        mass_true.append(parent_true[i].m)

    #Add mass arrays from each batch?    
    mass_true=np.concatenate( mass_true, axis=0 )

    #Add to original dataset
    truths = np.column_stack((truths, mass_true))

    #Plots the invariant mass distrubution
    plt.hist(mass_true,  bins=80,  label='Truth',range=[0,200])
    plt.show()
    
    #Save to file
    np.savetxt("/eos/user/p/pfitzhug/GANWork/gan_work/muon.txt", truths)
    return truths


if __name__ == '__main__': #Code only runs if its called by the terminal and not by another .py file
    
    
    # How to enter muon .output file
    
    import argparse
    parser = argparse.ArgumentParser(description='Calculate Invariant Mass')
    add_arg = parser.add_argument
    add_arg("filename", help='input filename', default=None, nargs='+')
    
    args = parser.parse_args()

    from tensorflow.compat.v1 import logging
    
    run_file(args.filename)