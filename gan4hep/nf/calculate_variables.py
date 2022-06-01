import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



from pylorentz import Momentum4

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



def calc_extra_var(truths):
    # print(truths)
    # Define muon mass and create two arrays filled with this value

    m_u = 1.883531627e-28

    masses_lead = np.full((len(truths[:, 0]), 1), m_u)
    masses_sub = np.full((len(truths[:, 0]), 1), m_u)

    # Take each column from the tru and generated data and rename to their parameter type
    pts_lead_true = np.array(truths[:, 0]).flatten()
    etas_lead_true = np.array(truths[:, 1]).flatten()
    phis_lead_true = np.array(truths[:, 2]).flatten()
    pts_sub_true = np.array(truths[:, 3]).flatten()
    etas_sub_true = np.array(truths[:, 4]).flatten()
    phis_sub_true = np.array(truths[:, 5]).flatten()

    # Create lists for 4 vector values
    muon_lead_true = []
    muon_sub_true = []
    parent_true = []

    # Create lists for invarient mass values
    mass_true = []
    pt_comb_true = []
    pseudo_true = []
    eta_angle_btwn_true = []
    phi_angle_btwn_true = []
    cos_theta_true=[]

    for i in range(len(truths)):
        # Use pylorentz to define 4 momentum arrays for each event
        muon_lead_true.append(
            Momentum4.m_eta_phi_pt(masses_lead[i], etas_lead_true[i], phis_lead_true[i], pts_lead_true[i]))
        muon_sub_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], etas_sub_true[i], phis_sub_true[i], pts_sub_true[i]))

        # Calculate the Higgs boson 4 vector
        parent_true.append(muon_lead_true[i] + muon_sub_true[i])

        # Retrieve the Higgs Mass
        mass_true.append(parent_true[i].m)

        # Retrive PT
        pt_comb_true.append(parent_true[i].p_t)

        # Retrieve Pseudorapidity
        pseudo_true.append(parent_true[i].eta)

        # Retrieve eta between the muons
        eta_angle_btwn_true.append(muon_lead_true[i].eta - muon_sub_true[i].eta)

        # Retrieve eta between the muons
        eta_angle_btwn_true.append(muon_lead_true[i].phi - muon_sub_true[i].phi)

        # Calculate cos theta

        #Calculate P12+-

        P1_pos=(muon_lead_true[i].e+muon_lead_true[i].p_z)/np.sqrt(2)
        P1_neg=(muon_lead_true[i].e-muon_lead_true[i].p_z)/np.sqrt(2)
        P2_pos=(muon_sub_true[i].e+muon_sub_true[i].p_z)/np.sqrt(2)
        P2_neg=(muon_sub_true[i].e-muon_sub_true[i].p_z)/np.sqrt(2)

        #Calculate Cos

        cos_theta=2*(P1_pos*P2_neg-P1_neg*P2_pos)
        cos_theta=cos_theta/(np.sqrt((parent_true[i].m**2)*((parent_true[i].m**2)+(parent_true[i].p_t**2))))
        cos_theta=cos_theta*(parent_true[i].p_z/np.abs(parent_true[i].p_z))

        cos_theta_true.append(cos_theta)

    # Add mass arrays from each batch?
    mass_true = np.concatenate(mass_true, axis=0)

    # Add to original dataset
    #truths = np.column_stack((truths, mass_true))
    truths = np.column_stack((truths, pt_comb_true))
    truths = np.column_stack((truths, pseudo_true))
    truths = np.column_stack((truths, cos_theta_true))
    # truths = np.column_stack((truths, eta_angle_btwn_true))
    # truths = np.column_stack((truths, phi_angle_btwn_true))

    #print('truths end of calc var function',truths)

    return truths

def load_data(filename, max_evts=1000000, testing_frac=0.1):

    #Read Data
    df = read_dataframe(filename, " ", None)
    df=df[:-1]
    truth_data_1 = df.to_numpy().astype(np.float32)
    print(f"reading dimuon {df.shape[0]} events from file {filename}")

    #Filter very high Pt values
    truth_data_1 = truth_data_1[truth_data_1[:, 0] < 1000]

    #For debugging
    truth_data_1=truth_data_1[:2000]

    #Calculate extra variables for training (mumu_pt,mumu_pseudo,costheta*)
    truth_data_1=calc_extra_var(truth_data_1)
    #print('truths end of dimuon func',truth_data_1)
    return truth_data_1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Normalizing Flow')
    add_arg = parser.add_argument
    add_arg('filename', help='Herwig input filename')

    add_arg("--max-evts", default=-1, type=int, help="maximum number of events")
    add_arg("--batch-size", type=int, default=512, help="batch size")
    add_arg("--multi", type=int, default=1, help="Number times more of generated events")
    add_arg("--data", default='dimuon_inclusive',
            choices=['herwig_angles', 'dimuon_inclusive', 'herwig_angles2'])


    args = parser.parse_args()


    #Reading Data
    full_data = load_data(
        args.filename, max_evts=args.max_evts)

   # Saving new data
    np.savetxt('calc_var_data.output', full_data, delimiter=' ')
