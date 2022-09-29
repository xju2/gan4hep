import ROOT as root
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pylorentz import Momentum4
import os
import pandas as pd
import seaborn as sns

import atlasplots as aplt

# 2.7 for phi cuts

def selection_cuts(predictions, truths):
    predictions_cut = []
    predictions_cut = predictions[(-1 * 2.7 < predictions[:, 1]) & (predictions[:, 1] < 1 * 2.7)]
    predictions_cut = predictions_cut[(-1 * np.pi < predictions_cut[:, 2]) & (predictions_cut[:, 2] < 1 * np.pi)]
    predictions_cut = predictions_cut[(-1 * 2.7 < predictions_cut[:, 4]) & (predictions_cut[:, 4] < 1 * 2.7)]
    predictions_cut = predictions_cut[(-1 * np.pi < predictions_cut[:, 5]) & (predictions_cut[:, 5] < 1 * np.pi)]

    print('Length of predicted data set post-cuts: ', len(predictions_cut))
    print('Fraction of events lost : ', 1 - len(predictions_cut) / len(predictions))
    select_cut_val = 1 - len(predictions_cut) / len(predictions)

    # predictions=np.c_[ predictions ]
    truths_cut = truths[(-2.7 < truths[:, 1]) & (truths[:, 1] < 2.7)]
    truths_cut = truths_cut[(-np.pi < truths_cut[:, 2]) & (truths_cut[:, 2] < np.pi)]
    truths_cut = truths_cut[(-2.7 < truths_cut[:, 4]) & (truths_cut[:, 4] < 2.7)]
    truths_cut = truths_cut[(-np.pi < truths_cut[:, 5]) & (truths_cut[:, 5] < np.pi)]

    print('Length of true data set post-cuts: ', len(truths_cut))
    print('Fraction of events lost : ', 1 - len(truths_cut) / len(truths))
    select_cut_val_truths = 1 - len(truths_cut) / len(truths)

    return predictions_cut, truths_cut, predictions, truths, select_cut_val


def dimuon_calc(truths,jetnum):
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

    if jetnum==1:
        jet1_pt_true=np.array(truths[:, 6]).flatten()
        jet1_eta_true =np.array(truths[:, 7]).flatten()
        jet1_phi_true =np.array(truths[:, 8]).flatten()

    if jetnum == 2:
        jet1_pt_true=np.array(truths[:, 6]).flatten()
        jet1_eta_true =np.array(truths[:, 7]).flatten()
        jet1_phi_true =np.array(truths[:, 8]).flatten()
        jet2_pt_true=np.array(truths[:, 9]).flatten()
        jet2_eta_true =np.array(truths[:, 10]).flatten()
        jet2_phi_true =np.array(truths[:, 11]).flatten()

    # Create lists for 4 vector values
    muon_lead_true = []
    muon_sub_true = []
    parent_true = []
    jet1_true=[]
    jet2_true=[]
    dijet_true=[]
    # Create lists for invarient mass values
    mass_true = []
    pt_comb_true = []
    pseudo_true = []
    eta_angle_btwn_true = []
    phi_angle_btwn_true = []
    cos_theta_true = []
    phi_seperation_true = []
    phi_seperation_true_2jet = []
    dijet_pt_true=[]
    dijet_mass_true=[]
    dijet_pseudo_true=[]
    dijet_dimuon_seperation_true=[]



    for i in range(len(truths)):
        # Use pylorentz to define 4 momentum arrays for each event
        muon_lead_true.append(
            Momentum4.m_eta_phi_pt(masses_lead[i], etas_lead_true[i], phis_lead_true[i], pts_lead_true[i]))
        muon_sub_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], etas_sub_true[i], phis_sub_true[i], pts_sub_true[i]))

        if jetnum == 1:
            #WARNING DONT KNOW MASS OF JETT SO PLACEHOLDER AT THE MOMENT
            # Calculate Jet1 4 Vector
            jet1_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], jet1_eta_true[i], jet1_phi_true[i], jet1_pt_true[i]))

        if jetnum == 2:
            #WARNING DONT KNOW MASS OF JETT SO PLACEHOLDER AT THE MOMENT
            # Calculate Jet1 4 Vector
            jet1_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], jet1_eta_true[i], jet1_phi_true[i], jet1_pt_true[i]))

            jet2_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], jet2_eta_true[i], jet2_phi_true[i], jet2_pt_true[i]))

            dijet_true.append(jet1_true[i] + jet2_true[i])
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

        # Retrieve phi between the muons
        phi_angle_btwn_true.append(muon_lead_true[i].phi - muon_sub_true[i].phi)
        if jetnum == 1:
        #azimuthal seperation between muons and jets
            phi_seperation_true.append((muon_lead_true[i].phi - muon_sub_true[i].phi) - jet1_true[i].phi)

            pt_jet1_true.append(jet1_true_true[i].p_t)
        if jetnum == 2:
            # azimuthal seperation between muons and jets
            phi_seperation_true.append((muon_lead_true[i].phi - muon_sub_true[i].phi) - jet1_true[i].phi)
            # azimuthal seperation between muons and jets
            phi_seperation_true_2jet.append((muon_lead_true[i].phi - muon_sub_true[i].phi) - jet2_true[i].phi)

            dijet_pt_true.append(dijet_true[i].p_t)

            dijet_pseudo_true.append(dijet_true[i].eta)

            dijet_dimuon_seperation_true.append(dijet_true[i].phi-parent_true[i].phi)
        # Calculate P12+-

        P1_pos = (muon_lead_true[i].e + muon_lead_true[i].p_z) / np.sqrt(2)
        P1_neg = (muon_lead_true[i].e - muon_lead_true[i].p_z) / np.sqrt(2)
        P2_pos = (muon_sub_true[i].e + muon_sub_true[i].p_z) / np.sqrt(2)
        P2_neg = (muon_sub_true[i].e - muon_sub_true[i].p_z) / np.sqrt(2)

        # Calculate Cos

        cos_theta = 2 * (P1_pos * P2_neg - P1_neg * P2_pos)
        cos_theta = cos_theta / (
            np.sqrt((parent_true[i].m ** 2) * ((parent_true[i].m ** 2) + (parent_true[i].p_t ** 2))))
        cos_theta = cos_theta * (parent_true[i].p_z / np.abs(parent_true[i].p_z))

        cos_theta_true.append(cos_theta)

    # Add mass arrays from each batch?
    mass_true = np.concatenate(mass_true, axis=0)

    # Add to original dataset
    truths = np.column_stack((truths, mass_true))
    truths = np.column_stack((truths, pt_comb_true))
    truths = np.column_stack((truths, pseudo_true))
    truths = np.column_stack((truths, cos_theta_true))
    # truths = np.column_stack((truths, eta_angle_btwn_true))
    truths = np.column_stack((truths, phi_angle_btwn_true))
    if jetnum==1:
        truths = np.column_stack((truths, phi_seperation_true))
    if jetnum==2:
        truths = np.column_stack((truths, phi_seperation_true))
        truths = np.column_stack((truths, phi_seperation_true_2jet))
        truths = np.column_stack((truths, dijet_pt_true))
        truths = np.column_stack((truths, dijet_pseudo_true))
        truths = np.column_stack((truths, dijet_dimuon_seperation_true))
    return truths

def main(truths,predictions,w_list,loss_list,new_run_folder,jetnum,xlabels):

    # Function to calculate invarient dimuon mass + calculated variables
    truths = dimuon_calc( truths,jetnum)
    predictions = dimuon_calc(predictions,jetnum)

    # Applying Selection Cuts
    print('Length of predicted data set pre-cuts: ', len(predictions))
    print('Length of truth data set pre-cuts: ', len(truths))
    predictions_cut, truths, predictions, truths_cut, select_cut_val = selection_cuts(predictions, truths)


    # Create temporary folder for using the plotticng program
    if os.path.exists('Temp_Data') == False:
        os.mkdir('Temp_Data')

    # Save data to run data folder
    np.save(os.path.join(new_run_folder, 'truths_calc.npy'), truths)
    np.save(os.path.join(new_run_folder, 'predictions_calc.npy'), truths)
    np.save(os.path.join(new_run_folder, 'loss_list.npy'), loss_list)
    np.save(os.path.join(new_run_folder, 'w_list.npy'), w_list)

    # Save to temporary folder
    np.save(os.path.join('Temp_Data', 'truths_calc.npy'), truths)
    np.save(os.path.join('Temp_Data', 'predictions_calc.npy'), predictions)
    np.save(os.path.join('Temp_Data', 'loss_list.npy'), loss_list)
    np.save(os.path.join('Temp_Data', 'w_list.npy'), w_list)

    print('Number of True Events: ', len(truths))
    print('Number of Generated Events: ', len(predictions))
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Normalizing Flow')
    add_arg = parser.add_argument
    add_arg('--jetnum', default=0, help='Number of Jets in File')
    args = parser.parse_args()
    jetnum=int(args.jetnum)



    truths = np.load('Temp_Data/truths.npy')
    predictions = np.load('Temp_Data/predictions.npy')
    w_list = np.load('Temp_Data/w_list.npy')
    loss_list = np.load('Temp_Data/loss_list.npy')
    f = open("Temp_Data/filename.txt", "r")


    xlabel_0jet = ['leading Muon pT', 'leading Muon eta', 'leading Muon phi', 'subleading Muon pT',
                   'subleading Muon eta', 'subleading Muon phi']
    xlabel_1jet = ['Jet 1 pT', 'Jet 1 eta', 'Jet 1 phi']
    xlabel_2jet = ['Jet 2 pT', 'Jet 2 eta', 'Jet 2 phi']
    xlabel_3jet = ['Jet 3 pT', 'Jet 3 eta', 'Jet 3 phi']
    xlabel_4jet = ['Jet 4 pT', 'Jet 4 eta', 'Jet 4 phi']



    if jetnum == 0:
        xlabels = xlabel_0jet
    elif jetnum == 1:
        xlabels = xlabel_0jet+xlabel_1jet
        print(xlabels)
    elif jetnum == 2:
        xlabels = xlabel_0jet+xlabel_1jet+xlabel_2jet
    elif jetnum == 3:
        xlabels = xlabel_0jet+xlabel_1jet+xlabel_2jet+xlabel_3jet
    elif jetnum == 4:
        xlabels = xlabel_0jet+xlabel_1jet+xlabel_2jet+xlabel_3jet+xlabel_4jet
    else:
        xlabels = xlabel_0jet

    if f.mode == 'r':
        new_run_folder = f.read()

    #Print Length of datasets
    print('Number of True Events: ', len(truths))
    print('Number of Generated Events: ', len(predictions))

    #Plot the remaining graphs
    main(truths,predictions,w_list,loss_list,new_run_folder,jetnum,xlabels)


