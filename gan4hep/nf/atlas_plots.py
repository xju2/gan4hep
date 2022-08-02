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
    # predictions=np.c_[ predictions ]

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





def end_of_run_plots(w_list,loss_list):

    #Plot Log loss
    plt.plot(loss_list)
    plt.ylabel('Training Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join('Plots', 'logloss.png'))
    plt.clf()
    #Plot Wasserstein Distance
    plt.plot(w_list)
    plt.ylabel('Wasserstein Distance')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join('Plots', 'wasserstein.png'))

    



def dimuon_calc(predictions, truths):
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
    cos_theta_true = []

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
    # truths = np.column_stack((truths, phi_angle_btwn_true))

    # For Predictions
    masses_lead = np.full((len(predictions[:, 0]), 1), m_u)
    masses_sub = np.full((len(predictions[:, 0]), 1), m_u)

    # Take each column from the tru and generated data and rename to their parameter type
    pts_lead_true = np.array(predictions[:, 0]).flatten()
    etas_lead_true = np.array(predictions[:, 1]).flatten()
    phis_lead_true = np.array(predictions[:, 2]).flatten()
    pts_sub_true = np.array(predictions[:, 3]).flatten()
    etas_sub_true = np.array(predictions[:, 4]).flatten()
    phis_sub_true = np.array(predictions[:, 5]).flatten()

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
    cos_theta_true = []

    for i in range(len(predictions)):
        # Use pylorentz to define 4 momentum arrays for each event
        muon_lead_true.append(
            Momentum4.m_eta_phi_pt(masses_lead[i], etas_lead_true[i], phis_lead_true[i], pts_lead_true[i]))
        muon_sub_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], etas_sub_true[i], phis_sub_true[i], pts_sub_true[i]))

        # Calculate the Higgs boson 4 vector
        parent_true.append(muon_lead_true[i] + muon_sub_true[i])
        # print(parent_true)
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

    # print(pt_comb_true)

    # Add mass arrays from each batch?
    mass_true = np.concatenate(mass_true, axis=0)

    # Add to original dataset
    predictions = np.column_stack((predictions, mass_true))
    predictions = np.column_stack((predictions, pt_comb_true))
    predictions = np.column_stack((predictions, pseudo_true))
    predictions = np.column_stack((predictions, cos_theta_true))
    # predictions = np.column_stack((predictions, eta_angle_btwn_true))
    # predictions = np.column_stack((predictions, phi_angle_btwn_true))

    return truths, predictions


def Plotting(truths_cut, predictions_cut, col_num, lower_range, upper_range, title):

    # Set the ATLAS Style
    aplt.set_atlas_style()

    # Create a figure and axes
    fig, (ax1, ax2) = aplt.ratio_plot(name="fig1", figsize=(800, 800), hspace=0.05)

    # Define a distribution
    sqroot = root.TF1("sqroot", "x*gaus(0) + [3]*abs(sin(x)/x)", 0, 10)
    sqroot.SetParameters(10, 4, 1, 20)

    # Randomly fill two histograms according to the above distribution
    hist1 = root.TH1F("truths", "Random Histogram 1", 40, lower_range, upper_range)

    for j in range(len(truths_cut)):
        hist1.Fill(truths_cut[j, col_num])

    # sqroot.SetParameters(10, 4, 1.1, 20)
    hist2 = root.TH1F("predictions", "Random Histogram 2", 40, lower_range, upper_range)

    for j in range(len(predictions_cut)):
        hist2.Fill(predictions_cut[j, col_num])

    # Normalisation
    hist1 = (hist1.Clone("hist1"));
    hist1.Scale(1. / hist1.Integral(), "width");

    hist2 = (hist2.Clone("hist2"));
    hist2.Scale(1. / hist2.Integral(), "width");

    # hist1.Fill(predictions_cut[j,0])

    # Draw the histograms on these axes
    ax1.plot(hist1, linecolor=root.kRed + 1, label="Truth", labelfmt="L")
    ax1.plot(hist2, linecolor=root.kBlue + 1, label="Generated", labelfmt="L")

    # Change to log scale
    # ax1.set_yscale('log')

    # Draw line at y=1 in ratio panel
    line = root.TLine(ax1.get_xlim()[0], 1, ax1.get_xlim()[1], 1)
    ax2.plot(line)

    # Calculate and draw the ratio
    ratio_hist = hist1.Clone("ratio_hist")
    ratio_hist.Divide(hist2)

    for i in range(1, ratio_hist.GetNbinsX()):
        ratio_hist.SetBinError(i, ratio_hist.GetBinError(i))

    ax2.plot(ratio_hist, "EP X0", linewidth=1)

    # Add extra space at top of plot to make room for labels
    ax1.add_margins(top=0.16, bottom=0.16)

    # Set axis titles

    ax1.set_ylabel("Normalised Events ", labelsize=20)
    ax2.set_ylabel("Truth / Generated", loc="top", labelsize=20)

    if col_num == 0 or col_num == 3 or col_num == 6 or col_num == 7:
        ax2.set_xlabel(title + " [GeV]", labelsize=20)
    else:
        ax2.set_xlabel(title + " [Radians]", labelsize=20)

    # Add extra space at top and bottom of ratio panel
    ax2.add_margins(top=0.1, bottom=0.1)

    # Go back to top axes to add labels
    ax1.cd()

    # Add the ATLAS Label
    # aplt.atlas_label(text="Internal", loc="upper left")
    aplt.atlas_label(text='Simulated Data', loc='upper left')

    # Add legend
    ax1.legend(loc=(0.70, 0.78, 1, 0.90))

    # Save the plot as a PDF
    fig.savefig("Plots/ratio" + str(col_num) + ".pdf")

    hist1.Draw()
    hist2.Draw()

def main(truths,predictions,w_list,loss_list,new_run_folder,xlabels):
    xlabels_extra = xlabels
    xlabels_extra.append('DiMuon Invarient Mass')
    xlabels_extra.append('DiMuon P_t')
    xlabels_extra.append('DiMuon Pseudorapidity (Eta)')
    xlabels_extra.append('Cos theta *')
    # xlabels=xlabels[:-3]
    # New Code
    num_of_variables = len(xlabels)
    # Function to calculate invarient dimuon mass
    truths, predictions = dimuon_calc(predictions, truths)
    # Applying Selection Cuts

    print('Length of predicted data set pre-cuts: ', len(predictions))
    print('Length of truth data set pre-cuts: ', len(truths))
    predictions_cut, truths, predictions, truths_cut, select_cut_val = selection_cuts(predictions, truths)

    # predictions_cut=predictions
    # truths_cut=truths
    # truths_cut=truths_cut[:len(predictions_cut)]

    pt_lead = [0, 0, 250]
    pt_sub = [3, 0, 250]
    eta_lead = [1, -2.7, 2.7]
    eta_sub = [4, -2.7, 2.7]
    phi_lead = [2, -3.4, 3.4]
    phi_sub = [5, -3.4, 3.4]
    dimuon_mass = [6, 110, 160]
    pt_dimuon = [7, 0, 130]
    dimuon_eta = [8, -5, 5]
    cos_theta = [9, -1, 1]

    var_name = ['leading Muon pT', 'leading Muon eta', 'leading Muon phi', 'subleading Muon pT', 'subleading Muon eta',
                'subleading Muon phi', 'Dimuon Mass', 'Pt Dimuon', 'Dimuon eta', 'Cos theta *']
    var_list = [pt_lead, eta_lead, phi_lead, pt_sub, eta_sub, phi_sub, dimuon_mass, pt_dimuon, dimuon_eta, cos_theta]
    from IPython.display import IFrame

    for i in range(len(var_list)):
        Plotting(truths_cut, predictions_cut, var_list[i][0], var_list[i][1], var_list[i][2], var_name[i])

if __name__ == '__main__':

    truths = np.load('Temp_Data/truths.npy')
    predictions = np.load('Temp_Data/predictions.npy')
    w_list = np.load('Temp_Data/w_list.npy')
    loss_list = np.load('Temp_Data/loss_list.npy')
    f = open("Temp_Data/filename.txt", "r")

    if f.mode == 'r':
        new_run_folder = f.read()

    xlabels = ['leading Muon pT', 'leading Muon eta', 'leading Muon phi', 'subleading Muon pT', 'subleading Muon eta',
                'subleading Muon phi']

    os.makedirs('Plots', exist_ok=True)



    # Plot loss and wasserstein plots
    end_of_run_plots(w_list,loss_list)


    print('Number of True Events: ', len(truths))
    print('Number of Generated Events: ', len(predictions))


    main(truths,predictions,w_list,loss_list,new_run_folder,xlabels)
