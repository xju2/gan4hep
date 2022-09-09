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


def dimuon_calc(truths):
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


    jet1_pt_true=np.array(truths[:, 6]).flatten()
    jet1_eta_true =np.array(truths[:, 7]).flatten()
    jet1_phi_true =np.array(truths[:, 8]).flatten()

    # Create lists for 4 vector values
    muon_lead_true = []
    muon_sub_true = []
    parent_true = []
    jet1_true=[]

    # Create lists for invarient mass values
    mass_true = []
    pt_comb_true = []
    pseudo_true = []
    eta_angle_btwn_true = []
    phi_angle_btwn_true = []
    cos_theta_true = []
    phi_seperation_true = []



    for i in range(len(truths)):
        # Use pylorentz to define 4 momentum arrays for each event
        muon_lead_true.append(
            Momentum4.m_eta_phi_pt(masses_lead[i], etas_lead_true[i], phis_lead_true[i], pts_lead_true[i]))
        muon_sub_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], etas_sub_true[i], phis_sub_true[i], pts_sub_true[i]))

        #WARNING DONT KNOW MASS OF JETT SO PLACEHOLDER AT THE MOMENT
        # Calculate Jet1 4 Vector
        jet1_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], jet1_eta_true[i], jet1_phi_true[i], jet1_pt_true[i]))

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

        #azimuthal seperation between muons and jets
        phi_seperation_true.append((muon_lead_true[i].phi - muon_sub_true[i].phi) - jet1_true[i].phi)

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
    truths = np.column_stack((truths, phi_seperation_true))

    return truths


def Plotting(truths_cut, predictions_cut, col_num, lower_range, upper_range, title,units):

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
    for i in range(len(predictions_cut[1,:])):
        if i==col_num:
            ax2.set_xlabel(title +'/ '+ units[i], labelsize=20)

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

def main(truths,predictions,w_list,loss_list,new_run_folder,jetnum,xlabels):

    # Function to calculate invarient dimuon mass + calculated variables
    truths = dimuon_calc( truths)
    predictions = dimuon_calc(predictions)

    # Applying Selection Cuts
    print('Length of predicted data set pre-cuts: ', len(predictions))
    print('Length of truth data set pre-cuts: ', len(truths))
    predictions_cut, truths, predictions, truths_cut, select_cut_val = selection_cuts(predictions, truths)


    #Original Variables
    pt_lead = [0, 0, 250]
    eta_lead = [1, -2.8, 2.8]
    phi_lead = [2, -3.4, 3.4]
    pt_sub = [3, 0, 250]
    eta_sub = [4, -2.8, 2.8]
    phi_sub = [5, -3.4, 3.4]

    #Jet 1 Variables
    jet1_pt = [6, 0, 80]
    jet1_eta = [7, -5, 5]
    jet1_phi = [8, -3.4, 3.4]

    #Jet 2 Variables
    jet2_pt = [9, 0, 80]
    jet2_eta = [10, -5, 5]
    jet2_phi = [11, -3.4, 3.4]

    #Jet 3 Variables
    jet3_pt = [12, 0, 80]
    jet3_eta = [13, -5, 5]
    jet3_phi = [14, -3.4, 3.4]

    #Jet 4 Variables
    jet4_pt = [15, 0, 80]
    jet4_eta = [16, -5, 5]
    jet4_phi = [17, -3.4, 3.4]


    xlabels_extra = xlabels
    xlabels_extra.append('DiMuon Invarient Mass')
    xlabels_extra.append('DiMuon P_t')
    xlabels_extra.append('DiMuon Pseudorapidity (Eta)')
    xlabels_extra.append('Cos theta *')
    xlabels_extra.append('Dimuon Phi')

    if jetnum==0:
        units = ['GeV,', '', 'Radians', 'GeV', '', 'Radians']
        units_calc = ['GeV', 'GeV', '', '', 'Radians']
        #Calc Variables
        dimuon_mass = [6, 0, 250]
        pt_dimuon = [7, 0, 250]
        dimuon_eta = [8, -5, 5]
        cos_theta = [9, -1, 1]
        dimuon_phi = [10, -5, 5]

        calc_var_ranges = [dimuon_mass, pt_dimuon, dimuon_eta, cos_theta, dimuon_phi]
    elif jetnum==1:
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet1)')
        units = ['GeV,', '', 'Radians', 'GeV', '', 'Radians']
        units_calc = ['GeV', 'GeV', '', '', 'Radians','']
        units_jet = ['GeV', '', 'Radians']
        #Calc Variables
        dimuon_mass = [9, 0, 250]
        pt_dimuon = [10, 0, 250]
        dimuon_eta = [11, -5, 5]
        cos_theta = [12, -1, 1]
        dimuon_phi = [13, -5, 5]
        jet1_sep = [14,-5, 5]
        calc_var_ranges = [dimuon_mass, pt_dimuon, dimuon_eta, cos_theta, dimuon_phi,jet1_sep]
    elif jetnum == 2:
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet1)')
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet2)')
        units = ['GeV,', '', 'Radians', 'GeV', '', 'Radians']
        units_calc = ['GeV', 'GeV', '', '', 'Radians', '','']
        units_jet = ['GeV', '', 'Radians']
        # Calc Variables
        dimuon_mass = [12, 0, 250]
        pt_dimuon = [13, 0, 250]
        dimuon_eta = [14, -5, 5]
        cos_theta = [15, -1, 1]
        dimuon_phi = [16, -5, 5]
        jet1_sep = [17, -5, 5]
        jet2_sep = [18, -5, 5]
        calc_var_ranges = [dimuon_mass, pt_dimuon, dimuon_eta, cos_theta, dimuon_phi, jet1_sep,jet2_sep]
    elif jetnum == 3:
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet1)')
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet2)')
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet3)')
        units = ['GeV,', '', 'Radians', 'GeV', '', 'Radians']
        units_calc = ['GeV', 'GeV', '', '', 'Radians', '','','']
        units_jet = ['GeV', '', 'Radians']
        # Calc Variables
        dimuon_mass = [15, 0, 250]
        pt_dimuon = [16, 0, 250]
        dimuon_eta = [17, -5, 5]
        cos_theta = [18, -1, 1]
        dimuon_phi = [19, -5, 5]
        jet1_sep = [20, -5, 5]
        jet2_sep = [21, -5, 5]
        jet3_sep = [22, -5, 5]
        calc_var_ranges = [dimuon_mass, pt_dimuon, dimuon_eta, cos_theta, dimuon_phi, jet1_sep,jet2,jet3_sep]
    elif jetnum == 4:
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet1)')
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet2)')
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet3)')
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet4)')
        units = ['GeV,', '', 'Radians', 'GeV', '', 'Radians']
        units_calc = ['GeV', 'GeV', '', '', 'Radians', '','','','']
        units_jet = ['GeV', '', 'Radians']
        # Calc Variables
        dimuon_mass = [18, 0, 250]
        pt_dimuon = [19, 0, 250]
        dimuon_eta = [20, -5, 5]
        cos_theta = [21, -1, 1]
        dimuon_phi = [22, -5, 5]
        jet1_sep = [23, -5, 5]
        jet2_sep = [24, -5, 5]
        jet3_sep = [25, -5, 5]
        jet4_sep = [26, -5, 5]
        calc_var_ranges = [dimuon_mass, pt_dimuon, dimuon_eta, cos_theta, dimuon_phi, jet1_sep,jet2_sep,jet3_sep,jet4_sep]
    else:
        xlabels_extra.append('Seperation Between Jet and Dimuon (Jet1)')
        units = ['GeV,', '', 'Radians', 'GeV', '', 'Radians']
        units_calc = ['GeV', 'GeV', '', '', 'Radians']
        # Calc Variables
        dimuon_mass = [6, 0, 250]
        pt_dimuon = [7, 0, 250]
        dimuon_eta = [8, -5, 5]
        cos_theta = [9, -1, 1]
        dimuon_phi = [10, -5, 5]

    num_of_variables = len(xlabels_extra)

    original_var_ranges=[pt_lead,eta_lead,phi_lead,pt_sub,eta_sub,phi_sub]
    jet1_var_ranges = [jet1_pt,jet1_eta,jet1_phi]
    jet2_var_ranges = [jet2_pt, jet2_eta, jet2_phi]
    jet3_var_ranges = [jet3_pt, jet3_eta, jet3_phi]
    jet4_var_ranges = [jet4_pt, jet4_eta, jet4_phi]

    if jetnum==0:

        var_name = xlabels_extra
        var_list = original_var_ranges+calc_var_ranges
        units=units+units_calc
    elif jetnum==1:

        var_name = xlabels_extra
        var_list =original_var_ranges+ jet1_var_ranges+calc_var_ranges
        units=units+units_jet+units_calc
    elif jetnum==2:

        var_list = original_var_ranges  + jet1_var_ranges +jet2_var_ranges+calc_var_ranges
        var_name = xlabels_extra
        units=units+units_jet+units_jet+units_calc
    elif jetnum == 3:

        var_list = original_var_ranges + jet1_var_ranges + jet2_var_ranges +jet3_var_ranges+calc_var_ranges
        var_name = xlabels_extra
        units = units +units_jet + units_jet + units_jet+units_calc
    elif jetnum == 4:

        var_list = original_var_ranges +calc_var_ranges+ jet1_var_ranges + jet2_var_ranges +jet3_var_ranges+ jet4_var_ranges
        var_name = xlabels_extra
        units = units + units_calc+units_jet + units_jet + units_jet+units_jet
    else:

        var_name = xlabels_extra
        var_list = [original_var_ranges+calc_var_ranges]
        units=units+units_calc
    from IPython.display import IFrame


    for i in range(len(var_list)):
        Plotting(truths_cut, predictions_cut, var_list[i][0], var_list[i][1], var_list[i][2], var_name[i],units)

    # Check if all elements in array are zero
    #for i in range(len(predictions_cut[1,:])):
    #    print((truths_cut[:,i] < 0).any(axis=0))

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



    #Create Directory to store plots
    os.makedirs('Plots', exist_ok=True)

    # Plot loss and wasserstein plots
    end_of_run_plots(w_list,loss_list)

    #Print Length of datasets
    print('Number of True Events: ', len(truths))
    print('Number of Generated Events: ', len(predictions))

    #Plot the remaining graphs
    main(truths,predictions,w_list,loss_list,new_run_folder,jetnum,xlabels)
