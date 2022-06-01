import matplotlib.pyplot as plt
import numpy as np
from pylorentz import Momentum4
import os
import pandas as pd
import seaborn as sns
#from variables import max_epochs,xlabels



def hmumu_plot(predictions, truths,outdir,epochs,xlabels,
    xranges=None, xbins=None):

    xlabels_extra=xlabels
    xlabels_extra.append('DiMuon Invarient Mass')
    xlabels_extra.append('DiMuon P_t')
    xlabels_extra.append('DiMuon Pseudorapidity (Eta)')
    xlabels=xlabels[:-3]
    #New Code
    num_of_variables=len(xlabels)
    county=epochs

    #Function to calculate invarient dimuon mass
    truths,predictions=dimuon_calc(predictions,truths)
    #Applying Selection Cuts
    
    print('Length of predicted data set pre-cuts: ',len(predictions))
    print('Length of truth data set pre-cuts: ',len(truths))
    predictions_cut,truths,predictions,truths_cut,select_cut_val=selection_cuts(predictions,truths)

    #xlabels_extra.append('Eta Between the Two Muons')
    #xlabels_extra.append('Phi Between the Two Muons')

    #Plots
    # Plot Original Variables
    original(predictions_cut,truths_cut,new_run_folder,xlabels_extra,county)


    # Plot Calculated Variables
    calculated(predictions_cut,truths_cut,new_run_folder,xlabels_extra,xlabels,county)


    # Plot ratio plots
    ratio(predictions_cut,truths_cut,new_run_folder,xlabels_extra,xlabels,county)


    # Plot correlation plots
    heatmap(predictions_cut,truths_cut,new_run_folder,xlabels_extra,county)


    #Plot Individual Heat Map plots
    var_corr(predictions_cut,truths_cut,new_run_folder,xlabels_extra,county)


    # Plot Large plots for calculated variables
    large_calc_plots(truths,predictions_cut,xlabels_extra,county,new_run_folder)


def original(predictions_cut,truths_cut,new_run_folder,xlabels_extra,county):

    # Original Variables
    num_of_variables = len(xlabels_extra)
    fig, axs = plt.subplots(1, len(xlabels)-3, figsize=(50, 10), constrained_layout=True)
    axs = axs.flatten()
    # config = dict(histtype='step', lw=2)
    config = dict(histtype='step', lw=2)
    i = 0
    for i in range(len(xlabels)-3):
        idx = i
        ax = axs[idx]
        yvals, _, _ = ax.hist(truths[:, idx], bins=40,
                              range=[min(predictions_cut[:, idx]), max(predictions_cut[:, idx])], label='Truth',
                              density=True, **config)
        max_y = np.max(yvals) * 1.1
        ax.hist(predictions_cut[:, idx], bins=40, range=[min(predictions_cut[:, idx]), max(predictions_cut[:, idx])],
                label='Generator', density=True, **config)
        ax.set_xlabel(xlabels_extra[i], fontsize=16)
        ax.legend(['Truth', 'Generator'], loc=3)
        # ax.set_yscale('log')

        # Save Figures
    plt.savefig(os.path.join(new_run_folder, 'image_at_epoch_{:04d}.png'.format(county)))
    plt.close('all')



def calculated(predictions_cut,truths,new_run_folder,xlabels_extra,xlabels,county):

    # Calculated Variables
    num_of_variables = len(xlabels_extra)
    num_calc_var = num_of_variables - len(xlabels)
    fig, axs = plt.subplots(1, num_calc_var, figsize=(20, 10), constrained_layout=True)
    axs = axs.flatten()
    # config = dict(histtype='step', lw=2)
    config = dict(histtype='step', lw=2)
    i = 0

    for i in range(num_calc_var):
        idx = i
        ax = axs[idx]
        idx = idx + len(xlabels)
        yvals, _, _ = ax.hist(truths[:, idx], bins=40,
                              range=[min(predictions_cut[:, idx]), max(predictions_cut[:, idx])], label='Truth',
                              density=True, **config)
        max_y = np.max(yvals) * 1.1
        ax.hist(predictions_cut[:, idx], bins=40, range=[min(predictions_cut[:, idx]), max(predictions_cut[:, idx])],
                label='Generator', density=True, **config)
        ax.set_xlabel(xlabels_extra[i + len(xlabels)], fontsize=16)
        ax.legend(['Truth', 'Generator'], loc=3)
        # ax.set_yscale('log')

        # Save Figures
    plt.savefig(os.path.join(new_run_folder, 'calc_image_at_epoch_{:04d}.png'.format(county)))
    plt.close('all')


def ratio (predictions_cut,truths,new_run_folder,xlabels_extra,xlabels,county):
    # Ratio Plots

    xlabelstemp = xlabels[:6]
    # New Code
    num_of_variables = len(xlabelstemp)
    county = epochs
    # Original Variables
    num_of_variables = len(xlabels_extra)
    fig, axs = plt.subplots(1, len(xlabels), figsize=(27, 5), constrained_layout=True)
    axs = axs.flatten()
    # config = dict(histtype='step', lw=2)
    config = dict(histtype='step', lw=2)
    i = 0

    for i in range(len(xlabels)):
        idx = i
        ax = axs[idx]
        ratio = []

        yvals, true1 = np.histogram(truths[:, idx], bins=40)
        yvals2, pred1 = np.histogram(predictions_cut[:, idx], bins=40)

        ratio = yvals / yvals2

        error=np.sqrt(len(ratio))
        ax.scatter(true1[:-1], ratio)
        ax.errorbar(true1[:-1], ratio, yerr=error, fmt='o')
        max_y = np.max(yvals) * 1.1

        ax.set_xlabel('Ratio of true to generated events for :' + xlabels_extra[i], fontsize=8)
    plt.savefig(os.path.join(new_run_folder, 'ratio_image_at_epoch_{:04d}.png'.format(county)))
    plt.close('all')

    # Calculated Variables
    num_of_variables = len(xlabels_extra)
    num_calc_var = num_of_variables - len(xlabels)
    fig, axs = plt.subplots(1, num_calc_var, figsize=(20, 10), constrained_layout=True)
    axs = axs.flatten()
    # config = dict(histtype='step', lw=2)
    config = dict(histtype='step', lw=2)
    i = 0

    for i in range(num_calc_var):
        idx = i
        ax = axs[idx]
        idx = idx + len(xlabels)
        ratio = []

        yvals, true1 = np.histogram(truths[:, idx], bins=40)

        yvals2, pred1 = np.histogram(predictions_cut[:, idx], bins=40)

        ratio = yvals / yvals2
        error = np.sqrt(len(ratio))
        ax.scatter(true1[:-1], ratio)
        ax.errorbar(true1[:-1], ratio, yerr=error, fmt='o')
        max_y = np.max(yvals) * 1.1

        ax.set_xlabel('Ratio of true to generated events for :' + xlabels_extra[i + len(xlabels)], fontsize=10)
    plt.savefig(os.path.join(new_run_folder, 'calc_ratio_image_at_epoch_{:04d}.png'.format(county)))
    plt.close('all')

    # Reset Style
    plt.rcParams.update(plt.rcParamsDefault)


def heatmap(predictions,truths,new_run_folder,xlabels_extra,county):
    # Converting to numpy array
    predictions = np.array(predictions)
    truths = np.array(truths)

    # Converting to Pandas
    df_truths = pd.DataFrame(truths[:, :], columns=xlabels_extra)
    df_predictions = pd.DataFrame(predictions[:, :], columns=xlabels_extra)

    # Correlation Plot for True Data
    sns.set(font_scale=2.0)
    plt.rcParams['figure.figsize'] = (40.0, 30.0)
    sns.heatmap(df_truths.corr(), annot=True, vmin=-1, vmax=1, center=0)
    plt.savefig(os.path.join(new_run_folder, 'heatmap_at_epoch_truths_{:04d}.png'.format(county)))
    plt.close('all')

    # Correlation Plot for Generated Data
    plt.rcParams['figure.figsize'] = (40.0, 30.0)
    sns.heatmap(df_predictions.corr(), annot=True, vmin=-1, vmax=1, center=0)
    plt.savefig(os.path.join(new_run_folder, 'heatmap_at_epoch_generated_{:04d}.png'.format(county)))
    plt.close('all')

    # Reset Style
    plt.rcParams.update(plt.rcParamsDefault)

def large_calc_plots(truths,predictions_cut,xlabels_extra,county,new_run_folder):


    #Plot Just Dimuon But Big with mean and SD   
    fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    idx=len(xlabels_extra)-3
    ax = axs
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40,   label='Truth',density=True,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40,   label='Generator',density=True,**config)
    ax.set_xlabel(r"DiMuon Invarient Mass")
    ax.axvline(truths[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()+truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()-truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean(), color='g', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()+predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()-predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    #plt.yscale('log')
    ax.legend(['Truth', 'Generator','Truth Mean','Generated SD','Truth SD','Generated Mean'])
    plt.savefig(os.path.join(new_run_folder, 'dimuon_image_at_epoch_{:04d}.png'.format(county)))
    plt.close('all')
    
    
    #Plot Log Dimuon
    fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    idx=len(xlabels_extra)-3
    ax = axs
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40,  label='Truth',density=True,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40,   label='Generator',density=True,**config)
    ax.axvline(truths[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()+truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()-truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean(), color='g', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()+predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()-predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.set_xlabel(r"DiMuon Invarient Mass")
    plt.yscale('log')
    ax.legend(['Truth', 'Generator','Truth Mean','Generated SD','Truth SD','Generated Mean'])
    plt.savefig(os.path.join(new_run_folder, 'dimuon_log_image_at_epoch_{:04d}.png'.format(county)))
    plt.close('all')
    plt.rcParams.update(plt.rcParamsDefault)

     #Plot Dimuon Pt
    fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    idx=len(xlabels_extra)-2
    ax = axs
    yvals, _, _ = ax.hist(truths[:, idx],  bins=80,   label='Truth',density=True,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=80,   label='Generator',density=True,**config)
    ax.set_xlabel(r"DiMuon Pt")
    ax.axvline(truths[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()+truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()-truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean(), color='g', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()+predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()-predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    #plt.yscale('log')
    ax.legend(['Truth', 'Generator','Truth Mean','Generated SD','Truth SD','Generated Mean'])
    plt.savefig(os.path.join(new_run_folder, 'dimuon_pt_image_at_epoch_{:04d}.png'.format(county)))
    plt.close('all')

    #Plot Dimuon Pt Log
    fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    idx=len(xlabels_extra)-2
    ax = axs
    yvals, _, _ = ax.hist(truths[:, idx],  bins=80,   label='Truth',density=True,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=80,  label='Generator',density=True,**config)
    ax.axvline(truths[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()+truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()-truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean(), color='g', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()+predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()-predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.set_xlabel(r"DiMuon Pt")
    plt.yscale('log')
    ax.legend(['Truth', 'Generator','Truth Mean','Generated SD','Truth SD','Generated Mean'])
    plt.savefig(os.path.join(new_run_folder, 'dimuon_pt_log_image_at_epoch_{:04d}.png'.format(county)))
    plt.close('all')
    plt.rcParams.update(plt.rcParamsDefault)

    #Plot Dimuon Pseudorapidity
    fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    idx=len(xlabels_extra)-1
    ax = axs
    yvals, _, _ = ax.hist(truths[:, idx],  bins=80,  label='Truth',density=True,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=80,  label='Generator',density=True,**config)
    ax.set_xlabel(r"DiMuon Pseudorapidity Mass")
    ax.axvline(truths[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()+truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()-truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean(), color='g', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()+predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()-predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    #plt.yscale('log')
    ax.legend(['Truth', 'Generator','Truth Mean','Generated SD','Truth SD','Generated Mean'])
    plt.savefig(os.path.join(new_run_folder, 'dimuon_pseudo_image_at_epoch_{:04d}.png'.format(county)))
    plt.close('all')
    
    #Plot Log pseudorapidity
    fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    idx=len(xlabels_extra)-1
    ax = axs
    yvals, _, _ = ax.hist(truths[:, idx],  bins=80, label='Truth',density=True,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=80,  label='Generator',density=True,**config)
    ax.axvline(truths[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()+truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()-truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean(), color='g', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()+predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()-predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.set_xlabel(r"DiMuon Pseudorapidity Mass")
    plt.yscale('log')
    ax.legend(['Truth', 'Generator','Truth Mean','Generated SD','Truth SD','Generated Mean'])
    plt.savefig(os.path.join(new_run_folder, 'dimuon_pseudo_log_image_at_epoch_{:04d}.png'.format(county)))
    plt.close('all')
    plt.rcParams.update(plt.rcParamsDefault)

      
def selection_cuts(predictions,truths):

    #predictions=np.c_[ predictions ]
    predictions_cut= predictions[(-1*np.pi<predictions[:,1]) & (predictions[:,1] < 1*np.pi)] 
    predictions_cut= predictions_cut[(-1*np.pi<predictions_cut[:,2]) & (predictions_cut[:,2] < 1*np.pi)] 
    predictions_cut= predictions_cut[(-1*np.pi<predictions_cut[:,4]) & (predictions_cut[:,4] < 1*np.pi)] 
    predictions_cut= predictions_cut[(-1*np.pi<predictions_cut[:,5]) & (predictions_cut[:,5] < 1*np.pi)] 

    print('Length of predicted data set post-cuts: ',len(predictions_cut))
    print('Fraction of events lost : ',1-len(predictions_cut)/len(predictions))
    select_cut_val=1-len(predictions_cut)/len(predictions)

    return predictions_cut,truths,predictions,truths,select_cut_val

def var_corr(predictions,truths,new_run_folder,xlabels_extra,county):

    print('')
    #Creating plots for every combination of variables for generated data
    j=0
    i=0
    for i in range(len(xlabels_extra)):
        j=0
        for j in range(len(xlabels_extra)):
            
                plt.hist2d(predictions[:,i],predictions[:,j], (100, 100), cmap=plt.cm.jet)
                plt.colorbar()
                #plt.scatter(predictions[:,i],predictions[:,j])
                plt.xlabel(xlabels_extra[i])
                plt.ylabel(xlabels_extra[j])
                plt.savefig(os.path.join(new_run_folder, 'Variables: '+xlabels_extra[i]+' and '+xlabels_extra[j]+' Scatterplot_at_epoch_generated_{:04d}.png'.format(county)))
                plt.close('all')
                
                
    #Creating plots for every combination of variables for true data
    j=0
    i=0
    for i in range(len(xlabels_extra)):
        j=0
        for j in range(len(xlabels_extra)):

                plt.hist2d(truths[:,i],truths[:,j], (100, 100), cmap=plt.cm.jet)
                plt.colorbar()
                
                
                #plt.scatter(truths[:,i],truths[:,j])
                plt.xlabel(xlabels_extra[i])
                plt.ylabel(xlabels_extra[j])
                plt.savefig(os.path.join(new_run_folder, 'Variables: '+xlabels_extra[i]+' and '+xlabels_extra[j]+' Scatterplot_at_epoch_truth_{:04d}.png'.format(county)))
                plt.close('all')
                

def dimuon_calc(predictions,truths):
        
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
    pt_comb_true=[]
    pseudo_true=[]
    eta_angle_btwn_true=[]
    phi_angle_btwn_true=[]
    
    
    for i in range(len(truths)):
        #Use pylorentz to define 4 momentum arrays for each event
        muon_lead_true.append(Momentum4.m_eta_phi_pt(masses_lead[i], etas_lead_true[i], phis_lead_true[i], pts_lead_true[i]))
        muon_sub_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], etas_sub_true[i], phis_sub_true[i], pts_sub_true[i]))

        #Calculate the Higgs boson 4 vector
        parent_true.append(muon_lead_true[i] + muon_sub_true[i])

        #Retrieve the Higgs Mass
        mass_true.append(parent_true[i].m)
        
                
        
        #Retrive PT
        pt_comb_true.append(parent_true[i].p_t)
        
        
        #Retrieve Pseudorapidity
        pseudo_true.append(parent_true[i].eta)
        
        #Retrieve eta between the muons
        eta_angle_btwn_true.append(muon_lead_true[i].eta-muon_sub_true[i].eta)
         
        #Retrieve eta between the muons
        eta_angle_btwn_true.append(muon_lead_true[i].phi-muon_sub_true[i].phi)
        

    #Add mass arrays from each batch?    
    mass_true=np.concatenate( mass_true, axis=0 )

    #Add to original dataset
    truths = np.column_stack((truths, mass_true))
    truths = np.column_stack((truths, pt_comb_true))
    truths = np.column_stack((truths, pseudo_true))
    #truths = np.column_stack((truths, eta_angle_btwn_true))
    #truths = np.column_stack((truths, phi_angle_btwn_true))
    
    # For Predictions
    masses_lead = np.full((len(predictions[:, 0]), 1), m_u)
    masses_sub = np.full((len(predictions[:, 0]), 1), m_u)

    # Take each column from the tru and generated data and rename to their parameter type
    pts_lead_pred = np.array(predictions[:, 0]).flatten()
    etas_lead_pred = np.array(predictions[:, 1]).flatten()
    phis_lead_pred = np.array(predictions[:, 2]).flatten()
    pts_sub_pred = np.array(predictions[:, 3]).flatten()
    etas_sub_pred = np.array(predictions[:, 4]).flatten()
    phis_sub_pred = np.array(predictions[:, 5]).flatten()

    # Create lists for 4 vector values
    muon_lead_pred = []
    muon_sub_pred = []
    parent_pred = []

    # Create lists for invarient mass values
    mass_pred = []
    pt_comb_pred = []
    pseudo_pred = []
    eta_angle_btwn_pred = []
    phi_angle_btwn_pred = []

    for i in range(len(predictions)):
        # Use pylorentz to define 4 momentum arrays for each event
        muon_lead_pred.append(
            Momentum4.m_eta_phi_pt(masses_lead[i], etas_lead_pred[i], phis_lead_pred[i], pts_lead_pred[i]))
        muon_sub_pred.append(Momentum4.m_eta_phi_pt(masses_sub[i], etas_sub_pred[i], phis_sub_pred[i], pts_sub_pred[i]))

        # Calculate the Higgs boson 4 vector
        parent_pred.append(muon_lead_pred[i] + muon_sub_pred[i])
        # print(parent_true)
        # Retrieve the Higgs Mass
        mass_pred.append(parent_pred[i].m)

        # Retrive PT
        pt_comb_pred.append(parent_pred[i].p_t)

        # Retrieve Pseudorapidity
        pseudo_pred.append(parent_pred[i].eta)

        # Retrieve eta between the muons
        eta_angle_btwn_pred.append(muon_lead_pred[i].eta - muon_sub_pred[i].eta)

        # Retrieve eta between the muons
        eta_angle_btwn_pred.append(muon_lead_pred[i].phi - muon_sub_pred[i].phi)

    # print(pt_comb_true)

    # Add mass arrays from each batch?
    mass_pred = np.concatenate(mass_pred, axis=0)

    # Add to original dataset
    predictions = np.column_stack((predictions, mass_pred))
    predictions = np.column_stack((predictions, pt_comb_pred))
    predictions = np.column_stack((predictions, pseudo_pred))
    # predictions = np.column_stack((predictions, eta_angle_btwn_true))
    # predictions = np.column_stack((predictions, phi_angle_btwn_true))

    return truths,predictions

def end_of_run_plots(w_list,loss_list,epochs,new_run_folder):
    #Plot Log loss
    print('new run',new_run_folder)
    plt.plot(loss_list)
    plt.ylabel('Training Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(new_run_folder, 'logloss.png'))
    plt.clf()
    #Plot Wasserstein Distance
    plt.plot(w_list)
    plt.ylabel('Wasserstein Distance')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(new_run_folder, 'wasserstein.png'))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Normalizing Flow')
    add_arg = parser.add_argument
    add_arg('--bestepoch',default=5000, help='Epoch for best results')
    add_arg('--jetnum', default=0, help='Number of Jets in File')
    args = parser.parse_args()
    jetnum=int(args.jetnum)
    epochs=int(args.bestepoch)
    #Define labels depending on number of jets (WiP)
    print('ahhhhhhhhhh')
    if jetnum==0:
        xlabels=['leading Muon pT', 'leading Muon eta', 'leading Muon phi', 'subleading Muon pT', 'subleading Muon eta', 'subleading Muon phi']
    elif jetnum==1:
        xlabels = ['leading Muon pT', 'leading Muon eta', 'leading Muon phi', 'subleading Muon pT',
                   'subleading Muon eta', 'subleading Muon phi', 'Jet 1 pT', 'Jet 1 eta', 'Jet 1 phi']
    else:
        xlabels = ['leading Muon pT', 'leading Muon eta', 'leading Muon phi', 'subleading Muon pT',
                   'subleading Muon eta', 'subleading Muon phi']


    #Load Data from Temp_Data
    truths=np.load('Temp_Data/truths.npy')
    predictions = np.load('Temp_Data/predictions.npy')
    w_list=np.load('Temp_Data/w_list.npy')
    loss_list = np.load('Temp_Data/loss_list.npy')
    f = open("Temp_Data/filename.txt", "r")
    if f.mode == 'r':
        new_run_folder= f.read()

    #Plot loss and wasserstein plots
    end_of_run_plots(w_list,loss_list,epochs,new_run_folder)

    #Plot everything else
    hmumu_plot(predictions, truths,new_run_folder,epochs,xlabels)


    
