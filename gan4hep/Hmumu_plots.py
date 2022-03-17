import matplotlib.pyplot as plt
import numpy as np
from pylorentz import Momentum4
import os

def hmumu_plot(predictions, truths, outname, xlabels,truth_data,new_run_folder,i,
    xranges=None, xbins=None):
    
    xlabelstemp=xlabels[:6]
    #New Code
    #Creating Plots with range between -1 and 1
    num_of_variables=len(xlabelstemp)

    fig, axs = plt.subplots(1, num_of_variables, figsize=(20, 0.5*num_of_variables), constrained_layout=True)
    axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    #Plot 1
    
    for i in range(num_of_variables):
        idx=i
        ax = axs[idx]
        x_range = [-1, 1]
        x_range_pt = [0, 1]  

        #Normalized Plots 
        yvals, _, _ = ax.hist(truths[:, idx], bins=40,  range=x_range,  label='Truth',density=True, **config)
        max_y = np.max(yvals) * 1.1
        ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator',density=True, **config)
        ax.axvline(truths[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
        ax.axvline(predictions[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
        ax.set_xlabel(xlabels[i])
        #ax.set_ylim(0, max_y)
        ax.legend(['Truth', 'Generator'])
        #ax.set_yscale('log')
    plt.savefig(os.path.join(new_run_folder, 'normalized_image_at_epoch_{:04d}.png'.format(i)))
    plt.close('all')

    #Apply Inverse Scaler to get original values back
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=(-1,1))
    truth_data = scaler.fit_transform(truth_data)
    truths=scaler.inverse_transform(truths)
    predictions=scaler.inverse_transform(predictions)
    
    #Function to calculate invarient dimuon mass
    truths,predictions=dimuon_calc(predictions,truths)
    
    
    
    #Applying Selection Cuts
    
    print('Length of predicted data set pre-cuts: ',len(predictions))
    print('Length of truth data set pre-cuts: ',len(truths))
    predictions_cut,truths,predictions,truths_cut,select_cut_val=selection_cuts(predictions,truths)

    #Creating  Original Valued Plots
    xlabels_extra=xlabelstemp
    xlabels_extra.append('DiMuon Invarient Mass')
    xlabels_extra.append('DiMuon P_t')
    xlabels_extra.append('DiMuon Pseudorapidity (Eta)')

    num_of_variables=len(xlabels_extra)
    fig, axs = plt.subplots(1, num_of_variables, figsize=(50, 1.5*num_of_variables+1), constrained_layout=True)
    axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    i=0
    for i in range(num_of_variables): 
        idx=i
        ax = axs[idx]
        x_range = [-np.pi, np.pi]
        x_range_pt = [0, 1]
        yvals, _, _ = ax.hist(truths[:, idx], bins=40,   label='Truth',density=True, **config)
        max_y = np.max(yvals) * 1.1
        ax.hist(predictions_cut[:, idx], bins=40,  label='Generator',density=True, **config)
        ax.axvline(truths[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
        ax.axvline(predictions_cut[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
        ax.set_xlabel(xlabels_extra[i])
        #ax.set_ylim(0, max_y)
        ax.legend(['Truth', 'Generator','Truth Mean='+str(truths[:, idx].mean()),'Generated Mean='+str(predictions[:, idx].mean())])
        #ax.set_yscale('log')
        
        #Save Figures
    plt.savefig(os.path.join(new_run_folder, 'image_at_epoch_{:04d}.png'.format(i)))
    plt.close('all')
    
    #Plot correlation plot   
    var_corr(predictions_cut,truths_cut,new_run_folder,i,xlabels_extra)
    #Reset Style
    plt.close('all')
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    
    #Plot Just Dimuon But Big with mean and SD   
    fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    idx=len(xlabels_extra)-3
    ax = axs
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40, range=[0,200],  label='Truth',density=True,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40, range=[0,200],  label='Generator',density=True,**config)
    ax.set_xlabel(r"DiMuon Invarient Mass")
    ax.axvline(truths[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()+truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()-truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean(), color='g', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()+predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()-predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    #plt.yscale('log')
    ax.legend(['Truth', 'Generator','Truth Mean','Truth SD','Generated Mean','Generated SD'])
    plt.savefig(os.path.join(new_run_folder, 'dimuon_image_at_epoch_{:04d}.png'.format(i)))
    plt.close('all')
    
    
    #Plot Log Dimuon
    fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    idx=len(xlabels_extra)-3
    ax = axs
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40, range=[0,200],  label='Truth',density=True,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40, range=[0,200],  label='Generator',density=True,**config)
    ax.axvline(truths[:, idx].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()+truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(truths[:, idx].mean()-truths[:, idx].std(), color='r', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean(), color='g', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()+predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.axvline(predictions_cut[:, idx].mean()-predictions_cut[:, idx].std(), color='y', linestyle='dashed', linewidth=1)
    ax.set_xlabel(r"DiMuon Invarient Mass")
    plt.yscale('log')
    ax.legend(['Truth', 'Generator','Truth Mean','Generated SD','Truth SD','Generated Mean'])
    plt.savefig(os.path.join(new_run_folder, 'dimuon_log_image_at_epoch_{:04d}.png'.format(i)))
    plt.close('all')
    
    
def sd_mean_calc(inputdata):
    
    #Calculate sd and mean for data (Currently unused)
    counts, bins = np.histogram(inputdata)
    mids = 0.5*(bins[1:] + bins[:-1])
    probs = counts / np.sum(counts)

    mean = np.sum(probs * mids)  
    sd = np.sqrt(np.sum(probs * (mids - mean)**2))
    return sd,mean    
      
def selection_cuts(predictions,truths):

    listy=[]
    for i in range(len(predictions)):
        listy.append(i)
                
    predictions=np.c_[ predictions, listy ]
    predictions_cut= predictions[(-1*np.pi<predictions[:,1]) & (predictions[:,1] < 1*np.pi)] 
    predictions_cut= predictions_cut[(-1*np.pi<predictions_cut[:,2]) & (predictions_cut[:,2] < 1*np.pi)] 
    predictions_cut= predictions_cut[(-1*np.pi<predictions_cut[:,4]) & (predictions_cut[:,4] < 1*np.pi)] 
    predictions_cut= predictions_cut[(-1*np.pi<predictions_cut[:,5]) & (predictions_cut[:,5] < 1*np.pi)] 

    print('Length of predicted data set post-cuts: ',len(predictions_cut))
    print('Fraction of events lost : ',1-len(predictions_cut)/len(predictions))

    select_cut_val=1-len(predictions_cut)/len(predictions)
    
    return predictions_cut,truths,predictions,truths,select_cut_val

def var_corr(predictions,truths,new_run_folder,i,xlabels_extra):
    import pandas as pd
    import seaborn as sns
        
    count=i
    var_name_list=['PT_Lead','Eta_Lead','Phi_Lead','PT_Sub','Eta_Sub','Phi_Sub','DIMuon Mass']
    #Converting to numpy array
    predictions=np.array(predictions)
    truths=np.array(truths)
    
    #Converting to Pandas
    df_truths = pd.DataFrame(truths[:,:], columns = xlabels_extra)
    df_predictions = pd.DataFrame(predictions[:,:-1], columns = xlabels_extra)
    
    
    #Correlation Plot for True Data
    sns.set(font_scale=2.0)
    plt.rcParams['figure.figsize'] = (20.0, 15.0)
    sns.heatmap(df_truths.corr(),annot=True)
    plt.savefig(os.path.join(new_run_folder, 'heatmap_at_epoch_truths_{:04d}.png'.format(count)))
    plt.close('all')
    
    #Correlation Plot for Generated Data
    plt.rcParams['figure.figsize'] = (20.0, 15.0)
    sns.heatmap(df_predictions.corr(),annot=True)
    plt.savefig(os.path.join(new_run_folder, 'heatmap_at_epoch_generated_{:04d}.png'.format(count)))
    plt.close('all')
    
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
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
                plt.savefig(os.path.join(new_run_folder, 'Variables: '+xlabels_extra[i]+' and '+xlabels_extra[j]+' Scatterplot_at_epoch_generated_{:04d}.png'.format(count)))
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
                plt.savefig(os.path.join(new_run_folder, 'Variables: '+xlabels_extra[i]+' and '+xlabels_extra[j]+' Scatterplot_at_epoch_truth_{:04d}.png'.format(count)))
                plt.close('all')
                
    #Creating Combined Correlation Plot (Truth)
    var_name_list=['PT_Lead','Eta_Lead','Phi_Lead','PT_Sub','Eta_Sub','Phi_Sub','DIMuon Mass']
    fig, axes = plt.subplots(nrows=len(xlabels_extra), ncols=len(xlabels_extra), figsize=(40,40))
    j=0
    i=0

    for i in range(len(xlabels_extra)):
        j=0
        for j in range(len(xlabels_extra)):
            
            plt.hist2d(truths[:,i],truths[:,j], (100, 100), cmap=plt.cm.jet)
            plt.colorbar()
            #axes[i][j].scatter( truths[:, i], truths[:, j])
            axes[i][j].set_ylabel(xlabels_extra[i])
            axes[i][j].set_xlabel(xlabels_extra[j])
        j=j+1

            
    plt.savefig(os.path.join(new_run_folder, 'correlation_at_epoch_truths_{:04d}.png'.format(count)))
    plt.close('all')
 
    #Creating Correlation Plot (Truth)
    fig, axes = plt.subplots(nrows=len(xlabels_extra), ncols=len(xlabels_extra), figsize=(40,40))
    j=0
    for i in range(len(xlabels_extra)):
        j=0
        for j in range(len(xlabels_extra)):

            
            plt.hist2d(predictions[:,i],predictions[:,j], (100, 100), cmap=plt.cm.jet)
            plt.colorbar()
                
                
            #axes[i][j].scatter( predictions[:, i], predictions[:, j])
            axes[i][j].set_ylabel(xlabels_extra[i])
            axes[i][j].set_xlabel(xlabels_extra[j])

        j=j+1

    plt.savefig(os.path.join(new_run_folder, 'correlation_at_epoch_generated_{:04d}.png'.format(count)))
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
        
        
        

    #Add mass arrays from each batch?    
    mass_true=np.concatenate( mass_true, axis=0 )

    #Add to original dataset
    truths = np.column_stack((truths, mass_true))
    truths = np.column_stack((truths, pt_comb_true))
    truths = np.column_stack((truths, pseudo_true))
    
    #For Predictions
    masses_lead = np.full((len(predictions[:,0]), 1), m_u)
    masses_sub = np.full((len(predictions[:,0]), 1), m_u)
    
    #Take each column from the tru and generated data and rename to their parameter type
    pts_lead_true = np.array(predictions[:, 0]).flatten()
    etas_lead_true = np.array(predictions[:, 1]).flatten()
    phis_lead_true = np.array(predictions[:, 2]).flatten()
    pts_sub_true = np.array(predictions[:, 3]).flatten()
    etas_sub_true =np.array(predictions[:, 4]).flatten()
    phis_sub_true = np.array(predictions[:, 5]).flatten()

    #Create lists for 4 vector values
    muon_lead_true=[]
    muon_sub_true=[]
    parent_true=[]
    
    #Create lists for invarient mass values
    mass_true=[]
    pt_comb_true=[]
    pseudo_true=[]
    
    
    
    for i in range(len(predictions)):
        #Use pylorentz to define 4 momentum arrays for each event
        muon_lead_true.append(Momentum4.m_eta_phi_pt(masses_lead[i], etas_lead_true[i], phis_lead_true[i], pts_lead_true[i]))
        muon_sub_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], etas_sub_true[i], phis_sub_true[i], pts_sub_true[i]))

        #Calculate the Higgs boson 4 vector
        parent_true.append(muon_lead_true[i] + muon_sub_true[i])
        #print(parent_true)
        #Retrieve the Higgs Mass
        mass_true.append(parent_true[i].m)

        
        
        #Retrive PT
        pt_comb_true.append(parent_true[i].p_t)
        
        
        #Retrieve Pseudorapidity
        pseudo_true.append(parent_true[i].eta)
        
        
        
        
        
        
        
    #print(pt_comb_true)     
        
    #Add mass arrays from each batch?    
    mass_true=np.concatenate( mass_true, axis=0 )

    #Add to original dataset
    predictions = np.column_stack((predictions, mass_true))
    predictions = np.column_stack((predictions, pt_comb_true))
    predictions = np.column_stack((predictions, pseudo_true))
    
    return truths,predictions
    
    
    