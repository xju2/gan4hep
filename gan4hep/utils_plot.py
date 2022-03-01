import matplotlib.pyplot as plt
import numpy as np
from pylorentz import Momentum4
import os
def add_mean_std(array, x, y, ax, color='k', dy=None, digits=2, fontsize=12, with_std=True):
    this_mean, this_std = np.mean(array), np.std(array)
    if dy is None:
        dy = y * 0.1
    ax.text(x, y, "mean: {0:.{1}f}".format(this_mean, digits), color=color, fontsize=12)
    if with_std:
        ax.text(x, y-dy, "std: {0:.{1}f}".format(this_std, digits), color=color, fontsize=12)
    return ax
        
        
def array2hist(array, ax=None, with_mean_std=True, bins=100, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(6,6))
    entries, _, _ = ax.hist(array, bins=bins, **kwargs)
    if with_mean_std:
        min_x, max_x = np.min(array), np.max(array)
        x = 0.5*(max_x + min_x)
        y = np.max(entries)*0.8
        add_mean_std(array, x, y, ax)
    return ax


def view_particle_4vec(particles, axs=None, labels=None, outname=None, **kwargs):
    """
    make histograms of the 4 vectors of the particles
    Inputs:
        particles: 2D arrays (num_particles x 4vectors)
        labels: naming of the 4vectors, default ['$P_x$ [GeV]', '$P_y$ [GeV]', '$P_z$ [GeV]', "$E$ [GeV]"]
    Return:
        ax
    """
    if labels is None:
        labels = ['$P_x$ [GeV]', '$P_y$ [GeV]', '$P_z$ [GeV]', "$E$ [GeV]"]

    if axs is None or len(axs) != 4:
        fig, axs = plt.subplots(2,2, figsize=(10,10))
        axs = axs.flatten()
    
    for idx in range(4):
        array2hist(particles[:, idx], axs[idx], **kwargs)
        axs[idx].set_xlabel(labels[idx])

    plt.legend()
    if outname is not None:
        plt.savefig(outname+'.pdf')
    return axs

def compare_4vec(predicts, truths, labels=None, nbins=35, min_x=-0.5, max_x=3, **kwargs):
    hist_config = {
        "alpha": 0.8,
        "lw": 2,
        'histtype': 'step',
    }
    axs = view_particle_4vec(predicts, label='prediction',
        labels=labels, bins=nbins, range=(min_x, max_x), **hist_config, **kwargs)
    view_particle_4vec(truths, axs=axs, label='truth',
        labels=labels, bins=nbins, range=(min_x, max_x), **hist_config, **kwargs)
    
def compare(predictions, truths, outname, xlabels,truth_data,new_run_folder,i,
    xranges=None, xbins=None):
    
    #Original Code
    '''
    default xranges: [-1, 1]
    default xbins: 40
     

    num_variables = predictions.shape[1]
    if xranges is not None:
        assert len(xranges) == num_variables,\
            "# of x-axis ranges must equal to # of variables"

    if xbins is not None:
        assert len(xbins) == num_variables,\
            "# of x-axis bins must equal to # of variables"

    nrows, ncols = 1, 2
    if num_variables > 2:
        ncols = 2
        nrows = num_variables // ncols
        if num_variables % ncols != 0:
            nrows += 1 
    else:
        ncols = num_variables
        nrows = 1

    _, axs = plt.subplots(nrows, ncols,
        figsize=(4*ncols, 4*nrows), constrained_layout=True)
    axs = axs.flatten()

    config = dict(histtype='step', lw=2, density=True)
    for idx in range(num_variables):
        xrange = xranges[idx] if xranges else (-1, 1)
        xbin = xbins[idx] if xbins else 40
    
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx], bins=xbin, range=xrange, label='Truth', **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=xbin, range=xrange, label='Generator', **config)
    ax.set_xlabel(r"{}".format(xlabels[idx]))
    ax.set_ylim(0, max_y)
    ax.legend()
    '''
    
    

    
    
    
    
    
    
    #New Code
    #Creating Plots with range between -1 and 1
    fig, axs = plt.subplots(1, 6, figsize=(20, 6), constrained_layout=True)
    axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    #Plot 1
    idx=0
    ax = axs[idx]
    x_range = [-1, 1]
    x_range_pt = [0, 1]

    yvals, _, _ = ax.hist(truths[:, idx], bins=40,  range=x_range,  label='Truth',density=True, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator',density=True, **config)
    ax.set_xlabel(r"Muons_PT_Lead")
    #ax.set_ylim(0, max_y)
    ax.legend(['Truth', 'Generator'])
    #ax.set_yscale('log')

    # Plot 2
    idx=1
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40, range=x_range,  label='Truth',density=True, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator',density=True, **config)
    ax.set_xlabel(r"Muons_Eta_Lead")
    ax.legend(['Truth', 'Generator'])
    #ax.set_ylim(0, max_y)
    #ax.set_yscale('log')

    # plot 3
    idx=2
    ax = axs[idx]
    x_range = [-1, 1]

    yvals, _, _ = ax.hist(truths[:, idx], bins=40, range=x_range,   label='Truth',density=True, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator',density=True, **config)
    ax.set_xlabel(r"Muons_Phi_Lead")
    ax.legend(['Truth', 'Generator'])

    # plot 4
    idx=3
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40,  range=x_range,  label='Truth',density=True, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator',density=True, **config)
    ax.set_xlabel(r"Muons_PT_Sub")
    ax.legend(['Truth', 'Generator'])

    # plot 5
    idx=4
    ax = axs[idx]
    x_range = [-1, 1]

    yvals, _, _ = ax.hist(truths[:, idx], bins=40,  range=x_range,  label='Truth',density=True, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator',density=True, **config)
    ax.set_xlabel(r"Muons_Eta_Sub")
    ax.legend(['Truth', 'Generator'])

    # plot 6
    idx=5
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40,  range=x_range,  label='Truth',density=True, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator',density=True, **config)
    ax.set_xlabel(r"Muons_Phi_Sub")
    ax.legend(['Truth', 'Generator'])
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
    
    
    #Code brakes otherwise
    if len(predictions)/len(truths)==1.00:
        predictions_cut,truths,predictions,truths_cut=selection_cuts(predictions,truths)
    else: #Do nothing
        predictions_cut=predictions
        truths_cut=truths
        
    
    
    #Creating  Original Valued Plots
    fig, axs = plt.subplots(1, 7, figsize=(20, 7), constrained_layout=True)
    axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    #Plot 1
    idx=0
    ax = axs[idx]
    x_range = [-np.pi, np.pi]
    x_range_pt = [0, 1]
    yvals, _, _ = ax.hist(truths[:, idx], bins=40,   label='Truth',density=False, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40,  label='Generator',density=False, **config)
    ax.set_xlabel(r"Muons_PT_Lead")
    #ax.set_ylim(0, max_y)
    ax.legend(['Truth', 'Generator'])
    #ax.set_yscale('log')

    # Plot 2
    idx=1
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40, range=[-np.pi, np.pi],  label='Truth',density=False, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40, range=[-np.pi, np.pi], label='Generator',density=False, **config)
    ax.set_xlabel(r"Muons_Eta_Lead")
    ax.legend(['Truth', 'Generator'])
    #ax.set_ylim(0, max_y)
    #ax.set_yscale('log')

    # plot 3
    idx=2
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx], bins=40, range=[-2*np.pi, 2*np.pi],   label='Truth',density=False, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40, range=[-2*np.pi, 2*np.pi], label='Generator',density=False, **config)
    ax.set_xlabel(r"Muons_Phi_Lead")
    ax.legend(['Truth', 'Generator'])

    # plot 4
    idx=3
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40,  label='Truth',density=False, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40, label='Generator',density=False, **config)
    ax.set_xlabel(r"Muons_PT_Sub")
    ax.legend(['Truth', 'Generator'])

    # plot 5
    idx=4
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx], bins=40,  range=[-np.pi, np.pi],  label='Truth',density=False, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40, range=[-np.pi, np.pi], label='Generator',density=False, **config)
    ax.set_xlabel(r"Muons_Eta_Sub")
    ax.legend(['Truth', 'Generator'])

    # plot 6
    idx=5
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40,  range=[-2*np.pi, 2*np.pi],  label='Truth',density=False, **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40, range=[-2*np.pi, 2*np.pi], label='Generator',density=False, **config)
    ax.set_xlabel(r"Muons_Phi_Sub")
    ax.legend(['Truth', 'Generator'])
    
    
    # Plot Dimuon/7
   
    idx=6
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40,   label='Truth',density=False,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40,  label='Generator',density=False,**config)
    ax.set_xlabel(r"DiMuon Invarient Mass")
    ax.legend(['Truth', 'Generator'])
    
    #Save Figures
    plt.savefig(os.path.join(new_run_folder, 'image_at_epoch_{:04d}.png'.format(i)))
    plt.close('all')
    
    
    
    #Plot correlation plot if num of gen = num of true
    if len(predictions)/len(truths)==1.00:
        var_corr(predictions_cut,truths_cut,new_run_folder,i)


    
    #Plot Dimuon But Big    
    fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    idx=6
    ax = axs
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40, range=[0,200],  label='Truth',density=False,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40, range=[0,200],  label='Generator',density=False,**config)
    ax.set_xlabel(r"DiMuon Invarient Mass")
    #plt.yscale('log')
    ax.legend(['Truth', 'Generator'])
    plt.savefig(os.path.join(new_run_folder, 'dimuon_image_at_epoch_{:04d}.png'.format(i)))
    plt.close('all')
    
    
    #Plot dimuon but around Higgs mass
    fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    
    idx=6
    ax = axs
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40, range=[120,130],  label='Truth',density=False,**config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions_cut[:, idx], bins=40, range=[120,130],  label='Generator',density=False,**config)
    ax.set_xlabel(r"DiMuon Invarient Mass")
    plt.yscale('log')
    ax.legend(['Truth', 'Generator'])
    plt.savefig(os.path.join(new_run_folder, 'dimuon_log_image_at_epoch_{:04d}.png'.format(i)))
    plt.close('all')
    
    
    
    
    
    
    
def selection_cuts(predictions,truths):

    listy=[]
    for i in range(len(predictions)):
        listy.append(i)
                
    predictions=np.c_[ predictions, listy ]
    truths=np.c_[ truths, listy ]
    #print(predictions)
    
    predictions_cut= predictions[(-1*np.pi<predictions[:,1]) & (predictions[:,1] < 1*np.pi)] 
    predictions_cut= predictions_cut[(-1*np.pi<predictions_cut[:,2]) & (predictions_cut[:,2] < 1*np.pi)] 
    predictions_cut= predictions_cut[(-1*np.pi<predictions_cut[:,4]) & (predictions_cut[:,4] < 1*np.pi)] 
    predictions_cut= predictions_cut[(-1*np.pi<predictions_cut[:,5]) & (predictions_cut[:,5] < 1*np.pi)] 
    
    truths_cut=[]
    truths2=truths
    for i in range(len(predictions_cut)):
        for j in range(len(truths2)):
            if predictions_cut[i,7]==truths2[j,7]:
                np.delete(truths2,j)
                truths_cut.append(truths[j])

                                  
            
    print(len(truths_cut))      
            
            
            
    truths= truths[(-1*np.pi<truths[:,1]) & (truths[:,1] < 1*np.pi)] 
    truths= truths[(-1*np.pi<truths[:,2]) & (truths[:,2] < 1*np.pi)] 
    truths= truths[(-1*np.pi<truths[:,4]) & (truths[:,4] < 1*np.pi)] 
    truths= truths[(-1*np.pi<truths[:,5]) & (truths[:,5] < 1*np.pi)] 

    print('Length of predicted data set post-cuts: ',len(predictions_cut))
    print('Length of truth data set post-cuts: ',len(truths))
    print('Fraction of events lost : ',1-len(predictions_cut)/len(predictions))

    return predictions_cut,truths,predictions,truths_cut

def var_corr(predictions,truths,new_run_folder,i):
    import seaborn as sns; sns.set_theme()
    #Creating Correlation Plot (Truth)
    predictions=np.array(predictions)
    truths=np.array(truths)
    #fig, axs = plt.subplots(7, 7, figsize=(40, 7), constrained_layout=True)
    #axes = axs.flatten()
    count=i
    #i=0
    #j=0
    #print('test',len(axs))
    var_name_list=['PT_Lead','Eta_Lead','Phi_Lead','PT_Sub','Eta_Sub','Phi_Sub','DIMuon Mass']

    
       
        
        
        
        
        
        
    import pandas as pd
    import seaborn as sns
    from numpy.random import randint
    import matplotlib.pyplot as plt


    #ax = sns.heatmap(predictions[:,1],prediction[:,2])
    #plt.savefig(os.path.join(new_run_folder, 'correlation_at_epoch_truths_{:04d}.png'.format(count)))
        
        
        
        
        
        
        
        
        
        
        
        
        
    fig, axes = plt.subplots(nrows=7, ncols=7, figsize=(40,40), sharex=True, sharey=True)
    j=0
    for i, ax in enumerate(axes.flatten()):
        j=0
        for j in range(7):
            if i < 7:
                
                axes[i][j].scatter( truths[:, i], truths[:, j])
                #axes[i][j].sns.heatmap( truths[:, i], truths[:, j])
                axes[i][j].set_ylabel(var_name_list[i])
                axes[i][j].set_xlabel(var_name_list[j])

            j=j+1

            
    #plt.savefig(os.path.join(new_run_folder, 'correlation_at_epoch_truths_{:04d}.png'.format(count)))
    #plt.close('all')
    
    
    
    
    #Creating Correlation Plot (Truth)
    predictions=np.array(predictions)
    truths=np.array(truths)
    #fig, axs = plt.subplots(7, 7, figsize=(40, 7), constrained_layout=True)
    #axes = axs.flatten()
    count=i
    #i=0
    #j=0
    #print('test',len(axs))
    var_name_list=['PT_Lead','Eta_Lead','Phi_Lead','PT_Sub','Eta_Sub','Phi_Sub','DIMuon Mass']

    
            
    fig, axes = plt.subplots(nrows=7, ncols=7, figsize=(40,40), sharex=True, sharey=True)
    j=0
    for i, ax in enumerate(axes.flatten()):
        j=0
        for j in range(7):
            if i < 7:
               
                axes[i][j].scatter( predictions[:, i], predictions[:, j])
                axes[i][j].set_ylabel(var_name_list[i])
                axes[i][j].set_xlabel(var_name_list[j])

            j=j+1

            
            
    plt.savefig(os.path.join(new_run_folder, 'correlation_at_epoch_predictions_{:04d}.png'.format(count)))
    
    
    
    
    
    #plt.close('all')
  
    '''
    residual=truths/predictions
    fig, axs = plt.subplots(1, 7, figsize=(40, 7), constrained_layout=True)
    axs = axs.flatten()
    
    i=0
   
    var_name_list=['PT_Lead','Eta_Lead','Phi_Lead','PT_Sub','Eta_Sub','Phi_Sub','DIMuon Mass']
    for idy in axs:   
        idy.scatter( predictions[:, i],residual[:, i])
        idy.set_ylabel('Residual')
        idy.set_xlabel('Generator ' +var_name_list[i])     
        i=i+1

            
    plt.savefig(os.path.join(new_run_folder, 'residual_at_epoch_{:04d}.png'.format(count)))
    #plt.close('all')'''
            
          

            

    fig, axs = plt.subplots(1, 7, figsize=(40, 7), constrained_layout=True)
    axs = axs.flatten()
    
    i=0
   
    var_name_list=['PT_Lead','Eta_Lead','Phi_Lead','PT_Sub','Eta_Sub','Phi_Sub','DIMuon Mass']
    for idy in axs: 
        predictions[:,i]=np.sort(predictions[:,i])
        truths[:,i]=np.sort(truths[:,i])
        idy.scatter( predictions[:, i],truths[:, i])
        idy.set_ylabel('Truth ' +var_name_list[i])
        idy.set_xlabel('Generator ' +var_name_list[i])     
        i=i+1

            
    plt.savefig(os.path.join(new_run_folder, 'listed_correlation_at_epoch_{:04d}.png'.format(count)))
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

    for i in range(len(predictions)):
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
    predictions = np.column_stack((predictions, mass_true))
    

    return truths,predictions
    
    
    
    