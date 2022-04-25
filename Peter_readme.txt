For running Peter's Normalising Flow branch 

1)Installation Instructions

In lxplus (Requires python 3.9, tensorflow, and pylorentz)
  
mkdir AnyName
cd AnyName
git clone --branch Peter https://github.com/xju2/gan4hep.git
cd gan4hep
pip install -e .
pip3 install pylorentz #If not already installed
cd ..
mkdir nf_work
cd nf_work
ln -s /eos/user/p/pfitzhug/AnyName/gan4hep/gan4hep/nf/train_nf.py
ln -s /eos/user/p/pfitzhug/AnyName/gan4hep/gan4hep/nf/Hmumu_plots.py

#Add the relevent .output file (mc16d_364100_dimuon_0Jets.output) to the nf_work folder then in nf_work run:

python train_nf.py \
--data dimuon_inclusive mc16d_364100_dimuon_0Jets.output TestNP --max-evts 100000

---------------
***************
---------------

2) Display Guide

5 numbers should be printed every epoch that are in order during the training:

Current Epoch, Current Loss, Current Wasserstein Distance, Best Wasserstein Distance, Best Epoch

At the end of the training, a file called Temp_Data should be created in the nf_work folder that contains 5 files:

predictions.npy -- Contains the generated events from the best epoch
truths.npy -- Contains the true test events used in the best epoch
loss_list.npy -- List of the log loss values from each epoch
w_list.npy -- List of the Wasserstein distances from each epoch
filename.txt -- Contains the file name of the current run

These files will also be saved in the corresponding run folder in /TestNP/imgs if you need to retrieve them later

---------------
***************
---------------

3) Plotting

To produce the required plots run:

python Hmumu_plots.py  \
--jetnum 0 --bestepoch

The jetnum argument is needed to assign what the X-labels should be for the plots and how many variables there are to plot depending on the number of variables

bestepoch argument is used when naming the save figure to display what epoch was used to produced these plots

Once run, all plots will be saved in the corresponding file saved in filename.txt

---------------
***************
---------------

4)Graphs Guide

Whilst running you will recieve this printout

Length of predicted data set pre-cuts:  9999
Length of truth data set pre-cuts:  9999
Length of predicted data set post-cuts:  9085
Fraction of events lost :  0.09140914091409136

This is due to the selection cuts making sure all eta and phi values are between -pi and pi and displaying what events get lost in the process

Each series of plots has its own function in Hmumu_plots.py that cna be commented out depending on what you want to plot (Except log loss and wass distance plots which will always be created at the start of the file):


    #Plots
    # Plot Original Variables
    original(predictions_cut,truths_cut,new_run_folder,xlabels_extra,county)
  
    Plots the original variables given in the .output file for both truth and generated data. Saved as image_at_epoch_xxx

    # Plot Calculated Variables
    calculated(predictions_cut,truths_cut,new_run_folder,xlabels_extra,xlabels,county)

    Plots the 3 extra calculated variables (dimuon mass, pt, and eta). Saved as calc_image_at_epoch_xxx

    # Plot ratio plots
    ratio(predictions_cut,truths_cut,new_run_folder,xlabels_extra,xlabels,county)

    Plots the ratio of the truth and generated histograms for the original and calculated data. Saved as calc_ratio_image_at_epoch_xxx.png and      image_at_epoch_xxx.png

    # Plot correlation plots
    heatmap(predictions_cut,truths_cut,new_run_folder,xlabels_extra,county)
    
    Plots the correlation between all variables for both truth and generated data. Saved as heatmap_at_epoch_generated_xxx.png and heatmap_at_epoch_truths_xxx.png


    #Plot Individual Heat Map plots
    var_corr(predictions_cut,truths_cut,new_run_folder,xlabels_extra,county)

    Plot heatmaps for every combination of variables (This will take the longest to run)
    
    # Plot Large plots for calculated variables
    large_calc_plots(truths,predictions_cut,xlabels_extra,county,new_run_folder)

    Plots larger histograms for the three calculate variables seperately with SD and mean lines included for both generated and true data
