For running Peter's Normalising Flow branch 

1)Installation Instructions

In lxplus
  
mkdir AnyName
cd AnyName
git clone --branch Peter https://github.com/xju2/gan4hep.git
cd gan4hep
pip install -e .
pip3 install pylorentz
cd ..
mkdir nf_work
cd nf_work
ln -s /eos/user/p/pfitzhug/AnyName/gan4hep/gan4hep/nf/train_nf.py

#Add the relevent .output file (mc16d_364100_dimuon_0Jets.output) to the nf_work folder then in nf_work run:

python train_nf.py \
--data dimuon_inclusive mc16d_364100_dimuon_0Jets.output TestNP --max-evts 100000

---------------
***************
---------------

2)Graphs Guide
#Output graphs will be sent to most recent folder in AnyName/nf_work/TestNP/imgs

#There will be a lot of plots!
  
  #image_at_epoch_xxx is the plot of the 6 original and 3 calculate variables with the x axis showing the original values
  #normalised_image_at_epoch_xxx is the plot of the 6 original variables with the x axis having been scaled by StandardScalar -->Off By Default
  #Graph names that start with variables:... are heat plots for the combination of every variable and therefore make up most of the graphs -->Off By Default
  #The three calculated variables each have 2 plots (one log and one non log) for themselves) (variables named 'dimuon' (invariant mass), 'dimuon_pt', and 'dimuon_pseudo')
  #Heatmap_at_epoch_xxx shows the correlation plot for either generated or true data -->Off By Default

