import numpy as np

# Load histogram data from separate .npy files
histogram_X = np.load('/sps/atlas/p/pfitzhug/gendatadump/dimuon_generated_events_num_of_events_5000000Model Version: V0.npy')
histogram_Y = np.load('/sps/atlas/p/pfitzhug/gendatadump/dimuon_generated_events_num_of_events_5000000Model Version: V1.npy')
histogram_z = np.load('/sps/atlas/p/pfitzhug/gendatadump/dimuon_generated_events_num_of_events_5000000Model Version: V2.npy')
histogram_a = np.load('/sps/atlas/p/pfitzhug/gendatadump/dimuon_generated_events_num_of_events_5000000Model Version: V3.npy')
histogram_b = np.load('/sps/atlas/p/pfitzhug/gendatadump/dimuon_generated_events_num_of_events_5000000Model Version: V4.npy')
histogram_c = np.load('/sps/atlas/p/pfitzhug/gendatadump/dimuon_generated_events_num_of_events_5000000Model Version: V5.npy')
histogram_d = np.load('/sps/atlas/p/pfitzhug/gendatadump/dimuon_generated_events_num_of_events_5000000Model Version: V6.npy')
histogram_e = np.load('/sps/atlas/p/pfitzhug/gendatadump/dimuon_generated_events_num_of_events_5000000Model Version: V7.npy')
histogram_comb= np.concatenate((histogram_X,histogram_Y,histogram_z,histogram_a,histogram_b,histogram_c,histogram_d,histogram_e), axis=0)
print(len(histogram_X))
print(len(histogram_Y))
print(len(histogram_z))
print(len(histogram_a))
print(len(histogram_b))
#print(histogram_comb)
print(len(histogram_comb))
print(np.shape(histogram_comb))
np.save("/sps/atlas/p/pfitzhug/40milgeneratedevents.npy", histogram_comb)
print("fin")

filename=histogram_comb
lead_pt=filename[:,0]
lead_eta=filename[:,1]
lead_phi=filename[:,2]
sub_pt=filename[:,3]
sub_eta=filename[:,4]
sub_phi=filename[:,5]
dimuon_mass = filename[:,6]

print('Num of events between 110 and 160GeV')
print(((110 < dimuon_mass) & (dimuon_mass < 160)).sum())