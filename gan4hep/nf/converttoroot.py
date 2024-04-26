
import numpy as np
import ROOT as root
import os


import argparse

parser = argparse.ArgumentParser(description='Normalizing Flow')
add_arg = parser.add_argument
add_arg("--num-rounds", default=5, type=int, help="number of rounds of increased number of statistics")
args = parser.parse_args()

num_rounds=args.num_rounds


num_event_list=[100000,500000,1000000,5000000,10000000,20000000,50000000]

num_event_CR1=[35776,177937,354353,1772094]
num_event_CR2=[44672,223988,447502,2239322,4478688]
num_event_CR3=[44700,224014,447409,2237381,4478676]
num_event_CR4=[39883,199678,400357,1998992,3994900]
num_event_CR=[0,0,0,0,0]


chi2_list1=[]
kol_list1=[]
chi2_list2=[]
kol_list2=[]
chi2_list3=[]
kol_list3=[]
chi2_list4=[]
kol_list4=[]

num_event_list=[5,10,15,20,25,30,35,40]

num_event_list=[10000,50000,100000,500000,1000000,2000000]


if os.path.exists('RootHistograms') == False:
    os.mkdir('RootHistograms')

j=0
for i in range(len(num_event_list)):
    print(i)
    #GENERATED
    #Converting to root the generated data
    counter=str(num_event_list[i])
    print("counter")
    print(counter)
    #filename = np.load("/sps/atlas/p/pfitzhug/" +str(counter) + "milgeneratedevents.npy")
    filename = np.load('/sps/atlas/p/pfitzhug/gendatadump/dimuon_generated_events_num_of_events_' + str(counter) + 'Model Version: V' + str(i)+'.npy')
    lead_pt=filename[:,0]
    lead_eta=filename[:,1]
    lead_phi=filename[:,2]
    sub_pt=filename[:,3]
    sub_eta=filename[:,4]
    sub_phi=filename[:,5]
    dimuon_mass = filename[:,6]

    print('Num of events between 110 and 160GeV')
    print(((110 < dimuon_mass) & (dimuon_mass < 160)).sum())

    file = root.TFile("RootHistograms/SpurGeneratedEvents_Num_of_Events_"+str(counter)+"_ModelVersionV" + str(j)+".root", 'recreate')
    tree = root.TTree("tree_name", "tree title")

    h1 = root.TH1D('h1', 'Invariant Mass (Generated)', 500, 110, 160)

    for k in dimuon_mass:
        #print(k)
        h1.Fill(k)

    #h1copy = h1.Clone("h1")
    #h1.Scale(1./h1.Integral(), "width")

    file.cd()
    h1.Write()

    file.Close()

    '''
    #TRUTH
    #converted output to numpy to easier convert to root for spursig comparison
    counter=str(num_event_list[i])

    filename = np.load('truth_full_FSR.npy')
    
    lead_pt=filename[:,0]
    lead_eta=filename[:,1]
    lead_phi=filename[:,2]
    sub_pt=filename[:,3]
    sub_eta=filename[:,4]
    sub_phi=filename[:,5]
    dimuon_mass = filename[:,6]

    np.random.shuffle(dimuon_mass)
    if j==0:
        num_event_CR=num_event_CR1
    if j==1:
        num_event_CR=num_event_CR2
    if j==2:
        num_event_CR=num_event_CR3
    if j==3:
        num_event_CR=num_event_CR4

    dimuon_mass=dimuon_mass[:num_event_CR[i]]



    print('Num of events between 110 and 160GeV')
    print(((110 < dimuon_mass) & (dimuon_mass < 160)).sum())

    file = root.TFile("RootHistograms/SpurTruthEvents_Num_of_Events_"+str(counter)+"_ModelVersionV" + str(j)+".root", 'recreate')
    tree = root.TTree("tree_name", "tree title")




    dimuon_mass=dimuon_mass[(dimuon_mass >= 110) & (dimuon_mass <= 160)]


    h2 = root.TH1D('h2', 'Invariant Mass (Truth)', 1400, 110, 160)

    for k in dimuon_mass:
        #print(k)
        h2.Fill(k)

    #h2copy = h2.Clone("h2")
    #h2.Scale(1./h2.Integral(), "width")

    
    file.cd()
    h2.Write()


    file.Close()
    '''


    '''

    counter=str(num_event_list[i])
    print(counter)
    #Comparison
    f = root.TFile.Open("RootHistograms/GeneratedEvents_Num_of_Events_"+str(counter)+"_ModelVersionV" + str(j)+".root", "read")
    
    f2 = root.TFile.Open("RootHistograms/TruthEvents_Num_of_Events_"+str(counter)+"_ModelVersionV" + str(j)+".root", "read")
    hist = f.Get('h1;1')
    hist2 = f2.Get('h1;1')

    print(f)    
    print(hist)
    num_event_CR=num_event_CR4
        #kol_list4.append(hist.KolmogorovTest(hist2))
    #chi2_list4.append(hist.Chi2Test(hist2,"UU,NORM,CHI2/NDF,P"))
    #print(hist.Chi2Test(hist2))
    '''













    

#file = root.TFile("treeV2.root", 'recreate')
#tree = root.TTree("tree_name", "tree title")

#h1   = root.TH1D( 'h1', 'Test random', 25, 110, 160 )
