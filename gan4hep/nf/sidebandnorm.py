

import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
counter_list=[5,10,15,20,25,30,35,40]
#counter_list=[10000,50000,100000,500000,1000000,2000000]
for i in range(len(counter_list)):

    #Retrieving NF generated root file
    counter=counter_list[i]
    j=0
    genfile = ROOT.TFile.Open("../NF_November/gan_work/RootHistograms/SpurGeneratedEvents_Num_of_Events_"+str(counter)+"_ModelVersionV" + str(j)+".root", 'READ')
    hist_gen = genfile.Get("h1")

    # Draw the generated events histogram
    canvas = ROOT.TCanvas("canvas", "My Canvas")
    hist_gen.Draw("hist")
    canvas.SaveAs("histogram_gen_"+str(counter_list[i])+".png")



    #Retrieve run 2 bkgd root files
    genfile2 = ROOT.TFile.Open("/home/pfitzhug/sidebandnorm/combined_histogram.root", 'READ')
    hist_data = genfile2.Get("hist")
    histcombinedbkgd = genfile2.Get("hist")

    # Draw the MC bkgd histogram
    canvas = ROOT.TCanvas("canvas", "My Canvas")
    hist_data.Draw("hist")
    canvas.SaveAs("histogram_data_"+str(counter_list[i])+".png")


    # Create a third histogram for the ratio
    low = 120
    high = 130
    print(hist_gen)
    print(hist_data)
    #Create the ratio
    hist_ratio = hist_data.Clone("hist_ratio")
    hist_ratio.Divide(hist_gen)

    #only fill the sidebands of the ratio
    for bin in range(1, hist_ratio.GetNbinsX()+1):
        #print(hist_ratio.GetBinContent(bin))
        if hist_ratio.GetBinCenter(bin) < high and hist_ratio.GetBinCenter(bin) > low:
            hist_ratio.SetBinContent(bin, 0)
            hist_ratio.SetBinError(bin, 0)
        #else:
        #hist_ratio.SetBinContent(bin, 0)


    # Draw the ratio histogram
    canvas = ROOT.TCanvas("canvas", "My Canvas")
    hist_ratio.Draw("hist")
    canvas.SaveAs("histogram_ratio_"+str(counter_list[i])+".png")



    # Define a 2nd order polynomial function
    #func = ROOT.TF1("func", "pol2", 110, 160,"WW")
    func = ROOT.TF1("p2", "[0]*x*x*x + [1]*x*x + [2]*x + [3]", 110, 160)  # Adjust the range as needed
    # Fit the ratio histogram with the fitted func
    hist_ratio.Fit(func)
    num_params = func.GetNpar()

    #func.SetParameter(0,poly_func[0])
    #func.SetParameter(1,poly_func[1])
    #func.SetParameter(2,poly_func[2])
    #Retreieve Parameter values
    p0=func.GetParameter(0)
    p1=func.GetParameter(1)
    p2=func.GetParameter(2)
    p3=func.GetParameter(3)
    num_params = func.GetNpar()
    for j in range(num_params):
        param_value = func.GetParameter(j)
        param_name = func.GetParName(j)
        print(f"Parameter {j}: {param_name} = {param_value}")
    # Draw the histogram
    canvas = ROOT.TCanvas("canvas", "My Canvas")

    hist_ratio.Draw()

    # Draw the histogram
    canvas = ROOT.TCanvas("canvas", "My Canvas")
    hist_ratio.Draw("hist")
    hist_ratio.SetLineColor(ROOT.kBlue)
    func.Draw("same")


    #Save png
    canvas.SaveAs("histogram_sideband_fit_"+str(counter_list[i])+".png")

    ##################################
    #Apply Reweighting
    # Create a new histogram with the same binning as the original
    reweightedhist = ROOT.TH1F("reweighted_hist", "Reweighted Histogram", hist_gen.GetNbinsX(), hist_gen.GetXaxis().GetXmin(), hist_gen.GetXaxis().GetXmax())

    # Loop over bins and reweight

    #reweightedhist.SetParameters(func.GetParameter(0), func.GetParameter(1), func.GetParameter(2))
    

    

    # Scale scaled_hist using the polynomial
    for j in range(1, hist_gen.GetNbinsX() + 1):
        x = hist_gen.GetBinCenter(j)
        #print(" ")
        #print("j value: ",j)
        #print("x value: ",x)
        #print(p0)
        #print(p1)
        #print(p2)
        #print("bin content value at x: ",hist_gen.GetBinContent(j))
        #scale_factor = func.Eval(x_center)
        scale_factor=p0*(x*x*x)+p1*(x*x)+p2*x+p3
        #print("scale factor value: ",scale_factor)
        #print("bin content * scale factor value: ",hist_gen.GetBinContent(j) * scale_factor)
        reweightedhist.SetBinContent(j, hist_gen.GetBinContent(j) * scale_factor)
        num_params = func.GetNpar()
        # Print each parameter's value and name
        for j in range(num_params):
            param_value = func.GetParameter(j)
            param_name = func.GetParName(j)
            #print(f"Parameter {j}: {param_name} = {param_value}")
        
    # Save the new histogram to a ROOT file
    output_file = ROOT.TFile.Open("reweighted_histogram_"+str(counter_list[i])+".root", "RECREATE")

    canvas = ROOT.TCanvas("canvas", "My Canvas")
    reweightedhist.Draw("hist")
    canvas.SaveAs("histogram_reweighted_"+str(counter_list[i])+".png")
    reweightedhist.Write()
    #output_file.Close()
    print(f"Reweighted histogram saved as 'reweighted_histogram_"+str(counter_list[i])+".root'")




    ###################
    #Create plot of all histograms at once
    # Create a canvas
    c2 = ROOT.TCanvas('c2', 'Histograms', 800, 600)
    c2.SetLogy()
    # Draw the first histogram
    reweightedhist.Draw()
    # Draw the second histogram on the same canvas


    hist_data.Draw("SAME")
    hist_data.SetLineColor(ROOT.kGreen)
    hist_ratio.Draw("SAME")
    hist_ratio.SetLineColor(ROOT.kOrange)
    hist_gen.Draw("SAME")
    func.Draw("same")
    func.SetLineColor(ROOT.kBlack)
    hist_gen.SetLineColor(ROOT.kRed)

    # Save the canvas as an image
    c2.SaveAs("Superimposed_"+str(counter_list[i])+".png")



    ###################
    #Create plot of how the fit should look like
    x=np.linspace(-10,160,200)
    y=p0*x**2+p1*x+p2
    plt.plot(x,y)
    plt.xlabel("hmumu")
    plt.ylabel("fit data/mc ratio")
    plt.grid(True)
    plt.show
    plt.savefig("fig.png")

    #x=110
    #print(0.908189*x**2-0.00798059*x+4.662e-05)


    #######################
    #Optional normaliing method
    hist_gen.Scale(hist_data.Integral()/hist_gen.Integral())

    # Save the new histogram to a ROOT file
    output_file = ROOT.TFile.Open("stupid_histogram_"+str(counter_list[i])+".root", "RECREATE")



    canvas = ROOT.TCanvas("canvas", "My Canvas")
    hist_gen.Draw("hist")

    canvas.SaveAs("histogram_stupid_"+str(counter_list[i])+".png")

    hist_gen.Write()
    #output_file.Close()
    print(f"Reweighted histogram saved as 'stupid_histogram_"+str(counter_list[i])+".root'")
