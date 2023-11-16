#!/usr/bin/python3.11
import os, sys, csv, math
import pylab as P
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import matplotlib
import time
from alive_progress import alive_bar

def parseCsvFile(filename, column_keys=None,verbose=True):
    ''' Read info from csv file '''
    file_path = data_dir+filename
    if not os.path.isfile(file_path):
        raise Exception(f'CSV file "{filename}" not found')
    if verbose:
        print('Parsing "{}"...'.format(filename))
    column_index = {} # mapping, key=column_key, value=corresponding column index
    data = {} # dict of data, key=column_key, value=data list (floats)
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        line_i = 0 # line index
        for row in csvreader:
            if line_i == 0:
                # Headers. Find interesting columns
                headers = row
                # prepare structure for all columns we want
                if column_keys is None:
                    # grab all data in file
                    column_keys_we_want = [elt.lower() for elt in headers]
                else:
                    # grab only requested data from file
                    assert type(column_keys)==type([])
                    column_keys_we_want = column_keys
                for column_key in column_keys_we_want:
                    data[column_key] = []
                for column_i, elt in enumerate(headers):
                    elt_lower = elt.lower()
                    if elt_lower in column_keys_we_want:
                        column_index[elt_lower] = column_i
                column_indices = list(column_index.values())
                line_i += 1
                if verbose:
                    print('Found columns {0}'.format(column_index))
                continue
            # Data line
            if len(row) < len(headers):
                break # finished reading all data
            for column_key in column_keys_we_want:
                val_s = row[column_index[column_key]]
                if not val_s:
                    val_s = '0'
                try:
                    data[column_key].append(float(val_s))
                except:
                    print('Could not interpret as float the value "{0}" in '\
                        'line {1}: {2}".\nBye!'.format(val_s, line_i+1, row))
                    sys.exit(1)
            line_i += 1
            continue # go to next data line
    if verbose:
        print('Finished parsing csv file')
    return data

def thieleInterpolator(x, y):
    ''' Thiele Reciprocal Difference method for Pade approximate
        From https://rosettacode.org/wiki/Thiele%27s_interpolation_formula#Python
    '''
    rho = [[yi]*(len(y)-i) for i, yi in enumerate(y)]
    for i in range(len(rho)-1):
        rho[i][1] = (x[i] - x[i+1]) / (rho[i][0] - rho[i+1][0])
    for i in range(2, len(rho)):
        for j in range(len(rho)-i):
            rho[j][i] = (x[j]-x[j+i]) / (rho[j][i-1]-rho[j+1][i-1]) + rho[j+1][i-2]
    rho0 = rho[0]
    def t(xin):
        a = 0
        for i in range(len(rho0)-1, 1, -1):
            a = (xin - x[i-1]) / (rho0[i] - rho0[i-2] + a)
        return y[0] + (xin-x[0]) / (rho0[1]+a)
    return t

def computeMaxEnergyDerivative(fig_filename, csv_filename, degree=10, do_show=True,verbose=True):
    ''' Return location (in strain) of maximum of energy derivative
        by approximating the Energy with Gaussian Process (which is now derivable)
        in order to find a precise location (on the approximation)
    '''
    data = parseCsvFile(csv_filename,verbose=verbose)
    loading_vel = 1 #mm/min
    sample_height = 1 #mm
    sample_diameter = 1 #mm
    # recording_interval = 2 #s
    strain_adjust = 0 #0.01647
    area = math.pi*pow(sample_diameter/2,2)

    #Iterations of bayesian regression (loss should be minimal)
    training_iter = 200

    strain_array = np.array(data['time'])/max(np.array(data['time']))
    stress_array = -np.array(data['stress_11_top'])/max(-np.array(data['stress_11_top']))
    energy_array = strain_array * stress_array

    der_energy_array = []
    ##discrete derivative
    for i in range(0,len(energy_array)-1):
      der_energy_array.append((energy_array[i+1]-energy_array[i])/(strain_array[i+1]-strain_array[i]))

    train_x = torch.from_numpy(strain_array[:100]).float()
    train_y = torch.from_numpy(energy_array[:100]).float()

    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()#noise_constraint=gpytorch.constraints.Interval(1e-1,1e1))
    model = ExactGPModel(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # with gpytorch.settings.cholesky_jitter(1e-3):
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    test_x = torch.linspace(0, 1, 100)

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    if verbose:
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))
    
            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            # ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
    
            plt.show()

    # test_x = torch.linspace(0.02, 0.04, 1000)
    X = torch.autograd.Variable(torch.Tensor(test_x), requires_grad=True)
    observed_pred = likelihood(model(X))

    y = observed_pred.mean.sum()
    y.backward()
    dydtest_x = X.grad

    ##Outputs
    test_x_np = test_x.numpy()
    derivative = dydtest_x.numpy()
    i_max = np.argmax(derivative)

    strain_array_true = strain_array * max(np.array(data['time']))/sample_height #- 0.006384
    test_x_np_true = test_x_np * max(np.array(data['time']))/sample_height #- 0.006384
    stress_array_true = stress_array * max(-np.array(data['stress_11_top'])) * 1000 / area
    
    if verbose:
        fig = P.figure(figsize=[6.4, 8])
        # first subplot: strain-stress
        ax = fig.add_subplot(211)
        plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.1)
        plt.plot(strain_array_true, stress_array_true, 'k-')
        plt.axvline(x=test_x_np_true[i_max])
        # plt.xlim(0,0.040)
        # plt.ylim(0,40)
        plt.ylabel('Top stress (MPa)', fontsize=15)
        # plt.legend()
        # Second subplot: derivative energy (approximation)
        ax = fig.add_subplot(212)
        plt.plot(strain_array_true[:-1], der_energy_array, color=(0.5, 0.5, 0.5), linewidth=0.8, linestyle='solid', label='Raw discrete derivative')
        plt.plot(test_x_np_true, derivative, color=(0, 0, 0), linewidth=1.2, linestyle='solid', label='GP approximation')
        plt.axvline(x=test_x_np_true[i_max])
        # plt.xlim(0,0.040)
        # plt.ylim(-20,140)
        plt.yticks([])
        plt.xlabel('Vertical Strain', fontsize=15)
        plt.ylabel('Mechanical work derivative', fontsize=15)
        plt.legend()
    
        if not do_show:
            plt.savefig(fig_filename, format='eps', dpi=1000)
            print('Figure saved as {0}'.format(fig_filename))
        else:
            plt.show()
    if i_max > stress_array_true.shape:
        return None
    else:
        return stress_array_true[i_max]

def output_yield_points(output_title,start,end):
    """Function outputs the yield points of all the FEM Simulations"""
    i = 0
    with open(output_title,'w') as output:
        with alive_bar(end-start) as bar:
            for csv_filename in os.listdir(data_dir):
                i += 1
                if (i > start) and (i <= end):
                    if csv_filename != "Weird_Data":
                        yield_point = computeMaxEnergyDerivative('fig_exp_protocol.eps', csv_filename, do_show=True,verbose=verbose)
                        if yield_point != None: 
                            output.writelines(f"{csv_filename}, {yield_point:.2f}\n")
                        else:   #If the yield point cannot be determined move the file to a different folder.
                            output.writelines(f"{csv_filename}, yield = None, {time.time()-start:.1f}\n") 
                            os.rename(data_dir+csv_filename,data_dir+"Weird_Data/"+csv_filename)
                    bar()

if __name__ == '__main__':
    cwd = os.getcwd() #Get the current directory
    data_dir = cwd + "/data/FEMResults/" #Put the data in this directory
    verbose = False
    num_files = len(os.listdir(data_dir))
    output_title = "output.txt"
    start_file = 0
    end_file = num_files
    if len(sys.argv) > 1:
        output_title = sys.argv[1]
        start_file = int(sys.argv[2])
        end_file = int(sys.argv[3])
    print(f"Calculating yield point for files {start_file} - {end_file} in the data directory and saving in "+output_title)
    output_yield_points(output_title,start_file,end_file) 
