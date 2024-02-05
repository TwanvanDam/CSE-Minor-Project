import os, sys, csv, math
import pylab as P
import numpy as np
import torch
import gpytorch
import matplotlib
matplotlib.use('TkAgg')  # Use a different backend, like TkAgg
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
 

def parseCsvFile(filename, column_keys=None,verbose=True):
    ''' Read info from csv file '''
    # Read data from csv file
    # Data is to be put in the data_dir directory
    file_path = data_dir + filename
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

def gps_without_grad(strain,stress,training_iter=3500,verbose=True):
    ''' Return the stress-strain curve interpolated by a Gaussian Process
        without the gradient of the stress-strain curve
        Similar to the function computeMaxEnergyDerivative2 but without the gradient'''
    
    train_x = torch.from_numpy(strain).float()
    train_y = torch.from_numpy(stress/np.mean(stress)).float()

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
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-7,1e3))
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
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.8f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
        ))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    test_x = torch.linspace(0, 0.15, 250)

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    # test_x = torch.linspace(0.02, 0.04, 1000)
    X = torch.autograd.Variable(torch.Tensor(test_x), requires_grad=True)

    observed_pred = likelihood(model(X))

    y = observed_pred.mean.detach().numpy()
    return test_x.numpy(), y*np.mean(stress)

def computeMaxEnergyDerivative2(fig_filename, csv_filename, degree=10, do_show=True,confidence=False,training_size=100,verbose=True,iterations=500):
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
    training_iter = iterations

    strain_array = np.array(data['time'])/max(np.array(data['time']))
    stress_array = -np.array(data['stress_11_top'])/max(-np.array(data['stress_11_top']))
    energy_array = strain_array * stress_array

    der_energy_array = []
    ##discrete derivative
    for i in range(0,len(energy_array)-1):
      der_energy_array.append((energy_array[i+1]-energy_array[i])/(strain_array[i+1]-strain_array[i]))

    window_size = 11  # Increase the window size until smooth
    
    #not used in this version
    der_energy_array_smoothed = savgol_filter(energy_array, window_size, 2, deriv=1, delta=strain_array[1]-strain_array[0])

    train_x = torch.from_numpy(strain_array[:training_size]).float()
    train_y = torch.from_numpy(energy_array[:training_size]).float()

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
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-7,1e3))
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
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.8f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
        ))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    test_x = torch.linspace(0, 1, 500)

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    # test_x = torch.linspace(0.02, 0.04, 1000)
    X = torch.autograd.Variable(torch.Tensor(test_x), requires_grad=True)

    observed_pred = likelihood(model(X))

    y = observed_pred.mean.sum()
    torch.autograd.backward(y, retain_graph=True)
    dydX = torch.autograd.grad(y, X, create_graph=True)[0]

    # Calculate the second derivative
    d2ydX2 = torch.autograd.grad(dydX.sum(), X, create_graph=True)[0]
    ##Outputs
    test_x_np = X.detach().numpy()
    derivative = dydX.detach().numpy()
    secondderivative = d2ydX2.detach().numpy()[30:150]
    
    secondderivative /= max(secondderivative)
    
    max_second = close_to_zero_algorithm(secondderivative)

    i_max = np.argmax(derivative[:100])

    strain_array_true = strain_array * max(np.array(data['time']))/sample_height #- 0.006384
    test_x_np_true = test_x_np * max(np.array(data['time']))/sample_height #- 0.006384
    stress_array_true = stress_array * max(-np.array(data['stress_11_top'])) * 1000 / area
    
    gp_interpolated_strain, gp_interpolated_stress = gps_without_grad(strain_array_true,stress_array_true,verbose=verbose)

    yield_strain2gp, offset_stress2gp = get_yield(gp_interpolated_stress,gp_interpolated_strain,offset=0.002,slope=(stress_array_true[10]-stress_array_true[0])/(strain_array_true[10]-strain_array_true[0]))
    yield_stress2gp = gp_interpolated_stress[gp_interpolated_strain==yield_strain2gp][0]

    yield_tangent, gradient = tangent_algorithm(test_x_np_true, secondderivative)
    tangent_line = gradient*(test_x_np_true[50:120]-yield_tangent)
    yield_stress_tangent = gp_interpolated_stress[np.argmin(np.abs(gp_interpolated_strain-yield_tangent))]
    
    fig = P.figure(figsize=[10, 8])
    # first subplot: strain-stress
    ax = fig.add_subplot(211)
    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.1)
    plt.xlim([0,0.15])
    plt.plot(strain_array_true, stress_array_true, '.',label="FEM datapoints",color=(0.5,0.5,0.5))
    plt.plot(gp_interpolated_strain,gp_interpolated_stress,'k-',label="GP approximation to FEM datapoints")
    plt.plot(gp_interpolated_strain[gp_interpolated_strain<0.5*np.max(gp_interpolated_strain)],offset_stress2gp[gp_interpolated_strain<0.5*np.max(gp_interpolated_strain)],'-',label="Offset stress")
    plt.axvline(yield_tangent,label="Tangent to linear second derivative yield point",color='r')
    plt.axvline(x=test_x_np_true[i_max],label="max first derivative",color='b')
    plt.axvline(yield_strain2gp,label="Classic 0.2% yield point",color='green')
    plt.ylabel('Top stress (MPa)', fontsize=15)
    plt.legend(fontsize='small',loc='lower right')
    
    # Second subplot: derivative energy (approximation)
    ax = fig.add_subplot(212)
    plt.axhline(0,linewidth=0.5,color='k')
    plt.xlim([0,0.15])
    plt.plot(strain_array_true[:-1], der_energy_array, color=(0.5, 0.5, 0.5), linewidth=0.8, linestyle='solid', label='Raw discrete derivative')
    plt.plot(test_x_np_true[:200], derivative[:200], 'k',linestyle='solid', label='GP approximation first derivative')
    plt.plot(test_x_np_true[30:150], secondderivative, linestyle='--',color='k', label='GP approximation second derivative')
    plt.plot(test_x_np_true[50:120], tangent_line, linestyle='--',color='r', label='Tangent to linear second derivative yield point')
    plt.axvline(x=test_x_np_true[i_max],label="max first derivative",color='b')
    plt.axvline(x=test_x_np_true[max_second],label="second derivative"+r"$\approx0$",color='orange')
    plt.axvline(yield_strain2gp,label="Classic 0.2% yield point",color='green')
    plt.axvline(yield_tangent,label="Tangent to linear second derivative yield point",color='r')
    plt.yticks([0])
    plt.xlabel('Vertical Strain', fontsize=15)
    plt.ylabel('Mechanical work derivative', fontsize=15)
    plt.legend(fontsize='small',loc='lower right')

    if not do_show:
        plt.savefig("Results/Plots/Max/"+fig_filename, format='png', dpi=300)
        print(f'Figure saved as {fig_filename}')
        plt.close()
    else:
        plt.show()
    return yield_stress_tangent,yield_stress2gp

def close_to_zero_algorithm(array):
    """
    Finds the index of the smallest value in the given array that is close to zero. Array is already sliced at 30:150.

    Parameters:
    array (numpy.ndarray): The input array.

    Returns:
    int: The index of the smallest value close to zero in the array.
    """
    if len(np.where(np.abs(array)<0.05)[0] != 0):
        smallest_value = 30+np.where(np.abs(array)<0.05)[0][0]
    else:
        smallest_value = 30+np.argmin(np.abs(array[:-((len(array)//2)-30)]))
    return smallest_value


def tangent_algorithm(x_array, y_array):
    """
    Calculates the yield strain and gradient using the tangent algorithm.

    Parameters:
    x_array (array-like): Array of x-values.
    y_array (array-like): Array of y-values.

    Returns:
    tuple: A tuple containing the yield strain and gradient.
    """
    #max index approx
    x_array = x_array[30:150]
    zero_der2_tol = 0.4 #threshold to find the moment when second derivative is linear
    for i_max_approx in range(0,len(x_array)):
        if y_array[i_max_approx] < zero_der2_tol:
            break
    gradient = np.gradient(y_array,x_array)[i_max_approx]
    yield_strain = x_array[i_max_approx] - y_array[i_max_approx]/gradient
    return yield_strain, gradient


def get_yield(stress, strain, offset=0.002, discrete=True, slope=None):
    """
    Calculate the yield strain and offset stress based on stress-strain data.

    Parameters:
    stress (array-like): Array of stress values.
    strain (array-like): Array of strain values.
    offset (float, optional): Offset value for calculating the offset stress. Default is 0.002.
    discrete (bool, optional): If True, calculate the slope using the first 10 data points. If False, use the first 20 data points. Default is True.
    slope (float, optional): User-defined slope value. If provided, it will be used instead of calculating the slope from the data points.

    Returns:
    tuple: A tuple containing the yield strain and offset stress.

    """
    if slope is None:
        if discrete:
            slope = (stress[10] - stress[0]) / (strain[10] - strain[0])
        else:
            slope = (stress[20] - stress[0]) / (strain[20] - strain[0])
    else:
        slope = slope
    offset_stress = slope * (strain - offset)

    yield_index = np.argmin(np.abs(stress - offset_stress))
    yield_strain = strain[yield_index]

    return yield_strain, offset_stress

def make_plots(filename,show=True,training_size=100,verbose=True,confidence=False,iterations=500):
    linear_stress, gpyield = computeMaxEnergyDerivative2(filename.replace(".csv",".png"),filename,training_size=training_size,verbose=verbose,do_show=show,confidence=confidence,iterations=iterations)
    return gpyield, linear_stress

if __name__ == '__main__':
    data_dir = "../data/FEMResults/" #Put the data in this directory

    make_plots("Scan001_002.csv",training_size=100,confidence=True,iterations=500,verbose=False)