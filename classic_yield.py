#!/usr/bin/python3.12
import os
import sys
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from max_derivative_ML import parseCsvFile
import torch
import gpytorch


def computeMaxEnergyDerivative(fig_filename, csv_filename, degree=10, do_show=True,verbose=True,save=False,training_size=100):
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
    training_iter = 500

    strain_array = np.array(data['time'])/max(np.array(data['time']))
    stress_array = -np.array(data['stress_11_top'])/max(-np.array(data['stress_11_top'])) #Normalized stress array
    energy_array = strain_array * stress_array

    der_energy_array = []
    ##discrete derivative
    for i in range(0,len(energy_array)-1):
      der_energy_array.append((energy_array[i+1]-energy_array[i])/(strain_array[i+1]-strain_array[i]))

    train_x = torch.from_numpy(strain_array).float()
    train_y = torch.from_numpy(energy_array).float()

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

    """if do_show or save:
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(12, 9))
    
            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            mean = observed_pred.mean.numpy()
            ax.plot(test_x.numpy(), mean, 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            # ax.set_ylim([-3, 3])
            ax.set_xlim([min(train_x.numpy())-0.05,max(train_x.numpy())+0.05])
            ax.set_ylim([min(train_y.numpy())-0.05,max(train_y.numpy())+0.05])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            if save:
                plt.savefig("Results/Plots/Confidence/Confidence"+fig_filename,dpi=300)
            if do_show:
                plt.show()
            if not do_show:
                plt.close()"""

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
    stress_array_true = stress_array * max(-np.array(data['stress_11_top'])) * 1000 / area #TODO multiplication factor
    
    gp_stress = (observed_pred.mean.detach().numpy()/ test_x) * (max(-np.array(data['stress_11_top']))) * 1000 / area
    
    if do_show or save:
        fig = P.figure(figsize=[6.4, 8])
        # first subplot: strain-stress
        ax = fig.add_subplot(211)
        plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.1)
        plt.plot(strain_array_true, stress_array_true, '.',label="Observed Data",color=(0.5, 0.5, 0.5))
        plt.plot(test_x_np_true,gp_stress,'-',label="GP approximation",color=(0, 0, 0))
        plt.axvline(x=test_x_np_true[i_max])
        plt.axvline(x=0.001*training_size,color='red',label='training window')
        # plt.xlim(0,0.040)
        plt.ylim(0,max(stress_array_true))
        plt.ylabel('Top stress (MPa)', fontsize=15)
        plt.legend()
        # Second subplot: derivative energy (approximation)
        ax = fig.add_subplot(212)
        plt.plot(strain_array_true[:-1], der_energy_array, color=(0.5, 0.5, 0.5), linewidth=0.8, linestyle='solid', label='Raw discrete derivative')
        plt.plot(test_x_np_true, derivative, color=(0, 0, 0), linewidth=1.2, linestyle='solid', label='GP approximation derivative')
        plt.axvline(x=test_x_np_true[i_max])
        # plt.xlim(0,0.040)
        # plt.ylim(-20,140)
        #plt.yticks([])
        plt.xlabel('Vertical Strain', fontsize=15)
        plt.ylabel('Mechanical work derivative', fontsize=15)
        plt.legend()
    
        if save:
            plt.savefig("Results/Plots/Max/"+fig_filename, dpi=300)
            print('Figure saved as {0}'.format(fig_filename))
        if do_show:
            plt.show()
        if not do_show:
            plt.close()
    if i_max > stress_array_true.shape:
        return None
    else:
        return stress_array_true[i_max]
  

offset = 0.002

loading_vel = 1 #mm/min
sample_height = 1 #mm
sample_diameter = 1 #mm
# recording_interval = 2 #
strain_adjust = 0 #0.01647
area = np.pi*sample_diameter*sample_diameter/4

def gps(strain,stress):
    """
    strain and stress arrays of the same size
    returns new stain and stress arrays 
    """
    X_observed = strain
    y_observed = stress
    # Gaussian Process regression model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1., (1e-2,1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=1e-6)

    # Fit the GP model to the observed data
    gp.fit(X_observed.reshape(-1, 1), y_observed)

    # Evaluate the true function at the point with maximum acquisition value
    x_test = np.linspace(0,0.5*max(strain),len(5*X_observed))
    y_test = gp.predict(x_test.reshape(-1,1))

    x_max = x_test[np.argmax(y_test)]
    return x_test, y_test


def classic_yield_method(strain,stress,offset=0.002):
    slope = (stress[20]-stress[0])/(strain[20]-strain[0])
    offset_stress = slope*(strain-offset)
    yield_index = np.argmin(np.abs(offset_stress-stress))
    yield_strain = strain[yield_index]
    yield_stress = stress[yield_index]
    return yield_strain,yield_stress,offset_stress

def get_yield(filename,offsett=0.001,plot=True,save=False):
    data = np.genfromtxt(filename,delimiter=',',skip_header=1)

    strain = data[:,0]/sample_height
    stress = -data[:,1]*1000/area

    discrete_yield_strain, discrete_yield_stress,offset_stress_discrete = classic_yield_method(strain,stress)

    gpstrain, gpstress = gps2(strain,stress)
    gp_yield_strain, gp_yield_stress, offset_stress_gp = classic_yield_method(gpstrain,gpstress)

    if plot or save:
        plt.figure(figsize=(16,9),dpi=150)
        plt.plot(strain,stress,'.-',label="FEMdata")
        plt.plot(strain[:100],offset_stress_discrete[:100],label=f"{offset*100}% offset discrete",color='r')
        plt.plot(gpstrain[:100],offset_stress_gp[:100],label=f"{offset*100}% offset gp",color='g')
        plt.plot(gpstrain,gpstress,label="gp approximation",color='k')
        plt.axvline(gp_yield_strain,label="yield point gp",color='k')
        plt.axvline(discrete_yield_strain,label="yield point discrete",color='r')
        plt.title(filename.split("/")[-1].split(".csv")[0])
        plt.legend()
    if save:
        plt.savefig("./Results/Plots/Classic/" + filename.split("/")[-1].split(".csv")[0]+".png")
    if plot:
        plt.show()
    if not plot:
        plt.close()
    return discrete_yield_strain, gp_yield_strain
    

if __name__== '__main__':
    cwd = os.getcwd()
    data_dir = cwd + "/data/FEMResults/"
    start = 0
    end = len(os.listdir(data_dir))
    i = 0

    if len(sys.argv) > 1:
        plot_file = sys.argv[1]
    else:
        plot_file = "001_005"

    get_yield(data_dir+"Scan"+plot_file+".csv",save=True,offsett=offset)
    """
    with open("output_classic.txt",'w') as output:
        with alive_bar(end-start) as bar:
            for csv_filename in os.listdir(data_dir):
                i += 1
                if (i > start) and (i <= end):
                    if csv_filename != "Weird_Data":
                        discrete_yield, gpyield = get_yield(data_dir+csv_filename,plot=False,save=False)
                        output.writelines(f"{csv_filename.split('/')[-1]},{discrete_yield},{gpyield},{gpyield-discrete_yield}\n")
                bar()"""

    
