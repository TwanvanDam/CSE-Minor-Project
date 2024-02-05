#####################
###### IMPORTS ######
#####################
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float64)

################################
###### Analysis variables ######
################################
nfeatures = 4
# If true do whole optimization, for n_iterations
# optimaztion takes very long so default is false
optimization = False
n_iterations = 30

#####################
##### Functions #####
#####################
# Kernel function for #dimensions=len(x1)
def kappa(x1, x2, **hyperparams):
    sig_f = hyperparams["sig_f"]
    length = hyperparams["length"]
    sum=0
    for i in range(len(x1)):
        sum += (1/length[i]**2) * np.abs(x1[i] - x2[i])**2
    arg = sig_f**2 * torch.exp(-0.5*sum)
    return arg
# Create covariance matrix
def cov_gaussian(X1, X2, **hyperparams):
    
    sig_f = hyperparams["sig_f"]
    length = hyperparams["length"]

    N = len(X1)
    c = len(X2)

    cov = torch.zeros((N,c))
    for i in range(N):
        for j in range(c):
            cov[i,j] = kappa(X1[i], X2[j], **hyperparams)
    return cov

# Initial conditions
hyperparams = {
    "sig_f": 1,
    "length": torch.ones((1,nfeatures))[0],
    "noise": 0.4,
}

cov_func = cov_gaussian

# Actual GP
def GP(X, t, X_hat, kernel, hyperparams):
    
    """
    :param X: Observation locations
    :param t: Observation values
    :param X_hat: Prediction locations
    :param kernel: covariance function
    :param hyperparams: The hyperparameters
    :return: posterior mean and covariance matrix
    """
    with torch.no_grad():
        noise = hyperparams["noise"]  # Note: noise**2 == beta^(-1)

        # Kernel of the observations
        k11 = kernel(X, X, **hyperparams)
        # Kernel of observations vs to-predict
        k12 = kernel(X, X_hat, **hyperparams)

        C_inv = torch.inverse(k11 + noise**2 * torch.eye(k11.shape[0]))

        kT_Cinv = torch.matmul(k12.T, C_inv)

        # Compute posterior mean
        mu = torch.matmul(kT_Cinv, t)  # Bishop 6.66

        # Compute the posterior covariance
        C = kernel(X_hat, X_hat, **hyperparams) + noise**2 * torch.eye(X_hat.shape[0])
        cov = C - torch.matmul(kT_Cinv, k12)  # Bishop 6.67
    return mu, cov
# Log marginal likelihood of observations (X,t)
def GP_logmarglike(X, t, kernel, hyperparams):    
    # Kernel of the observations
    k11 = kernel(X, X, **hyperparams)

    C = k11 + hyperparams["noise"] ** 2 * torch.eye(k11.shape[0])

    logmarglike = (
        -0.5 * torch.sum(torch.log(torch.diagonal(C)))
        - 0.5 * torch.matmul(t.T, torch.matmul(torch.inverse(C), t))
        - 0.5 * len(t) * torch.log(torch.tensor(2 * torch.pi))
    )  
    return logmarglike


###########################################
################ Read data ################
###########################################
dataframe1 = pd.read_excel(r"ActualProject\merged_data_offset_final.xlsx")  # Cleaned data
npoints_obs = 584  # 584 datapoints in total
# 90% of data is training
n1 = round(npoints_obs*0.9)
n2 = npoints_obs - n1

# Combine all 
X_full = dataframe1.iloc[:, 1:nfeatures+1]
X_full = X_full.to_numpy()
y_full = dataframe1.iloc[:, -1]
y_full = np.array([y_full.to_numpy()])
X_full = np.concatenate((X_full, y_full.T), axis=1)

# Randomly sample training and predicitions/test
np.random.shuffle(X_full)
y_train = torch.tensor(X_full[:n1, -1])
X_train = torch.tensor(X_full[:n1, :-1])
y_pred = torch.tensor(X_full[n1:, -1])
X_pred = torch.tensor(X_full[n1:, :-1])
feature_names = dataframe1.columns[1:nfeatures+1]


############################################
########### Preform optimization ###########
############################################
if optimization:

    # Initial values
    sig_init = 1.0
    noise_init = 1.0
    length_init = torch.ones((1,nfeatures))[0]

    # Initialize logsigma, lognoise, and loglengthscale.
    logsig = torch.log(torch.tensor([sig_init])).requires_grad_(True)
    lognoise = torch.log(torch.tensor([noise_init])).requires_grad_(True)
    loglen = torch.log(length_init).requires_grad_(True)

    # We use the bfgs optimizer.
    bfgs = torch.optim.LBFGS((logsig, lognoise, loglen), max_iter=n_iterations)

    sigmas = []
    noises = []
    lens = []

    # This function is called by the optimizer, it cannot have any arguments, therefore we use the global variables.
    def update():
        bfgs.zero_grad()
        hyperparams["sig_f"] = torch.exp(logsig)
        hyperparams["noise"] = torch.exp(lognoise)
        hyperparams["length"] = torch.exp(loglen)
        if (torch.exp(loglen).detach()[1].item() >= 5):
            hyperparams["length"][1] = 1

        loss = -GP_logmarglike(X_train, y_train, cov_func, hyperparams)

        # Check for NaN in the loss
        if torch.isnan(loss):
            print("NaN loss encountered. Stopping optimization.")
            return loss

        loss.backward()
        
        print(torch.exp(logsig).detach()[0].item(), torch.exp(lognoise).detach()[0].item())
        print([torch.exp(loglen).detach()[0].item(), torch.exp(loglen).detach()[1].item(), torch.exp(loglen).detach()[2].item(), torch.exp(loglen).detach()[3].item()])
        # print([torch.exp(loglen).detach()[0].item(), torch.exp(loglen).detach()[1].item()])
        sigmas.append(torch.exp(logsig).detach()[0].item())
        noises.append(torch.exp(lognoise).detach()[0].item())
        lens.append([torch.exp(loglen).detach()[0].item(), torch.exp(loglen).detach()[1].item()])
        return loss


    # We perform a single step. In this step, multiple iterations of the optimizer are performed.
    bfgs.step(update)

    print(
        f"BFGS optimal result: sigma_f: {sigmas[-1]}, noise: {noises[-1]}, length scale: {lens[-1]}"
    )
    print(hyperparams["length"])

    normal_length_scales = hyperparams["length"]
    fig1 = plt.figure()
    y_pos = np.arange(nfeatures)
    plt.barh(y_pos, normal_length_scales.detach().numpy())
    plt.yticks(y_pos, labels=feature_names, fontsize=7)
    plt.xlabel('Log length scales')
    plt.xscale("log")
    plt.gca().invert_yaxis()  # labels read top-to-bottom
        # plt.title(
        #     (
        #         f"Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: "
        #         f"{gpr.log_marginal_likelihood(gpr.kernel_.theta)}"
        #     ),
        #     fontsize=8,
        #     )
    plt.legend()
    plt.savefig("ActualProject/BarPlot.png")
    plt.show()

##################################
######### Train the GP ###########
##################################
plt.figure()
if optimization == False:
    hyperparams["sig_f"] = 9.031901804214803
    hyperparams["noise"] = 15.710805980839417
    hyperparams["length"] = [0.14027438961812003, 0.5847724723614013, 0.9996002949299121, 0.9999999643330146]

# Compute posterior mean and covariance
print("Train GP")
mu_pred, cov_pred = GP(X_train, y_train, X_pred, cov_func, hyperparams)
# Compute the standard deviation at the test points to be plotted
sig = torch.sqrt(torch.diag(cov_pred))

plt.scatter(y_pred, mu_pred, color="blue", label=f"Predictions")
x_ = np.linspace(torch.min(y_pred)-5, torch.max(y_pred)+5, 2000)
plt.plot(x_, x_, color="black", linestyle="--", label="Truth values")
plt.legend()
plt.xlabel("y_pred"), plt.ylabel("y_test (from GP)")
plt.savefig("ActualProject/PlotPredictions.png")
plt.show()
