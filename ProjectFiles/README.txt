Description of files and their pipeline of the project "Estimating Yield Strength from Material Properties using Machine Learning"
Jan 2024
Minor Computational Science and Engineering
Twan, Guido, Jorn, Marijn and Joey

====================
Data processing:
The MaxDerivative file calcutes the yield point for every sample using the energy method and the offset method
The Merge file, merges the features with the yield points and it can also clean the dataset
--------------------
FEM:

--------------------
Machine learning:

-Gaussian processess (GPs) using SKlearn
	See the GPs Sklearn folder. It contains the folders:
	1. Barplot (4 features): 	creates barplot of the 4 length scales of one GP model.
	2. Filtering_manual_runs_file: 	filters based on a value of the logmarginal likelihood 
					the .txt file given as output by GPs_Sklearn_our_dataset.
	3. GPs_Sklearn_our_dataset:	creates a .txt file containing the model parameters results of 100 trained GP models.
	4. Manual_runs_histogram:	creates a histogram showing how many times each feature ended up in a certain importance position.
	5. Model_performance:		creates a plot showing how one trained GP model performs: predicts on the training (or test) locations

	Each folder contains a Python file with the corresponding (dummy) input files so that the code can be run directly.

	The (recommended) pipeline is as follows:
	First, see folder 5 for grasping some intuition on GPs with Sklearn
	Then, see folder 1 to understand Automatic Relevance Detection (ARD) based on the length scales (one GP run)
	Lastly, see folder 3, followed by folder 2 and then folder 4 to investigate ARD (multiple GP runs)

-Gaussian Processes using Pytorch
	Contains only one Python file, by default it does not do the full optimization. 

-Polynomial Chaos Expansion (PCE)
	This is implemented in the pce.py file. In the beginning of the file are some variables for configuration.
	part of the output is saved in the directory specified in this configuration, the other part is printed.
