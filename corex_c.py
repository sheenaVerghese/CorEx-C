#?check paper2_rough.txt for test data

"""Maximally Informative Representations using CORrelation EXplanation
#in spyder: debugfile('/home/bb79e/Code/Bin_classify/bb79e3/vis_mis_iris2_test.py', args='/home/bb79e/Code/Dataset/Iris_train_wclass.csv -f -t -c -l 3 -m -1 -o /home/bb79e/Code/Bin_classify/bb79e3/testCorex_log_iris',wdir='/home/bb79e/Code/Bin_classify/bb79e3')
Greg Ver Steeg and Aram Galstyan. "Maximally Informative
Hierarchical Representations of High-Dimensional Data"
AISTATS, 2015. arXiv preprint arXiv:1410.7404.

Code below written by:
Greg Ver Steeg (gregv@isi.edu), 2015.

License: Apache V2 (This development version not yet released)

Modified by: Sheena Leeza-Verghese
Modifications:
1) Binary classification for one layer by calling call-bin
"""

import numpy as np  # Tested with 1.8.0
from os import makedirs
from os import path
from numpy import ma
#from scipy.misc import logsumexp  # Tested with 0.13.0 --ori code
from scipy.special import logsumexp
from multiprocessing import Pool
import pandas as pd
from scipy.stats import gaussian_kde
import os
import sys

#np.set_printoptions(threshold=sys.maxsize)
def unwrap_f(arg):
    """Multiprocessing pool.map requires a top-level function."""
    return Corex.calculate_p_xi_given_y(*arg)

def logsumexp2(z):
    """Multiprocessing pool.map requires a top-level function."""
    return logsumexp(z, axis=2)


class Corex(object):
    """
    Correlation Explanation

    A method to learn a hierarchy of successively more abstract
    representations of complex data that are maximally
    informative about the data. This method is unsupervised,
    requires no assumptions about the data-generating model,
    and scales linearly with the number of variables.

    Code follows sklearn naming/style (e.g. fit(X) to train)

    Parameters
    ----------
    n_hidden : int, optional, default=2
        Number of hidden units.

    dim_hidden : int, optional, default=2
        Each hidden unit can take dim_hidden discrete values.

    max_iter : int, optional
        Maximum number of iterations before ending.

    n_repeat : int, optional
        Repeat several times and take solution with highest TC.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode. 1 outputs TC(X;Y) as you go
        2 output alpha matrix and MIs as you go.

    seed : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    labels : array, [n_hidden, n_samples]
        Label for each hidden unit for each sample.

    clusters : array, [n_visible]
        Cluster label for each input variable.

    p_y_given_x : array, [n_hidden, n_samples, dim_hidden]
        The distribution of latent factors for each sample.

    alpha : array-like, shape (n_components,)
        Adjacency matrix between input variables and hidden units. In range [0,1].

    mis : array, [n_hidden, n_visible]
        Mutual information between each (visible/observed) variable and hidden unit

    tcs : array, [n_hidden]
        TC(X_Gj;Y_j) for each hidden unit

    tc : float
        Convenience variable = Sum_j tcs[j]

    tc_history : array
        Shows value of TC over the course of learning. Hopefully, it is converging.

    References
    ----------

    [1]     Greg Ver Steeg and Aram Galstyan. "Discovering Structure in
            High-Dimensional Data Through Correlation Explanation."
            NIPS, 2014. arXiv preprint arXiv:1406.1222.

    [2]     Greg Ver Steeg and Aram Galstyan. "Maximally Informative
            Hierarchical Representations of High-Dimensional Data"
            AISTATS, 2015. arXiv preprint arXiv:1410.7404.

    """
    def __init__(self, n_hidden=2, dim_hidden=2,            # Size of representations
                 max_iter=100, n_repeat=1, ram=8., max_samples=1000, n_cpu=1,   # Computational limits
                 eps=1e-5, marginal_description='gaussian', smooth_marginals=False,    # Parameters
                 missing_values=-1, seed=None, verbose=False,outfile=""):
        
        
        self.dim_hidden = dim_hidden  # Each hidden factor can take dim_hidden discrete values
        self.n_hidden = n_hidden  # Number of hidden factors to use (Y_1,...Y_m) in paper
        self.missing_values = missing_values  # For a sample value that is unknown

        self.max_iter = max_iter  # Maximum number of updates to run, regardless of convergence
        self.n_repeat = n_repeat  # Run multiple times and take solution with largest TC
        self.ram = ram  # Approximate amount of memory to use in GB
        self.max_samples = max_samples  # The max number of samples to use for estimating MI and unique info
        self.n_cpu = n_cpu  # number of CPU's to use, None will detect number of CPU's
        self.pool = None  # Spin up and close pool of processes in main loop (depending on self.n_cpu)

        self.eps = eps  # Change in TC to signal convergence
        self.smooth_marginals = smooth_marginals  # Less noisy estimation of marginal distributions
        #set by SHeena : start--
        seed=1234
        #count method for debugging purpose only
        self.countS=0
        self.countS2=0
        self.loopCount=0
        #end debugging method
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        self.fileName = outfile+"/"
        #end set by SHeena --
        
        np.random.seed(seed)  # Set seed for deterministic results
        self.verbose = verbose
        if verbose > 0:
            np.set_printoptions(precision=3, suppress=True, linewidth=200)
            print ('corex, rep size:', n_hidden, dim_hidden)
        if verbose:
            np.seterr(all='ignore')
            # Can change to 'raise' if you are worried to see where the errors are
            # Locally, I "ignore" underflow errors in logsumexp that appear innocuous (probabilities near 0)
        else:
            np.seterr(all='ignore')
        self.tc_min = 0.01  # Try to "boost" hidden units with less than tc_min. Haven't tested value much.
        self.marginal_description = marginal_description
        if verbose:
            print ("Marginal description: ", marginal_description)
        if (self.marginal_description =='discrete'):
            self.marginal_xi_dict=np.array([])

    def label(self, p_y_given_x):
        """Maximum likelihood labels for some distribution over y's"""
        return np.argmax(p_y_given_x, axis=2).T

    @property
    def labels(self):
        """Maximum likelihood labels for training data. Can access with self.labels (no parens needed)"""
        return self.label(self.p_y_given_x)

    @property
    def clusters(self):
        """Return cluster labels for variables"""
        return np.argmax(self.alpha[:, :, 0], axis=0)

    @property
    def tc(self):
        """The total correlation explained by all the Y's.
        """
        return np.sum(self.tcs)

    def fit(self, X):
        """Fit CorEx on the data X. See fit_transform.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit CorEx on the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_visible]
            The data.

        Returns
        -------
        Y: array-like, shape = [n_samples, n_hidden]
           Learned values for each latent factor for each sample.
           Y's are sorted so that Y_1 explains most correlation, etc.
        """

        if self.n_cpu == 1:
            self.pool = None
        else:
            self.pool = Pool(self.n_cpu)
        Xm = ma.masked_equal(X, self.missing_values)
        best_tc = -np.inf
        for n_rep in range(self.n_repeat):

            self.initialize_parameters(X)
            #f=open(self.fileName+"testb4loop_fitTransform.txt",'a')
            #print("any nans?",file=f)
            #print(np.isnan(self.p_y_given_x.reshape(-1)).any(),file=f)
            #print("any infs?",file=f)
            #print(np.isinf(self.p_y_given_x.reshape(-1)).any(),file=f)
            #print("XM",file=f)
            #print(Xm,file=f)
            #f.close()
            #np.savetxt(self.fileName+"p_ygivnx_afterInit_Param.txt",self.p_y_given_x.reshape(-1))

            for nloop in range(self.max_iter):
                self.loopCount=nloop
                self.log_p_y = self.calculate_p_y(self.p_y_given_x)
                self.theta = self.calculate_theta(Xm, self.p_y_given_x)

                if self.n_hidden > 1:  # Structure learning step
                    self.update_alpha(self.p_y_given_x, self.theta, Xm, self.tcs)

                self.p_y_given_x, self.log_z = self.calculate_latent(self.theta, Xm)
                #fileName=self.fileName+'fit_trans_log_z.txt'
                #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_cor_test3.txt",'a')
                #f=open(fileName,'a')
                #print("in fit_transform --tcs\n",file=f)
                #print(self.tcs,file=f)
                #print("\n tc nloop= \n ",file=f)
                #print(nloop,file=f)
                #print("\n",file=f)
                #print(self.tc,file=f)
                #print("\n log_z \n",file=f)
                #print(self.log_z,file=f)
                #print("p_y_giv_x",file=f)
                #np.savetxt(f,self.p_y_given_x.reshape(-1))
                
                self.update_tc(self.log_z)  # Calculate TC and record history to check convergence
                #print("\n after update tc --tcs\n",file=f)
                #print(self.tcs,file=f)
                #print("\n",file=f)
                #print("any nans for theta -- test if nan is in theta first then in p_y_giv x?",file=f)
                #print(np.isnan(self.theta.reshape(-1)).any(),file=f)
                #print("any infs test if nan is in theta first then in p_y_giv x?",file=f)
                #print(np.isinf(self.theta.reshape(-1)).any(),file=f)
                
                #print("any nans for p_y_givn x test if inf is in theta first then in p_y_giv x?",file=f)
                #print(np.isnan(self.p_y_given_x.reshape(-1)).any(),file=f)
                #print("any infs test if nan is in theta first then in p_y_giv x?",file=f)
                #print(np.isinf(self.p_y_given_x.reshape(-1)).any(),file=f)
                #f.close()
                #g=open(self.fileName+"checktheta_p_y_gvx_fitTransform.txt",'a')
                #np.savetxt(g,["loop is \n"],fmt="%s")
                #np.savetxt(g,[nloop])
                #np.savetxt(g,["\n theta \n"],fmt="%s")
                #np.savetxt(g,self.theta.reshape(-1))
                #np.savetxt(g,["\n p_y_giv_x\n"],fmt="%s")
                #np.savetxt(g,self.p_y_given_x.reshape(-1))
                #g.close()
                self.print_verbose()
                if self.convergence():
                    break

            if self.verbose:
                print ('Overall tc:', self.tc)
            print('self_tcs',self.tcs)
            print('self_tc',self.tc)
            print('log_z',self.log_z)
            #Sheena changed self.tc>best_tc to self.tc>=best_tc to allow best dict to init
            #I assume this from the explanation from the attribute section
            if self.tc > best_tc:
                best_tc = self.tc
                best_dict = self.__dict__.copy()  # TODO: what happens if n_cpu > 1 and n_repeat > 1? Does pool get copied? Probably not...just a pointer to the same object... Seems fine.
        self.__dict__ = best_dict
        #SHEENA: not sure how this is supposed to be best_tc... it should probably be best_tc down here
        if self.verbose:
            print ('Best tc:', self.tc)

        self.sort_and_output(Xm)
        #added by SHeena below for debugging 
        lgmg=self.calculate_marginals_on_samples_sheenaDebug(self.theta,Xm)
        g=open(self.fileName+"lg_marg_p_yj_xi_svtxt.txt",'a')
        np.savetxt(g,lgmg.reshape(-1))
        g.close()
        g=open(self.fileName+"lg_marg_p_yj_xi.txt",'a')
        print(lgmg,file=g)
        g.close()
        
                #np.savetxt(g,[nloop])
                #np.savetxt(g,["\n theta \n"],fmt="%s")
                #np.savetxt(g,self.theta.reshape(-1))
                #np.savetxt(g,["\n p_y_giv_x\n"],fmt="%s")
                #np.savetxt(g,self.p_y_given_x.reshape(-1))
                #g.close()
        #modified for bin classification --just call_bin
        #log_marg_x = self.calculate_marginals_on_samples(self.theta, Xm, return_ratio=True)

        if self.pool is not None:
            self.pool.close()
            self.pool = None
        return self.labels

    def transform(self, X, details=False):
        """
        Label hidden factors for (possibly previously unseen) samples of data.
        Parameters: samples of data, X, shape = [n_samples, n_visible]
        Returns: , shape = [n_samples, n_hidden]
        """
        Xm = ma.masked_equal(X, self.missing_values)
        p_y_given_x, log_z = self.calculate_latent(self.theta, Xm)
        labels = self.label(p_y_given_x)
        log_marg_x = self.calculate_marginals_on_samples(self.theta, Xm, return_ratio=False)
        #fileName=self.fileName+'transform.txt'
        #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_cor_test3.txt",'a')
        #f=open(fileName,'a')
        #print('in transform log_marg_x.shape\n',file=f)
        #print("in transform log_marg_x.shape\n")
        #print(log_marg_x.shape,file=f)
        #print('\nlog_marg_x\n',file=f)
        #print(log_marg_x.shape)
        #print("log_marg_x\n")
        #print(log_marg_x)
        #print(log_marg_x,file=f)
        #print("\n self.log_p_y\n",file=f)
        #print(self.log_p_y.shape,file=f)
        #print("\n",file=f)
        #print(self.log_p_y,file=f)
        #print("\n",file=f)
        #print("\n self.alpha \n",file=f)
        #print(self.alpha.shape,file=f)
        #print("\n",file=f)
        #print(self.alpha,file=f)
        #print("\n")
        #print("log_p_y\n")
        #print(log_p_y)
        #f.close()
        lgmg=self.calculate_marginals_on_samples_sheenaDebug(self.theta,Xm)
        g=open(self.fileName+"lg_marg_p_yj_xi_transformed_svtxt.txt",'a')
        np.savetxt(g,lgmg.reshape(-1))
        g.close()
        g=open(self.fileName+"lg_marg_p_yj_xi_transformed.txt",'a')
        print(lgmg,file=g)
        g.close()
        if details == 'surprise':
            # Totally experimental
            log_marg_x = self.calculate_marginals_on_samples(self.theta, Xm, return_ratio=False)
            n_samples = Xm.shape[0]
            surprise = []
            for l in range(n_samples):
                q = - sum([max([log_marg_x[j,l,i,labels[l, j]]
                                for j in range(self.n_hidden)])
                           for i in range(self.n_visible)])
                surprise.append(q)
            return p_y_given_x, log_z, np.array(surprise)
        elif details:
            return p_y_given_x, log_z
        else:
            return p_y_given_x,labels

    def initialize_parameters(self, X):
        """Set up starting state

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_visible]
            The data.

        """
        self.n_samples, self.n_visible = X.shape[:2]
        if self.marginal_description == 'discrete':
            values_in_data = set(np.unique(X).tolist())-set([self.missing_values])
            self.dim_visible = int(max(values_in_data)) + 1
            #fileName=self.fileName+'output_initialize_parameters_discrete.txt'
            #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_cor_test3.txt",'a')
            #f=open(fileName,'a')
            #print('in initialize param -discrete dim_visible\n',file=f)
            #print(self.dim_visible,file=f)
            #f.close()
            if not set(range(self.dim_visible)) == values_in_data:
                print ("Warning: Data matrix values should be consecutive integers starting with 0,1,...")
            assert max(values_in_data) <= 32, "Due to a limitation in np.choice, discrete valued variables" \
                                              "can take values from 0 to 31 only."
        else:
            #??added by Sheena. Class label is discrete. Added below. THis code down here is iffy. hAVENT PROPOERLY STUDIED THE DISCRETE CODE!--Done
            values_in_data = set(np.unique(X[:,-1]).tolist())-set([self.missing_values])
            #fileName=self.fileName+'output_initialize_parameters_ClassLabel.txt'
            #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_cor_test3.txt",'a')
            #f=open(fileName,'a')
            #print('in initialize param -discrete dim_visible\n',file=f)
            #print(self.dim_visible,file=f)
            
            #print("values in data \n",file=f)
            #print(X,file=f)
            #print("\nset\n",file=f)
            #print(set([self.missing_values]),file=f)
            #f.close()
            self.dim_visible = int(max(values_in_data)) + 1
            if not set(range(self.dim_visible)) == values_in_data:
                print ("Warning: Data matrix values should be consecutive integers starting with 0,1,...")
            assert max(values_in_data) <= 32, "Due to a limitation in np.choice, discrete valued variables" \
                                                 "can take values from 0 to 31 only."
            #--addition ended above--Sh
            
        self.initialize_representation()

    def calculate_p_y(self, p_y_given_x):
        """Estimate log p(y_j) using a tiny bit of Laplace smoothing to avoid infinities."""
        
        #fileName=self.fileName+'output_calculate_p_y.txt'
        #f=open(fileName,'a')
        #print("loopCount",file=f)
        #print(self.loopCount,file=f)
        pseudo_counts = 0.001 + np.sum(p_y_given_x, axis=1, keepdims=True)
        #print("\n pseudo counts \n",file=f)
        #np.savetxt(f,pseudo_counts.reshape(-1))
        #print(pseudo_counts,file=f)
        #print("\n log_p_y\n",file=f)
        log_p_y = np.log(pseudo_counts) - np.log(np.sum(pseudo_counts, axis=2, keepdims=True))
        #print(log_p_y,file=f)
        #np.savetxt(f,log_p_y)
        return log_p_y

    def calculate_theta(self, Xm, p_y_given_x):
        """Estimate marginal parameters from data and expected latent labels."""
        theta = []
        #g=open(self.fileName+"calculate_theta_pygivx_full.txt",'a')
        #np.savetxt(g,["loop num"],fmt="%s")
        #np.savetxt(g,[self.loopCount])
        #np.savetxt(g,["\n p_y_givx \n"],fmt="%s")
        #np.savetxt(g,p_y_given_x.reshape(-1))
        #np.savetxt(g,["next\n"],fmt='%s')
        #g.close()
        
        #g=open(self.fileName+"calculate_theta_in_loop.txt",'a')
        #print("loopCount",file=g)
        #print(self.loopCount,file=g)
        #gets one inout feature at a time.
        for i in range(self.n_visible):
            #print("\n n_features =",file=g)
            #print(i,file=g)
            #print("\n not_missing",file=g)
            not_missing = np.logical_not(ma.getmaskarray(Xm)[:, i])
            #print(not_missing,file=g)
            theta.append(self.estimate_parameters(Xm.data[not_missing, i], p_y_given_x[:, not_missing],i))
            #print("\n theta",file=g)
            #print(theta.shape,file=g)
            #np.savetxt(g,theta.reshape(-1))
        #g.close()
        return np.array(theta)

    def update_alpha(self, p_y_given_x, theta, Xm, tcs):
        """A rule for non-tree CorEx structure.
        """
        #f=open(self.fileName+"update_alpha.txt",'a')
        sample = np.random.choice(np.arange(Xm.shape[0]), min(self.max_samples, Xm.shape[0]), replace=False)
        #print("sample",file=f)
        #print(sample,file=f)
        p_y_given_x = p_y_given_x[:, sample, :]
        #print("p_y-giv_x",file=f)
        #print(p_y_given_x,file=f)
        not_missing = np.logical_not(ma.getmaskarray(Xm[sample]))
        #print("not_missing",file=f)
        #print(not_missing,file=f)
        alpha = np.empty((self.n_hidden, self.n_visible))
        n_samples, n_visible = Xm.shape
        memory_size = float(self.max_samples * n_visible * self.n_hidden * self.dim_hidden * 64) / 1000**3  # GB
        batch_size = np.clip(int(self.ram * n_visible / memory_size), 1, n_visible)
        for i in range(0, n_visible, batch_size):
            log_marg_x = self.calculate_marginals_on_samples(theta[i:i+batch_size], Xm[sample, i:i+batch_size])
            correct_predictions = np.argmax(p_y_given_x, axis=2)[:, :, np.newaxis] == np.argmax(log_marg_x, axis=3)
            for ip in range(i, min(i + batch_size, n_visible)):
                alpha[:, ip] = self.unique_info(correct_predictions[:, not_missing[:, ip], ip - i].T)

        for j in np.where(np.abs(tcs) < self.tc_min)[0]:  # Priming for un-used hidden units
            amax = np.clip(np.max(alpha[j, :]), 0.01, 0.99)
            alpha[j, :] = alpha[j, :]**(np.log(0.99)/np.log(amax)) + 0.001 * np.random.random(self.n_visible)
        self.alpha = alpha[:, :, np.newaxis]  # TODO: This is the "correct" update but it is quite noisy. Add smoothing?

    def unique_info(self, correct):
        """*correct* has n_samples rows and n_hidden columns.
            It indicates whether the ml estimate based on x_i for y_j is correct for sample l
            Returns estimate of fraction of unique info in each predictor j=1...m
        """
        n_samples, n_hidden = correct.shape
        total = np.clip(np.sum(correct, axis=0), 1, n_samples)
        ordered = np.argsort(total)[::-1]

        unexplained = np.ones(n_samples, dtype=bool)
        unique = np.zeros(n_hidden, dtype=int)
        for j in ordered:
            unique[j] = np.dot(unexplained.astype(int), correct[:, j])  # np.sum(correct[unexplained, j])
            unexplained = np.logical_and(unexplained, np.logical_not(correct[:, j]))

        frac_unique = [float(unique[j]) / total[j] for j in range(n_hidden)]
        return np.array(frac_unique)

    def calculate_latent(self, theta, Xm):
        """"Calculate the probability distribution for hidden factors for each sample."""
        n_samples, n_visible = Xm.shape
        log_p_y_given_x_unnorm = np.empty((self.n_hidden, n_samples, self.dim_hidden))
        memory_size = float(n_samples * n_visible * self.n_hidden * self.dim_hidden * 64) / 1000**3  # GB
        batch_size = np.clip(int(self.ram * n_samples / memory_size), 1, n_samples)
        #fileName=self.fileName+'calculate_latent.txt'
        #f=open(fileName,'a')
        #np.savetxt(f,["loopCount\n"],fmt="%s")
        #np.savetxt(f,[self.loopCount])
        #print('in calculate latent -- log_p_y\n',file=f)
        #print(self.log_p_y,file=f)
        #print("nans in log_p_y",file=f)
        #print(np.isnan(self.log_p_y.reshape(-1)).any(),file=f)
        #print("\n",file=f)
        for l in range(0, n_samples, batch_size):
            log_marg_x = self.calculate_marginals_on_samples(theta, Xm[l:l+batch_size])  # LLRs for each sample, for each var.
            #print("\n log_marg_x\n",file=f)
            #print(log_marg_x,file=f)
            #print("\n nans in log_marg_x\n",file=f)
            #print(np.isnan(log_marg_x.reshape(-1)).any(),file=f)
            #print("\n inf in log_marg_x\n",file=f)
            #print(np.isinf(log_marg_x.reshape(-1)).any(),file=f)
            #print("\n nans in alpha \n",file=f)
            #print(np.isnan(self.alpha.reshape(-1)).any(),file=f)
            #print("\n inf in alpha\n",file=f)
            #print(np.isinf(self.alpha.reshape(-1)).any(),file=f)
            #print("\n alpha\n",file=f)
            #print(self.alpha,file=f)
            #print("\n einsum has nans?\n ",file=f)
            #print(np.isnan(np.einsum('ikl,ijkl->ijl', self.alpha, log_marg_x).reshape(-1)).any(),file=f)
            #print("\n inf in einsum\n",file=f)
            #print(np.isinf(np.einsum('ikl,ijkl->ijl', self.alpha, log_marg_x).reshape(-1)).any(),file=f)
            #print("\n",file=f)
            #g=open(self.fileName+"log_marg_x_svtxt.txt",'ab')
            #np.savetxt(g,log_marg_x.reshape(-1))
            #np.savetxt(g,["next\n"],fmt='%s')
            #g.close()
            #g=open(self.fileName+"thetha.txt_svtxt.txt",'ab')
            #np.savetxt(g,theta.reshape(-1))
            #np.savetxt(g,["next\n"],fmt='%s')
            #g.close()
            
            log_p_y_given_x_unnorm[:, l:l+batch_size, :] = self.log_p_y + np.einsum('ikl,ijkl->ijl', self.alpha, log_marg_x)
        #f.close()
        return self.normalize_latent(log_p_y_given_x_unnorm)

    def normalize_latent(self, log_p_y_given_x_unnorm):
        """Normalize the latent variable distribution

        For each sample in the training set, we estimate a probability distribution
        over y_j, each hidden factor. Here we normalize it. (Eq. 7 in paper.)
        This normalization factor is quite useful as described in upcoming work.

        Parameters
        ----------
        Unnormalized distribution of hidden factors for each training sample.

        Returns
        -------
        p_y_given_x : 3D array, shape (n_hidden, n_samples, dim_hidden)
            p(y_j|x^l), the probability distribution over all hidden factors,
            for data samples l = 1...n_samples
        log_z : 2D array, shape (n_hidden, n_samples)
            Point-wise estimate of total correlation explained by each Y_j for each sample,
            used to estimate overall total correlation.

        """
        #fileName=self.fileName+'normalize_latent.txt'
        #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_cor_test3.txt",'a')
        #f=open(fileName,'a')
        #np.savetxt(f,["loopCount\n"],fmt="%s")
        #np.savetxt(f,[self.loopCount])
        #print('in normalize latent -- log_p_y_given_x_unorm\n',file=f)
        #print(log_p_y_given_x_unnorm,file=f)
        #print("\n has log_p_y_giv_x_unnorm nans \n",file=f)
        #print(np.isnan(log_p_y_given_x_unnorm.reshape(-1)).any(),file=f)
        log_z = logsumexp(log_p_y_given_x_unnorm, axis=2)  # Essential to maintain precision.
        #print("\n log_z\n",file=f)
        #print(log_z,file=f)
        #print("\n logsumexp(log_p_y_given_x_unnorm, axis=2) is nan \n ",file=f)
        #print(np.isnan(log_z.reshape(-1)).any(),file=f)
        log_z = log_z.reshape((self.n_hidden, -1, 1))
        #print("\n log_z.reshape((self.n_hidden, -1, 1)) \n",file=f)
        #print(log_z,file=f)
        #print("\n log_z.reshape((self.n_hidden, -1, 1)) is nan \n ",file=f)
        #print(np.isnan(log_z.reshape(-1)).any(),file=f)
        #print('\n',file=f)
        #print("any nans for np.exp(log_p_y_gov_x-unnorm - log_z?",file=f)
        #print(np.isnan(np.exp(log_p_y_given_x_unnorm - log_z).reshape(-1)).any(),file=f)
        #print("any infs for np.exp(log_p_y_gov_x-unnorm - log_z?",file=f)
        #print(np.isinf(np.exp(log_p_y_given_x_unnorm - log_z).reshape(-1)).any(),file=f)
        #f.close()
        return np.exp(log_p_y_given_x_unnorm - log_z), log_z

    def calculate_p_xi_given_y(self, xi, thetai,i):
        not_missing = np.logical_not(ma.getmaskarray(xi))
        #fileName=self.fileName+'output_calc_p-xi_notMIssing20.txt'
        #f=open(fileName,'a')
        #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_calc_p-xi_notMIssing20.txt",'a')
        #print("\ncount=\n",file=f)
        #print(self.countS,file=f)
        #np.savetxt(f,["loopCount\n"],fmt="%s")
        #np.savetxt(f,[self.loopCount])
        #print("\n not missing\n",file=f)
        #print(not_missing,file=f)
        #f.close()
        z = np.zeros((self.n_hidden, len(xi), self.dim_hidden))
        z[:, not_missing, :] = self.marginal_p(xi[not_missing], thetai,i)
        #fileName=self.fileName+'output_calc_p-xi20.txt'
        #f=open(fileName,'a')
        #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_calc_p-xi20.txt",'a')
        #print("count=\n",file=f)
        #print(self.countS,file=f)
        #np.savetxt(f,["loopCount\n"],fmt="%s")
        #np.savetxt(f,[self.loopCount])
        #np.savetxt(f,["\n loopCount\n"],fmt="%s")
        #np.savetxt(f,[self.loopCount])
        #print("\n z \n",file=f)
        #print(z,file=f)
        #f.close()
        return z  # n_hidden, n_samples, dim_hidden

    def calculate_marginals_on_samples(self, theta, Xm, return_ratio=True):
        """Calculate the value of the marginal distribution for each variable, for each hidden variable and each sample.

        theta: array parametrizing the marginals
        Xm: the data
        returns log p(y_j|x_i)/p(y_j) for each j,sample,i,y_j. [n_hidden, n_samples, n_visible, dim_hidden]
        """
        n_samples, n_visible = Xm.shape
        log_marg_x = np.zeros((self.n_hidden, n_samples, n_visible, self.dim_hidden))  #, dtype=np.float32)
        if n_visible > 1 and self.pool is not None:
            args = zip([self] * len(theta), Xm.T, theta)
            log_marg_x = np.array(self.pool.map(unwrap_f, args)).transpose((1, 2, 0, 3))
        else:
            for i in range(n_visible):
                log_marg_x[:, :, i, :] = self.calculate_p_xi_given_y(Xm[:, i], theta[i],i)
        #fileName=self.fileName+'output_calcMArg20.txt'
        #f=open(fileName,'a')
        #np.savetxt(f,["loopCount\n"],fmt="%s")
        #np.savetxt(f,[self.loopCount])
        #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_calcMArg20.txt",'a')
        #print("\n calc_p_xi_giv_y\n",file=f)
        #print(log_marg_x,file=f)
        #print("\nlogmarg-shape\n",file=f)
        #print(log_marg_x.shape,file=f)
        #print("\nXm\n",file=f)
        #print(Xm,file=f)
        #print("theta \n",file=f)
        #print(theta,file=f)
        #f.close()
        if return_ratio:  # Return log p(xi|y)/p(xi) instead of log p(xi|y)
            # Again, I use the same p(y) here for each x_i, but for missing variables, p(y) on obs. sample may be different.
            # log_marg_x -= logsumexp(log_marg_x + self.log_p_y.reshape((self.n_hidden, 1, 1, self.dim_hidden)), axis=3)[..., np.newaxis]
            log_marg_x += self.log_p_y.reshape((self.n_hidden, 1, 1, self.dim_hidden))
            if self.pool is not None:
                #fileName=self.fileName+'output_cor_test20.txt'
                #f=open(fileName,'a')
                #np.savetxt(f,["loopCount\n"],fmt="%s")
                #np.savetxt(f,[self.loopCount])
                #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_cor_test20.txt",'a')
                #print("\np(x_i) in if\n",file=f)
                #print(np.array(self.pool.map(logsumexp2, log_marg_x))[..., np.newaxis].shape,file=f)
                #print("\n",file=f)
                #print(np.array(self.pool.map(logsumexp2, log_marg_x))[..., np.newaxis],file=f)
                #print("\n",file=f)
                log_marg_x -= np.array(self.pool.map(logsumexp2, log_marg_x))[..., np.newaxis]
                #print(log_marg_x)
                #f.close()
            else:
                #fileName=self.fileName+'output_log_marg20.txt'
                #f=open(fileName,'a')
                #np.savetxt(f,["loopCount\n"],fmt="%s")
                #np.savetxt(f,[self.loopCount])
                #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_log_marg20.txt",'a')
                #print("\n p(x_i) in else\n",file=f)
                #print(logsumexp(log_marg_x, axis=3)[..., np.newaxis].shape,file=f)
                #print("\n",file=f)
                #print(logsumexp(log_marg_x, axis=3)[..., np.newaxis],file=f)
                log_marg_x -= logsumexp(log_marg_x, axis=3)[..., np.newaxis]

                #f.close()
            #fileName=self.fileName+'output_log_marg20.txt'
            #f=open(fileName,'a')
            #np.savetxt(f,["loopCount\n"],fmt="%s")
            #np.savetxt(f,[self.loopCount])
            #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_log_marg20.txt",'a')
            #print("\nlog_marg after -log_py\n",file=f)
            log_marg_x -= self.log_p_y.reshape((self.n_hidden, 1, 1, self.dim_hidden))
            #print(log_marg_x,file=f)
            #f.close()
        return log_marg_x
    
    
    
    def calculate_marginals_on_samples_sheenaDebug(self, theta, Xm, return_ratio=True):
        """Calculate the value of the marginal distribution for each variable, for each hidden variable and each sample.

        theta: array parametrizing the marginals
        Xm: the data
        returns log p(y_j|x_i) for each j,sample,i,y_j. [n_hidden, n_samples, n_visible, dim_hidden]
        """
        n_samples, n_visible = Xm.shape
        log_marg_x = np.zeros((self.n_hidden, n_samples, n_visible, self.dim_hidden))  #, dtype=np.float32)
        if n_visible > 1 and self.pool is not None:
            args = zip([self] * len(theta), Xm.T, theta)
            log_marg_x = np.array(self.pool.map(unwrap_f, args)).transpose((1, 2, 0, 3))
        else:
            for i in range(n_visible):
                log_marg_x[:, :, i, :] = self.calculate_p_xi_given_y(Xm[:, i], theta[i],i)
        #fileName=self.fileName+'output_calcMArg20.txt'
        #f=open(fileName,'a')
        #np.savetxt(f,["loopCount\n"],fmt="%s")
        #np.savetxt(f,[self.loopCount])
        #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_calcMArg20.txt",'a')
        #print("\n calc_p_xi_giv_y\n",file=f)
        #print(log_marg_x,file=f)
        #print("\nlogmarg-shape\n",file=f)
        #print(log_marg_x.shape,file=f)
        #print("\nXm\n",file=f)
        #print(Xm,file=f)
        #print("theta \n",file=f)
        #print(theta,file=f)
        #f.close()
        if return_ratio:  # Return log p(xi|y)/p(xi) instead of log p(xi|y)
            # Again, I use the same p(y) here for each x_i, but for missing variables, p(y) on obs. sample may be different.
            # log_marg_x -= logsumexp(log_marg_x + self.log_p_y.reshape((self.n_hidden, 1, 1, self.dim_hidden)), axis=3)[..., np.newaxis]
            log_marg_x += self.log_p_y.reshape((self.n_hidden, 1, 1, self.dim_hidden))
            if self.pool is not None:
                #fileName=self.fileName+'output_cor_test20.txt'
                #f=open(fileName,'a')
                #np.savetxt(f,["loopCount\n"],fmt="%s")
                #np.savetxt(f,[self.loopCount])
                #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_cor_test20.txt",'a')
                #print("\np(x_i) in if\n",file=f)
                #print(np.array(self.pool.map(logsumexp2, log_marg_x))[..., np.newaxis].shape,file=f)
                #print("\n",file=f)
                #print(np.array(self.pool.map(logsumexp2, log_marg_x))[..., np.newaxis],file=f)
                #print("\n",file=f)
                log_marg_x -= np.array(self.pool.map(logsumexp2, log_marg_x))[..., np.newaxis]
                #print(log_marg_x)
                #f.close()
            else:
                #fileName=self.fileName+'output_log_marg20.txt'
                #f=open(fileName,'a')
                #np.savetxt(f,["loopCount\n"],fmt="%s")
                #np.savetxt(f,[self.loopCount])
                #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_log_marg20.txt",'a')
                #print("\n p(x_i) in else\n",file=f)
                #print(logsumexp(log_marg_x, axis=3)[..., np.newaxis].shape,file=f)
                #print("\n",file=f)
                #print(logsumexp(log_marg_x, axis=3)[..., np.newaxis],file=f)
                log_marg_x -= logsumexp(log_marg_x, axis=3)[..., np.newaxis]

                #f.close()
            #fileName=self.fileName+'output_log_marg20.txt'
            #f=open(fileName,'a')
            #np.savetxt(f,["loopCount\n"],fmt="%s")
            #np.savetxt(f,[self.loopCount])
            #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_log_marg20.txt",'a')
            #print("\nlog_marg after -log_py\n",file=f)
            #log_marg_x -= self.log_p_y.reshape((self.n_hidden, 1, 1, self.dim_hidden))
            #print(log_marg_x,file=f)
            #f.close()
        return log_marg_x

    def initialize_representation(self):
        if self.n_hidden > 1:
            self.alpha = (0.5+0.5*np.random.random((self.n_hidden, self.n_visible, 1)))
        else:
            self.alpha = np.ones((self.n_hidden, self.n_visible, 1), dtype=float)
        self.tc_history = []
        self.tcs = np.zeros(self.n_hidden)

        p_rand = np.random.dirichlet(np.ones(self.dim_hidden), (self.n_hidden, self.n_samples))
        #np.savetxt(self.fileName+"p-randNOLOG.txt",p_rand.reshape(-1))
        #g=open(self.fileName+"log_p_rand.txt",'a')
        #np.savetxt(g,np.log(p_rand).reshape(-1))
        #g.close()
        
        #f=open(self.fileName+"p_rand2.txt",'a')
        #print("any nans log(p_rand)?",file=f)
        #print(np.isnan(np.log(p_rand).reshape(-1)).any(),file=f)
        #print("any inf log(p_rand)?",file=f)
        #print(np.isinf(np.log(p_rand).reshape(-1)).any(),file=f)
        #f.close()
        self.p_y_given_x, self.log_z = self.normalize_latent(np.log(p_rand))

    def update_tc(self, log_z):
        #fileName=self.fileName+'log_z_update_tc.txt'
        #f=open(fileName,'a')
        #print("log_z.reshape(-1) \n",file=f)
        #print(self.log_z.reshape(-1),file=f)
        #print("any nans\n",file=f)
        #print(np.isnan(self.log_z.reshape(-1)).any(),file=f)
        #print("\n",file=f)
        #f.close()
        
        #fileName=self.fileName+'update_tc.txt'
        #f=open(fileName,'a')
        #print("update_tc \n",file=f)
        #print("\n np.mean(log_z, axis=1)\n",file=f)
        #print(np.mean(log_z, axis=1),file=f)
        self.tcs = np.mean(log_z, axis=1).reshape(-1)
        #print("\n self.tcs \n after reshape \n",file=f)
        #print(self.tcs,file=f)
        #print("\n")
        #f.close()
        self.tc_history.append(np.sum(self.tcs))

    def print_verbose(self):
        if self.verbose:
            print(self.tcs)
        if self.verbose > 1:
            print(self.alpha[:, :, 0])
            print(self.theta)
            if hasattr(self, "mis"):
                print(self.mis)

    def convergence(self):
        if len(self.tc_history) > 10:
            dist = -np.mean(self.tc_history[-10:-5]) + np.mean(self.tc_history[-5:])
            return np.abs(dist) < self.eps  # Check for convergence.
        else:
            return False

    def __getstate__(self):
        # In principle, if there were variables that are themselves classes... we have to handle it to pickle correctly
        # But I think I programmed around all that.
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def save(self, filename):
        """ Pickle a class instance. E.g., corex.save('saved.dat') """
        import pickle
        if path.dirname(filename) and not path.exists(path.dirname(filename)):
            makedirs(path.dirname(filename))
        pickle.dump(self, open(filename, 'wb'), protocol=-1)

    def load(self, filename):
        """ Unpickle class instance. E.g., corex = ce.Marginal_Corex().load('saved.dat') """
        import pickle
        return pickle.load(open(filename, 'rb'))

    def sort_and_output(self, Xm):
        order = np.argsort(self.tcs)[::-1]  # Order components from strongest TC to weakest
        self.tcs = self.tcs[order]  # TC for each component
        self.alpha = self.alpha[order]  # Connections between X_i and Y_j
        self.p_y_given_x = self.p_y_given_x[order]  # Probabilistic labels for each sample
        self.theta = self.theta[:, :, order, :]  # Parameters defining the representation
        self.log_p_y = self.log_p_y[order]  # Parameters defining the representation
        self.log_z = self.log_z[order]  # -log_z can be interpreted as "surprise" for each sample
        if not hasattr(self, 'mis'):
            # self.update_marginals(Xm, self.p_y_given_x)
            self.mis = self.calculate_mis(self.p_y_given_x, self.theta, Xm)
        else:
            self.mis = self.mis[order]
        bias, sig = self.mi_bootstrap(Xm, n_permutation=20)
        self.mis = (self.mis - bias) * (self.mis > sig)

    def calculate_mis(self, p_y_given_x, theta, Xm):
        mis = np.zeros((self.n_hidden, self.n_visible))
        sample = np.random.choice(np.arange(Xm.shape[0]), min(self.max_samples, Xm.shape[0]), replace=False)
        n_observed = np.sum(np.logical_not(ma.getmaskarray(Xm[sample])), axis=0)

        n_samples, n_visible = Xm.shape
        memory_size = float(n_samples * n_visible * self.n_hidden * self.dim_hidden * 64) / 1000**3  # GB
        batch_size = np.clip(int(self.ram * n_visible / memory_size), 1, n_visible)
        for i in range(0, n_visible, batch_size):
            log_marg_x = self.calculate_marginals_on_samples(theta[i:i+batch_size, ...], Xm[sample, i:i+batch_size])  # n_hidden, n_samples, n_visible, dim_hidden
            mis[:, i:i+batch_size] = np.einsum('ijl,ijkl->ik', p_y_given_x[:, sample, :], log_marg_x) / n_observed[i:i+batch_size][np.newaxis, :]
        return mis  # MI in nats

    def mi_bootstrap(self, Xm, n_permutation=20):
        # est. if p-val < 1/n_permutation = 0.05
        mis = np.zeros((self.n_hidden, self.n_visible, n_permutation))
        for j in range(n_permutation):
            p_y_given_x = self.p_y_given_x[:, np.random.permutation(self.n_samples), :]
            theta = self.calculate_theta(Xm, p_y_given_x)
            mis[:, :, j] = self.calculate_mis(p_y_given_x, theta, Xm)
        return np.mean(mis, axis=2), np.sort(mis, axis=2)[:, :, -2]

    # IMPLEMENTED MARGINAL DISTRIBUTIONS
    # For each distribution, we need:
    # marginal_p_METHOD(self, xi, thetai), to define the marginal probability of x_i given theta_i (for each y_j=k)
    # estimate_parameters_METHOD(self, x, p_y_given_x), a way to estimate theta_i from samples (for each y_j=k)
    # marginal_p should be vectorized. I.e., xi can be a single xi or a list.

    def marginal_p(self, xi, thetai,i=0):
        """Estimate marginals, log p(xi|yj) for each possible type. """
        self.countS+=1
        #if added by Sheena. To change to original, remove top level if and else
        if(i!=(self.n_visible-1)):
            if self.marginal_description == 'gaussian':
                mu, sig = thetai  # mu, sig have size m by k
                #fileName=self.fileName+'output_marginalP20.txt'
                #f=open(fileName,'a')
                #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_marginalP20.txt",'a')
                #print("count=\n",file=f)
                #print(self.countS,file=f)
                #np.savetxt(f,["loopCount\n"],fmt="%s")
                #np.savetxt(f,[self.loopCount])
                #print("\n mu \n",file=f)
                #print(mu,file=f)
                #print("\n sig \n",file=f)
                #print(sig,file=f)
                #print("\n 2* sig\n",file=f)
                #print(2.*sig,file=f)
                xi = xi.reshape((-1, 1, 1))
                #f.close()
                #fileName=self.fileName+'output_xiP20.txt'
                #f=open(fileName,'a')
                #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_xiP20.txt",'a')
                #print("count=\n",file=f)
                #print(self.countS,file=f)
                #np.savetxt(f,["loopCount\n"],fmt="%s")
                #np.savetxt(f,[self.loopCount])
                #print("xi.reshape",file=f)
                #print(xi,file=f)
                #f.close()
                #print("\n (xi -mu) ^2 / (2*sig) \n",file=f)
                #print(-(xi - mu)**2 / (2. * sig),file=f)
                #print("\n0.5 * np.log(2 * np.pi * sig) \n",file=f)
                #print(0.5 * np.log(2 * np.pi * sig),file=f)
                #fileName=self.fileName+'output_zP20.txt'
                #f=open(fileName,'a')
                #f = open("/home/bb79e/Code/Bin_classify/bb79e3/output_zP20.txt",'a')
                #print("count=\n",file=f)
                #print(self.countS,file=f)
                #np.savetxt(f,["loopCount\n"],fmt="%s")
                #np.savetxt(f,[self.loopCount])
                #print("\n -(xi - mu)**2 / (2. * sig) - 0.5 * np.log(2 * np.pi * sig)\n",file=f)
                #print(-(xi - mu)**2 / (2. * sig) - 0.5 * np.log(2 * np.pi * sig),file=f)
                #print("is inf?",file=f)
                #print(np.isinf((-(xi - mu)**2 / (2. * sig) - 0.5 * np.log(2 * np.pi * sig)).reshape(-1).any()),file=f)
                #print("\n",file=f)
                #print("is nan?",file=f)
                #var for debug only. added by SHeena
                #dummyVar =np.isnan((-(xi - mu)**2 / (2. * sig) - 0.5 * np.log(2 * np.pi * sig)).reshape(-1).any())
                #print(dummyVar,file=f)
                #if(dummyVar==True):
                    #g=open(self.fileName+"marginal_p_dummyVAr_test.txt",'ab')
                    #np.savetxt(g,mu)
                    #var="\n mu above \n sig below \n"
                    #np.savetxt(g,[var],fmt="%s")
                    #np.savetxt(g,sig)
                   
                    #np.savetxt(g,["\n \n "],fmt="%s")
                    #g.close()
                #print("\n",file=f)
                #f.close()
                
                #fileName=self.fileName+'output_full_z20.txt'
                #f=open(fileName,'a')
                #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_full_z20.txt",'a')
                #print("count=\n",file=f)
                #print(self.countS,file=f)
                #np.savetxt(f,["loopCount\n"],fmt="%s")
                #np.savetxt(f,[self.loopCount])
                #print("\n mu \n",file=f)
                #print(mu,file=f)
                #print("\n sig \n",file=f)
                #print(sig,file=f)
                #print("\n -(xi - mu)**2 / (2. * sig) - 0.5 * np.log(2 * np.pi * sig)\n",file=f)
                #print(-(xi - mu)**2 / (2. * sig) - 0.5 * np.log(2 * np.pi * sig),file=f)
                #f.close()
                return (-(xi - mu)**2 / (2. * sig) - 0.5 * np.log(2 * np.pi * sig)).transpose((1, 0, 2))  # log p(xi|yj)
    
            elif self.marginal_description == 'discrete':
                # Discrete data: should be non-negative integers starting at 0: 0,...k. k < 32 because of np.choose limits
                logp = [theta[np.newaxis, ...] for theta in thetai]  # Size dim_visible by n_hidden by dim_hidden
                #fileName=self.fileName+'output_full_z20.txt'
                #f=open(fileName,'a')
                #np.savetxt(f,["loopCount\n"],fmt="%s")
                #np.savetxt(f,[self.loopCount])
                #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_full_z20.txt",'a')
                #print("count=\n",file=f)
                #print(self.countS,file=f)
                #print("\n discrete logp marginalp \n",file=f)
                #print(logp,file=f)
                #f.close()
                
                return np.choose(xi.reshape((-1, 1, 1)), logp).transpose((1, 0, 2))
    
            else:
                print('Marginal description "%s" not implemented.' % self.marginal_description)
                sys.exit()
        else: #top level else added by Sheena, only comes in here for class var
             #need to downcast the array xi cos if the ori was continuous np.choose won't work.
             xi=xi.astype('int')
             # Discrete data: should be non-negative integers starting at 0: 0,...k. k < 32 because of np.choose limits
             logp = [theta[np.newaxis, ...] for theta in thetai]  # Size dim_visible by n_hidden by dim_hidden
             
             #fileName=self.fileName+'output_full_z20.txt'
             #f=open(fileName,'a')
             #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_full_z20.txt",'a')
             #np.savetxt(f,["loopCount\n"],fmt="%s")
             #np.savetxt(f,[self.loopCount])
             #print("count=\n",file=f)
             #print(self.countS,file=f)
             #print("\n discrete logp marginalp \n",file=f)
             #print(logp,file=f)
             #f.close()
                
             return np.choose(xi.reshape((-1, 1, 1)), logp).transpose((1, 0, 2))
            

    def estimate_parameters(self, xi, p_y_given_x,i=0):
        #i added by Sheena
        #assume last col is class label
        #if added by Sheena. Remove topmost if to get ori code
        #i is base 0
        if(i!=(self.n_visible-1)):
            if self.marginal_description == 'gaussian':
                n_obs = np.sum(p_y_given_x, axis=1).clip(0.1)  # m, k
                mean_ml = np.einsum('i,jik->jk', xi, p_y_given_x) / n_obs  # ML estimate of mean of Xi
                #ori sig_ml commented out by SHeena: sig_ml = np.einsum('jik,jik->jk', (xi[np.newaxis, :, np.newaxis] - mean_ml[:, np.newaxis, :])**2, p_y_given_x) / (n_obs - 1).clip(0.01)  # UB estimate of sigma^2(variance)
                #inner clip (n_obs-1).clip() is needed cos n_obs-1 can become 0 if n_obs==1, the outerclip is required in case sig becomes zero.
                sig_ml = (np.einsum('jik,jik->jk', (xi[np.newaxis, :, np.newaxis] - mean_ml[:, np.newaxis, :])**2, p_y_given_x) / (n_obs - 1).clip(0.01)).clip(0.000000000001)  # UB estimate of sigma^2(variance)
                #p_y_given_x -> p_y_given_x.clip() to ensure zero doesn't happen for sigma
                #sig_ml = np.einsum('jik,jik->jk', (xi[np.newaxis, :, np.newaxis] - mean_ml[:, np.newaxis, :])**2, p_y_given_x.clip(min=0.00000001,max=0.99999999)) / (n_obs - 1).clip(0.01)  # UB estimate of sigma^2(variance)
                #g=open(self.fileName+"estpARAMS.txt",'a')
                #np.savetxt(g,["count"],fmt="%s")
                #np.savetxt(g,[self.countS2])
                #np.savetxt(g,["loop"],fmt="%s")
                #np.savetxt(g,[self.loopCount])
                #np.savetxt(g,["xi\n"],fmt="%s")
                #np.savetxt(g,xi.reshape(-1))
                #np.savetxt(g,["\n mu\n"],fmt="%s")
                #np.savetxt(g,mean_ml.reshape(-1))
                #print("print v of mean",file=g)
                #print(mean_ml,file=g)
                #np.savetxt(g,["\n sig\n"],fmt="%s")
                #np.savetxt(g,sig_ml.reshape(-1))
                #print("print_v of sig",file=g)
                #print(sig_ml,file=g)
                #np.savetxt(g,["n-obs"],fmt="%s")
                #np.savetxt(g,n_obs.reshape(-1))
                #g.close()
                
                #added bysheena : for debugging purposes only: countS2
                #self.countS2=self.countS2+1
                
                #fileName=self.fileName+'output_estimatePAram_z20.txt'
                #f=open(fileName,'a')
                #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_estimatePAram_z20.txt",'a')
                #print("\n count = \n",file=f)
                #print(self.countS2,file=f)
                #print("loopCount\n",file=f)
                #print(self.loopCount,file=f)
                #print("\nestimate param - xi\n",file=f)
                #print(xi,file=f)
                #print("\n n_obs \n",file=f)
                #print(n_obs,file=f)
                #print("\np_y_giv_x\n",file=f)
                #print(p_y_given_x,file=f)
                #print("p_y_g_x savetxt version \n",file=f)
                #np.savetxt(f,p_y_given_x.reshape(-1))
                #f.close()
                
               # fileName=self.fileName+'output_sig_estimate20.txt'
                #f=open(fileName,'a')
                #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_sig_estimate20.txt",'a')
                #print("\n count = \n",file=f)
                #print(self.countS2,file=f)
                #print("loopCount\n",file=f)
                #print(self.loopCount,file=f)
                #print("\n mean \n,",file=f)
                #print(mean_ml,file=f)
                #print("\n np.einsum('jik,jik->jk', (xi[np.newaxis, :, np.newaxis] - mean_ml[:, np.newaxis, :])**2, p_y_given_x)  \n",file=f)
                #print(np.einsum('jik,jik->jk', (xi[np.newaxis, :, np.newaxis] - mean_ml[:, np.newaxis, :])**2, p_y_given_x) ,file=f)
                #f.close()
                
                #fileName=self.fileName+'output_sig2_estimate20.txt'
                #f=open(fileName,'a')
                #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_sig2_estimate20.txt",'a')
                #print("\n count = \n",file=f)
                #print(self.countS2,file=f)
                #print("loopCount\n",file=f)
                #print(self.loopCount,file=f)
                #print("n_obs -1\n",file=f)
                #print(n_obs-1,file=f)
                #print("\n np.einsum('jik,jik->jk', (xi[np.newaxis, :, np.newaxis] - mean_ml[:, np.newaxis, :])**2, p_y_given_x)/(n_obs - 1).clip(0.01)  \n",file=f)
                #print((np.einsum('jik,jik->jk', (xi[np.newaxis, :, np.newaxis] - mean_ml[:, np.newaxis, :])**2, p_y_given_x)/(n_obs - 1)).clip(0.01) ,file=f)
                #f.close()
                if not self.smooth_marginals:
                    return np.array([mean_ml, sig_ml])  # FOR EACH Y_j = k !!
                else:  # mu = lam mu_ml + 1-lam mu0 for lam minimizing KL divergence risk
                    mean0 = np.mean(xi)
                    sig0 = np.sum((xi - mean0)**2) / (len(xi) - 1)
                    m1, m2, se1, se2 = self.estimate_se(xi, p_y_given_x, n_obs)
                    d1 = mean_ml - m1
                    d2 = sig_ml - m2
                    lam = d1**2 / (d1**2 + se1**2)
                    gam = d2**2 / (d2**2 + se2**2)
                    lam, gam = np.where(np.isfinite(lam), lam, 0.5), np.where(np.isfinite(gam), gam, 0.5)
                    # lam2 = 1. - 1. / (1. + n_obs)  # Constant pseudo-count, doesn't work as well.
                    # gam2 = 1. - 1. / (1. + n_obs)
                    mean_prime = lam * mean_ml + (1. - lam) * mean0
                    sig_prime = gam * sig_ml + (1. - gam) * sig0
                    return np.array([mean_prime, sig_prime])  # FOR EACH Y_j = k !!
    
            elif self.marginal_description == 'discrete':
                # Discrete data: should be non-negative integers starting at 0: 0,...k
                x_select = (xi == np.arange(self.dim_visible)[:, np.newaxis])  # dim_v by ns
                prior = np.mean(x_select, axis=1).reshape((-1, 1, 1))  # dim_v, 1, 1
                n_obs = np.sum(p_y_given_x, axis=1)  # m, k
                counts = np.dot(x_select, p_y_given_x)  # dim_v, m, k
                p = counts + 0.001  # Tiny smoothing to avoid numerical errors
                p /= p.sum(axis=0, keepdims=True)
                if self.smooth_marginals:  # Shrinkage interpreted as hypothesis testing...
                    G_stat = 2 * np.sum(np.where(counts > 0, counts * (np.log(counts) - np.log(n_obs * prior)), 0), axis=0)
                    G0 = self.estimate_sig(x_select, p_y_given_x, n_obs, prior)
                    z = 1
                    lam = G_stat**z / (G_stat**z + G0**z)
                    lam = np.where(np.isnan(lam), 0.5, lam)
                    p = (1 - lam) * prior + lam * p
                    
                #fileName=self.fileName+'output_discrete_log_estimate20.txt'
                #f=open(fileName,'a')
                #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_discrete_log_estimate20.txt",'a')
                #print("\n count = \n",file=f)
                #print(self.countS2,file=f)
                #print("\np\n",file=f)
                #print(p,file=f)
                #f.close()
                #g=open(self.fileName+"estpARAMS.txt",'ab')
                #np.savetxt(g,["loopCount discrete\n"],fmt="%s")
                #np.savetxt(g,[self.loopCount])
                #np.savetxt(g,["xi\n"],fmt="%s")
                #np.savetxt(g,xi.reshape(-1))
                #np.savetxt(g,["\n class labels np.log\n"],fmt="%s")
                #np.savetxt(g,np.log(p).reshape(-1))
                #g.close()
                return np.log(p)
    
            else:
                print('Marginal description "%s" not implemented.' % self.marginal_description)
                sys.exit()
        else:
            x_select = (xi == np.arange(self.dim_visible)[:, np.newaxis])  # dim_v by ns
            prior = np.mean(x_select, axis=1).reshape((-1, 1, 1))  # dim_v, 1, 1
            n_obs = np.sum(p_y_given_x, axis=1)  # m, k
            counts = np.dot(x_select, p_y_given_x)  # dim_v, m, k
            p = counts + 0.001  # Tiny smoothing to avoid numerical errors
            p /= p.sum(axis=0, keepdims=True)
            if self.smooth_marginals:  # Shrinkage interpreted as hypothesis testing...
                G_stat = 2 * np.sum(np.where(counts > 0, counts * (np.log(counts) - np.log(n_obs * prior)), 0), axis=0)
                G0 = self.estimate_sig(x_select, p_y_given_x, n_obs, prior)
                z = 1
                lam = G_stat**z / (G_stat**z + G0**z)
                lam = np.where(np.isnan(lam), 0.5, lam)
                p = (1 - lam) * prior + lam * p
            #g=open(self.fileName+"estpARAMS.txt",'ab')
            #np.savetxt(g,["loopCount class labels\n"],fmt="%s")
            #np.savetxt(g,[self.loopCount])
            #np.savetxt(g,["xi\n"],fmt="%s")
            #np.savetxt(g,xi.reshape(-1))
            #np.savetxt(g,["\n class labels np.log\n"],fmt="%s")
            #np.savetxt(g,np.log(p).reshape(-1))
            #g.close()
           
            #fileName=self.fileName+'output_discrete_log_estimate20.txt'
            #f=open(fileName,'a')
            #f=open("/home/bb79e/Code/Bin_classify/bb79e3/output_discrete_log_estimate20.txt",'a')
            #print("\n count = \n",file=f)
            #print(self.countS2,file=f)
            #print("\np\n",file=f)
            #print(p,file=f)
            #f.close()
            return np.log(p)

    def estimate_se(self, xi, p_y_given_x, n_obs):
        # Get a bootstrap estimate of mean and standard error for estimating mu and sig^2 given | Y_j=k  (under null)
        # x_copy = np.hstack([np.random.choice(xi, size=(len(xi), 1), replace=False) for _ in range(20)])
        x_copy = np.random.choice(xi, size=(len(xi), 20), replace=True)  # w/o replacement leads to...higher s.e. and more smoothing.
        m, n, k = p_y_given_x.shape
        mean_ml = np.einsum('il,jik->jkl', x_copy, p_y_given_x) / n_obs[..., np.newaxis]  # ML estimate
        sig_ml = np.einsum('jikl,jik->jkl', (x_copy.reshape((1, n, 1, 20)) - mean_ml.reshape((m, 1, k, 20)))**2, p_y_given_x) / (n_obs[..., np.newaxis] - 1).clip(0.01) # ML estimate
        m1 = np.mean(mean_ml, axis=2)
        m2 = np.mean(sig_ml, axis=2)
        se1 = np.sqrt(np.sum((mean_ml - m1[..., np.newaxis])**2, axis=2) / 19.)
        se2 = np.sqrt(np.sum((sig_ml - m2[..., np.newaxis])**2, axis=2) / 19.)
        return m1, m2, se1, se2

    def estimate_sig(self, x_select, p_y_given_x, n_obs, prior):
        # Permute p_y_given_x, est mean Gs
        # TODO: This should be done using sampling with replacement instead of permutation.
        Gs = []
        for i in range(20):
            order = np.random.permutation(p_y_given_x.shape[1])
            counts = np.dot(x_select, p_y_given_x[:, order, :])  # dim_v, m, k
            Gs.append(2 * np.sum(np.where(counts > 0, counts * (np.log(counts) - np.log(n_obs * prior)), 0), axis=0))
        return np.mean(Gs, axis=0)

