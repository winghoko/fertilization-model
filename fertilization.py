'''
Implements the broadcast spawning fertilization model described in
Chan and Ko (2024) [https://doi.org/10.1093/icb/icae071].

The bulk of this module consists of the FertilizationModel class. In
addition, a convenient function factory make_const_func() is provided,
and a custom exception NotYetSolvedError is defined.
'''

import numpy as np, scipy.integrate as scint
from scipy.special import factorial

def make_const_func(value):
    '''
    Convenient function factory for creating constant functions. Can be useful
    when v, chi, and/or alpha are constant in a FertilizationModel instance.
    
    Arugment:
        value: the value to be returned by the function. Note that the value 
            should also take the proper type and "shape." For example, a scalar
            value is different from value that is a 1-axis 1-entry numpy array.
            
    Returns: a function that take a single argument and return the value 
        assigned.
    '''

    def func(t):
        return value

    return func

class NotYetSolvedError(NotImplementedError):
    '''
    Custom exception to indicated that a function is called without first 
    running the pre-requisite solver(s).
    '''
    pass

class FertilizationModel:
    '''
    Encapsulate the broadcast spawning fertilization model described in
    Chan and Ko (2024) [https://doi.org/10.1093/icb/icae071].
    
    In typical use, an instance is initialized with suitable parameters (some
    of which are functions. See the documentations of the __init__() magic 
    method for more details), after which the .solve_S0() AND .solve_pmn()
    methods are called, in that order, to "solve" for the behavior of the 
    model. Then, the appropriate function(s) can be called to find the desired
    egg or sperm concentration(s). For example, with the simple assumption that
    the first fertilizing sperm is the one that succeeds, the .S_ord() method
    with m = 1 can be called to find the concentration of eggs from a specific
    group that are fertilized by sperms of a specific group.
    
    Note that changing the parameters of the model after the initialization of
    the model object is strongly discouraged. If the fertilization outcome for
    multiple model parameters are needed (e.g., to find the dependence of 
    fertilziation on sperm concentration), it is advisable to initialize 
    multiple model objects.
    '''

    def __init__(self, v_func, x_func, S_init, a_func, sigma0, E_init, t_end):
        '''
        Initiialize the FertilizationModel class. Arguments:
            v_func: function to compute sperm speed for each sperm group at 
                time t. It should take time (scalar) as input and return an 
                n_lambda dimensional numpy array.
            x_func: function to compute sperm decay rate for each sperm group
                at time t. It should take time (scalar) as input and return 
                an n_lambda dimensional numpy array.
            S_init: An n_lambda dimensional numpy array representing the 
                initial MOTILE sperm concentration of each sperm group.
            a_func: function to compute ratio of fertilizable area to total 
                area for each egg group when hit by a sperm from a given
                sperm group. It should take time (scalar) as input and return
                an (n_lambda, n_gamma) dimensional numpy array of fertilizable
                ratio.
            sigma0: An n_gamma dimensional numpy array representing the total
                fertilizable area for each egg group.
            E_init: An n_gamma dimensional numpy array representing the 
                initial concentration of each egg group.
            t_end: the end time that the numerical differential equation 
                solver compute to (note: any results from solved function 
                computed with input time t > t_end should be considered 
                unrealiable).
        '''
        self.v = v_func
        self.x = x_func
        self.Si = S_init
        self.a = a_func
        self.sgm0 = sigma0
        self.Ei = E_init
        self.tf = t_end

        self.reset()

    def reset(self):
        '''
        Reset / recompute all quantities calculated based on the inputs 
        provided during object instantation. Should be run manually if any
        input attributes are modified by hand.
        '''
        self.S0_pwr = self.raise_S0_not_solved
        self.p_pwr = self.raise_pmn_not_solved

        self.Si_diag = np.diag(self.Si)
        self.Ei_diag = np.diag(self.Ei)
        self.sgm0Ei = self.sgm0 * self.Ei
        self.Eall = np.dot(self.sgm0, self.Ei)
        self.nlamda = self.Si.size
        self.ngamma = self.Ei.size

    @classmethod
    def raise_S0_not_solved(cls, *args, **kwargs):
        '''
        Raise NotYetSolvedError whenever self.S0_pwr (which is called by
        self.S0, as well as any functions relying on it) is being called 
        without first being solved.
        '''
        raise NotYetSolvedError("Please run .solve_S0() first")

    @classmethod
    def raise_pmn_not_solved(cls, *args, **kwargs):
        '''
        Raise NotYetSolvedError whenever self.p_pwr (which is called by 
        self.pmn and self.Ek, as well as any functions relying on them) is 
        being called without first being solved.
        '''
        raise NotYetSolvedError("Please run .solve_pmn() first")

    def dS0_pwr(self, t, y):
        '''
        WARNING: this function should be considered implementation details
        and may be changed in future without warning.
        
        Compute the integrand in the exponent of S0. Arguments:
            t: time (must be scalar).
            y: NOT USED. Included to comply with call signature expected 
                by scipy.integrate.solve_ivp().
        Returns: an n_lambda dimensional numpy array representing the 
            integrand of the exponent of S0 at time t.
        '''
        return self.x(t) + self.Eall * self.v(t)

    def solve_S0(self, **kwargs):
        '''
        Method to be called to solve for S_0 (free live sperm concentration). 
        Run scipy.integrate.solve_ivp() on appropriate function under the hood.
        Takes no positional arguments, while all keywords arguments are passed 
        to scipy.integrate.solve_ivp().
        
        Returns: the object returned by scipy.integrate.solve_ivp().
        
        Side effects: the callable atttributes .S0_pwr is now properly defined.
        Consequently, calling the methods .S0 and .dp_pwr will no longer raise 
        NotYetSolvedError.
        '''
        out = scint.solve_ivp(
            self.dS0_pwr, (0, self.tf), 
            np.zeros_like(self.Si), 
            dense_output=True, 
            **kwargs
        )
        self.S0_pwr = out.sol
        self.S0_pwr.__doc__ = '''
            Calculate the exponent of S_0 (free live sperm concentration).
            Argument:
                t: time at which S_0 is calculated (may be scalar or a 1-axis
                    numpy array).
            Returns: an n_lambda dimensional [for scalar t] or (n_lambda, t.size)
                dimensional [for 1-axis array t] numpy array representing the 
                exponent of S_0 at time t 
        '''

        return out

    def S0(self, t):
        '''
        Calculate S_0, the free live sperm concentration. Argument:
            t: time at which S_0 is calculated (may be scalar or a 1-axis
                numpy array).
        Returns: an n_lambda dimensional (for scalar t) or (n_lambda, t.size)
            dimensional (for 1-axis array t) numpy array representing the 
            the value of S_0 at time t 
        '''
        return np.tensordot(self.Si_diag, np.exp(-self.S0_pwr(t)), 1)

    def dp_pwr(self, t, y):
        '''
        WARNING: this function should be considered implementation details
        and may be changed in future without warning.
        
        Calculate the integrand of the exponent of E_k (fertilized egg 
        concentration) and p_{m,n} (probability of future fertilization).
        Argument: 
            t: time (must be scalar).
            y: NOT USED. Included to comply with call signature expected 
                by scipy.integrate.solve_ivp().
        Returns:  an n_gamma dimensional numpy array representing the 
            integrand of the exponent of p_{m,n} and E_k at time t.
        '''
        vSall = self.v(t) * self.Si * np.exp(-self.S0_pwr(t))
        avSall = np.tensordot(vSall, self.a(t), 1)
        return avSall * self.sgm0

    def solve_pmn(self, **kwargs):
        '''
        Method to be called to solve for E_k (fertilized egg concentration) and
        p_{m,n} (probability of future fertilization of egg). 
        Run scipy.integrate.solve_ivp() on appropriate function under the hood.
        Takes no positional arguments, while all keywords arguments are passed 
        to scipy.integrate.solve_ivp().
        
        Returns: the object returned by scipy.integrate.solve_ivp().
        
        Side effects: the callable atttributes .p_pwr is now properly defined.
        Consequently, calling the methods .Ek and .pmn, as well as any further
        methods that rely on them (e.g., .S_mn), will no longer raise 
        NotYetSolvedError.
        '''
        f = self.dp_pwr
        out = scint.solve_ivp(
            f, (0, self.tf), 
            np.zeros_like(self.Ei),
            dense_output=True,
            **kwargs
        )
        self.p_pwr = out.sol
        self.p_pwr.__doc__ = '''
            Calculate the exponent of E_k (fertilized egg concentration) and 
            p_{m,n} (probability of future fertilization).
            
            NOTE that in the latter case the integral is from t to T, so one
            should call p_pwr(T) - p_pwr(t) for the correct exponent.
            
            Argument: 
                t: time at which the exponent is calculated (may be scalar or
                    a 1-axis numpy array).
                    
            Returns: an n_gamma dimensional [for scalar t] or (n_gamma, t.size)
                dimensional [for 1-axis array t] numpy array representing the 
                exponent of E_k and p_{m,n} at time t 
        '''

        return out

    def Ek(self, t, k):
        '''
        Calculate the E_k, the concentration of eggs that are k times 
        fertilized. Arguments:
            t: the time at which the concentration is calculated, can
                be a scalar or a 1-axis numpy array.
            k: the number of fertilization for which the concentration
                is calculated. Must be a positive scalar integer.
        Returns: an n_gamma dimensional [for scalar ]) or (n_gamma, t.size)
            dimensional [for 1-axis array t] numpy array representing the 
            value of E_k at time t.
        '''
        Lambda = self.p_pwr(t)
        return np.tensordot(
            self.Ei_diag, np.exp(-Lambda) * (Lambda**k) / factorial(k), 1
        )

    def E_any(self, t):
        '''
        Calculate E_*, the concentration of eggs that are fertilized at
        least one time. Arguments:
            t: the time at which the concentration is calculated, can
                be a scalar or a 1-axis numpy array.
        Returns: an n_gamma dimensional [for scalar t] or (n_gamma, t.size)
            dimensional [for 1-axis array t] numpy array representing the 
            value of E_k at time t.
        '''
        Lambda = self.p_pwr(t)
        return np.tensordot(self.Ei_diag, 1 - np.exp(-Lambda), 1)

    def pmn(self, t1, t2, k):
        '''
        Calculate the probablity p_{k = n - m} that an egg will be 
        fertilized k more times between t1 and t2. Arguments:
            t1: the initial time, can be a scalar or a 1-axis numpy array.
            t2: the end time, must have compatible dimension with t1.
            k: the number of future fertilization. Must be a positive scalar
                integer.
        Returns: an n_gamma dimensional [for scalar t] or (n_gamma, t.size)
            dimensional [for 1-axis array t] numpy array representing the 
            value of p_{k = n-m}(t1, t2).
        '''
        Lambda = self.p_pwr(t2) - self.p_pwr(t1)
        return np.exp(-Lambda) * (Lambda**k) / factorial(k)

    def dSany(self, t):
        '''
        WARNING: this function should be considered implementation details
        and may be changed in future without warning.
        
        Compute the integrand for the calculation of S_* simultaneously
        for all sperm and egg group pairs. Arguments:
            t: time at which the concentration is calculated (must be scalar).
        Returns: a (n_lambda, n_gamma) dimensional numpy array representing
            the integrand of S_* at time t.
        '''
        lFactor = self.v(t) * self.S0(t)
        return lFactor[:, np.newaxis] * self.a(t) * self.sgm0Ei[np.newaxis, :]

    def dSmn(self, t1, t2, m, n):
        '''
        WARNING: this function should be considered implementation details
        and may be changed in future without warning.
        
        Compute the integrand for the calculation of S_{m,n} simultaneously
        for all sperm and egg group pairs. Arguments:
            t1: time at which the tracked fertilization occurs (must be 
                scalar).
            t2: time at which the total number of fertilization is counted
                (must be scalar).
            m: the order of fertilization of the fertilization event being
                tracked (must be a positive scalar integer).
            n: the total number of fertilization (must be a positive scalar 
                integer).
        Returns: a (n_lambda, n_gamma) dimensional numpy array representing
            the integrand of S_{m,n} at time t.
        '''
        gFactor = self.sgm0 * self.Ek(t1, m-1) * self.pmn(t1, t2, n-m)
        lFactor = self.v(t1) * self.S0(t1)
        return lFactor[:, np.newaxis] * self.a(t1) * gFactor[np.newaxis, :]

    def dSn(self, t1, t2, n):
        '''
        WARNING: this function should be considered implementation details
        and may be changed in future without warning.
        
        Compute the integrand for the calculation of S_{*, n} simultaneously
        for all sperm and egg group pairs. Arguments:
            t1: time at which the tracked fertilization occurs (must be 
                scalar).
            t2: time at which the total number of fertilization is counted
                (must be scalar).
            n: the total number of fertilizations (must be a positive scalar
                integer).
        Returns: a (n_lambda, n_gamma) dimensional numpy array representing
            the integrand of S_{*,n} at time t.
        '''
        gFactor = self.sgm0 * self.Ek(t2, n-1)
        lFactor = self.v(t1) * self.S0(t1)
        return lFactor[:, np.newaxis] * self.a(t1) * gFactor[np.newaxis, :]

    def dSm(self, t, m):
        '''
        WARNING: this function should be considered implementation details
        and may be changed in future without warning.
        
        Compute the integrand for the calculation of S_{m,*} simultaneously
        for all sperm and egg group pairs. Arguments:
            t: time at which the m-th fertilization occurs (must be scalar).
            m: the order of fertilization of the fertilization event being
                tracked (must be a positive scalar integer).
        Returns: a (n_lambda, n_gamma) dimensional numpy array representing
            the integrand of S_{m,*} at time t.
        '''
        gFactor = self.sgm0 * self.Ek(t, m-1)
        lFactor = self.v(t) * self.S0(t)
        return lFactor[:, np.newaxis] * self.a(t) * gFactor[np.newaxis, :]

    def dEpoly(self, t, Dt, **kwargs):
        '''
        WARNING: this function should be considered implementation details
        and may be changed in future without warning.
        
        Compute the integrand for the calculation of E_{poly} simultaneously
        for all egg group pairs. Arguments:
            t: time by which ferilization occurs (must be scalar).
            Dt: time window between first fertilization and the (fast) block
                on further fertilizations.
        Returns: a (n_lambda, n_gamma) dimensional numpy array representing
            the integrand of E_{poly} at time t
        '''
        gFactor = self.sgm0 * self.Ek(t, 0) * (1 - self.pmn(t, t + Dt, 0))
        lFactor = self.v(t) * self.S0(t)
        return lFactor[:, np.newaxis] * self.a(t) * gFactor[np.newaxis, :]

    def S_mn(self, t, m, n, **kwargs):
        '''
        Compute S_{m,n} simultaneously for all sperm and egg group pairs.
        Uses scipy.integrate.quad_vec() under the hood. Arguments:
            t: time at which the total number of fertilization is counted, as
                well as the time by which tracked fertilization has occured
                (can be a scalar or a 1-axis numpy array).
            m: the order of fertilization of fertilizing event being tracked 
                (must be a positive scalar integer).
            n: the total number of fertilization (must be a positive scalar 
                integer).
         Remaining keyword arguments are passed to scipy.integrate.quad_vec().
        
        Returns:
            If t is scalar, object return by scipy.integrate.quad_vec(),
                where the value of S_{m,n} is accessible as the zeroth element
                as an (n_lambda, n_gamma) dimensional numpy array.
            If t is a 1-axis numpy array, a (n_lambda, n_gamma, t.size) numpy
                array, where the last axis corresponds to the time points in t,
                and the first two axes give the n_lambda-by-n_gamma dimensional
                matrix of S_{m,n} at the specified time.
        '''
        if np.isscalar(t):
            return scint.quad_vec(
                self.dSmn, 0, t, args=(t, m, n), **kwargs
            )
        else:
            t_shape = list(t.shape)
            t_array = t.ravel()
            out_array = np.zeros((self.nlamda, self.ngamma, t_array.size))
            func = self.S_mn
            for i, t_val in enumerate(t_array):
                out_array[:, :, i] = func(t_val, m, n, **kwargs)[0]
            return out_array.reshape([self.nlamda, self.ngamma] + t_shape)

    def S_ord(self, t, m, **kwargs):
        '''
        Compute S_{m,*} simultaneously for all sperm and egg group pairs.
        The "_ord" in the method name stands for "ordinal." Uses 
        scipy.integrate.quad_vec() under the hood. Arguments:
            t: time by which the tracked fertilization has occured (can be a
                scalar or a 1-axis numpy array).
            m: the order of fertilization of the fertilizing event being
                tracked (must be a positive scalar integer).
        Remaining keyword arguments are passed to scipy.integrate.quad_vec().

        Returns:
            If t is scalar, object return by scipy.integrate.quad_vec(),
                where the value of S_{m,*} is accessible as the zeroth element
                as an (n_lambda, n_gamma) dimensional numpy array.
            If t is a 1-axis numpy array, a (n_lambda, n_gamma, t.size) numpy
                array, where the last axis corresponds to the time points in t,
                and the first two axes give the n_lambda-by-n_gamma dimensional
                matrix of S_{m,*} at the specified time.
        '''
        if np.isscalar(t):
            return scint.quad_vec(self.dSm, 0, t, args=(m,), **kwargs)
        else:
            t_shape = list(t.shape)
            t_array = t.ravel()
            out_array = np.zeros((self.nlamda, self.ngamma, t_array.size))
            func = self.S_ord
            for i, t_val in enumerate(t_array):
                out_array[:, :, i] = func(t_val, m, **kwargs)[0]
            return out_array.reshape([self.nlamda, self.ngamma] + t_shape)

    def S_card(self, t, n, **kwargs):
        '''
        Compute S_{*,n} simultaneously for all sperm and egg group pairs.
        The "_card" in the method name stands for "cardinal." Uses
        scipy.integrate.quad_vec() under the hood. Arguments:
            t: time by which the total number of fertilization is counted 
                (can be a scalar or a 1-axis numpy array).
            n: the total number of fertilization (must be a positive scalar
                integer).
        Remaining keyword arguments are passed to scipy.integrate.quad_vec().
        
        Returns:
            If t is scalar, object return by scipy.integrate.quad_vec(),
                where the value of S_{*,n} is accessible as the zeroth element
                as an (n_lambda, n_gamma) dimensional numpy array
            If t is a 1-axis numpy array, a (n_lambda, n_gamma, t.size) numpy
                array, where the last axis corresponds to the time points in t,
                and the first two axes give the n_lambda-by-n_gamma dimensional
                matrix of S_{*,n} at the specified time
        '''
        if np.isscalar(t):
            return scint.quad_vec(
                self.dSn, 0, t, args=(t, n), **kwargs
            )
        else:
            t_shape = list(t.shape)
            t_array = t.ravel()
            out_array = np.zeros((self.nlamda, self.ngamma, t_array.size))
            func = self.S_card
            for i, t_val in enumerate(t_array):
                out_array[:, :, i] = func(t_val, n, **kwargs)[0]
            return out_array.reshape([self.nlamda, self.ngamma] + t_shape)

    def S_any(self, t, **kwargs):
        '''
        Compute S_* simultaneously for all sperm and egg group pairs. Uses 
        scipy.integrate.quad_vec() under the hood. Arguments:
            t: time by which the fertilization has occured (can be a scalar 
                or a 1-axis numpy array).
        Remaining keyword arguments are passed to scipy.integrate.quad_vec().
        
        Returns:
            If t is scalar, object return by scipy.integrate.quad_vec(),
                where the value of S_* is accessible as the zeroth element
                as an (n_lambda, n_gamma) dimensional numpy array
            If t is a 1-axis numpy array, a (n_lambda, n_gamma, t.size) numpy
                array, where the last axis corresponds to the time points in t,
                and the first two axes give the n_lambda-by-n_gamma dimensional
                matrix of S_* at the specified time
        '''
        if np.isscalar(t):
            return scint.quad_vec(self.dSany, 0, t, **kwargs)
        else:
            t_shape = list(t.shape)
            t_array = t.ravel()
            out_array = np.zeros((self.nlamda, self.ngamma, t_array.size))
            func = self.S_any
            for i, t_val in enumerate(t_array):
                out_array[:, :, i] = func(t_val, **kwargs)[0]
            return out_array.reshape([self.nlamda, self.ngamma] + t_shape)

    def E_poly(self, t, Dt, **kwargs):
        '''
        Compute E_{poly} simultaneously for all sperm and egg group pairs.
        By definition, polyspermy occurs if additional sperm(s) fertilizes the
        egg within a time window of the first fertilization event. Uses 
        scipy.integrate.quad_vec() under the hood. Arguments:
            t: time by which the first fertilization has occured (can be
                a scalar or a 1-axis numpy array).
            Dt: the time window between the first fertilization event and the
                fast block on further fertilizations.
        Remaining keyword arguments are passed to scipy.integrate.quad_vec().
        
        Returns:
            If t is scalar, object return by scipy.integrate.quad_vec(),
                where the value of E_{poly} is accessible as the zeroth element
                as an (n_lambda, n_gamma) dimensional numpy array
            If t is a 1-axis numpy array, a (n_lambda, n_gamma, t.size) numpy
                array, where the last-axis corresponds to the time points in t,
                and the first two axes is the n_lambda-by-n_gamma dimensional
                matrix given E_{poly} at the specified time
        '''
        if np.isscalar(t):
            return scint.quad_vec(self.dEpoly, 0, t, args=(Dt,), **kwargs)
        else:
            t_shape = list(t.shape)
            t_array = t.ravel()
            out_array = np.zeros((self.nlamda, self.ngamma, t_array.size))
            func = self.E_poly
            for i, t_val in enumerate(t_array):
                out_array[:, :, i] = func(t_val, Dt, **kwargs)[0]
            return out_array.reshape([self.nlamda, self.ngamma] + t_shape)

    def aggregate_S_mn(self, t_eval, n_min=3, n_max=10, cutoff=0.98, *, 
        S_eps = 1.0e-6, calc_Sn=False, verbose=False, **kwargs):
        '''
        Compute S_{m,n} starting from n = 1 until a cutoff value of n = n_cut 
        is reached. For each n, the value of S_{m,n} for all possible values 
        of m is calculated. The value of n_cut can be specified explicitly or 
        be determined implicitly by requiring that terms sum of computed 
        S_{m,n} is a sufficiently large portion of S_*.
        
        NOTE that the sum of S_{m,n} is compared to S_* only at the last time
        point (for the case where t_eval is a numpy array).
        
        Arguments:
            t_eval: the time at which the total number of fertilization is 
                counted, as well as the time by which tracked fertilization 
                has occured. Should be a scalar or a 1-axis numpy array.
            n_min: the minimum number of n (total number of fertilization)
                to compute until the calculation terminates. Must be a 
                positive scalar integer.
            n_max: the maximum number of n (total number of fertilization) 
                to compute before the calculation terminates. Must be a 
                positive scalar integer (pick a sufficiently large value
                to ensure that the cutoff criterion is being used for 
                termination).
            cutoff: the cutoff value to use for terminating computation. By 
                definition, calculation is terminated if the computed S_{m,n} 
                (up to fixed n >= n_min) is at least cutoff * S_*. More 
                precisely, since both S_{m,n} and S_* are n_lamda-by-n_gamma 
                matrices, calculation is terminated only if S_{m,n} / S_* > 
                cutoff for ALL entries in the matrix (pick cutoff > 1 to 
                ensure that n_max criterion is being used for termination).
                
        Keyword only arguments:
            S_eps: the value below which an entry of S_* is being treated as 
                zero and the ratio of S_{m,n} / S_* is DEFINED to be 1.0.
                Should be a floating point scalar.
            calc_Sn: whether to also compute the values of S_{*,n} up to the
                same value of n. Can be True of False
            verbose: whether to print out additional information during the 
                computation. Can be True or False.
        Remaining keyword arguments are passed to scipy.integrate.quad_vec(),
        which is used under the hood when calculating S_* and S_{m,n}
        
        Returns:
            If calc_Sn is False, a dictionary whose keys are the doublets (m,n)
                and whose correponding values are the calculated values of 
                S_{m,n}. If t is a scalar, S_{m,n} is provided as an (n_lambda,
                n_gamma)-dimensional array. If t is a 1-axis numpy array, 
                S_{m,n} is provided in the form of an (n_lambda, n_gamma, 
                t_size)-dimensional numpy array. See the documentation of 
                .S_mn() for details.
            If calc_Sn is True, a doublet of dictionaries, the first of which 
                is the dictionary of S_{m,n} as stated above, the second of 
                which is the dictionary of S_{n}, whose keys are integer 
                (values of n) and whose correpsonding values are the values 
                of S_{*,n} in the form of an (n_lambda, n_gamma)-dimensional
                array or an (n_lambda, n_gamma, t_size)-dimensional numpy 
                array, depending on whether t is a scalar or a 1-axis numpy
                array.
        '''

        # coerce the time points into a 1D numpy array
        if np.isscalar(t_eval):
            t_eval = np.array([t_eval])
            t_shape = list()
        else:
            t_shape = list(t_eval.shape)
        t_eval = t_eval.ravel()
        
        # define variables that persist through loops
        out_dict = dict()
        total_mn = np.zeros((self.nlamda, self.ngamma))
        total_all = self.S_any(t_eval[-1], **kwargs)[0]

        for n in range(1, n_max + 1):

            for m in range(1, n + 1):

                out = self.S_mn(t_eval, m, n, **kwargs)
                total_mn += out[:, :, -1]
                out_dict[(m,n)] = out.reshape(
                    [self.nlamda, self.ngamma] + t_shape
                )

            # check the proportion of fertilizing sperm accounted for
            r = np.min(np.divide(
                total_mn, total_all, 
                out=np.ones_like(total_all), where=(total_all > S_eps)
            ))

            if verbose:
                print("Results for n = {} aggregrated".format(n))
                print("  ratio of computed/total = {:.5f}".format(r))

            if (n >= n_min) and (r > cutoff):
                if verbose:
                    print("\nAggregation terminated at n = {}".format(n))
                break

        else: # else of for
            if verbose:
                print("\nMaximum n of {} reached. Terminating.".format(n_max))

        # calculate the corresponding S_{*,n} is asked
        n_term = n
        if calc_Sn:

            out2_dict = dict()
            if verbose: print("")

            for n in range(1, n_term + 1):

                out =  np.zeros([self.nlamda, self.ngamma] + t_shape)

                for m in range (1, n + 1):
                    out += out_dict[(m, n)]

                out2_dict[n] = out

                if verbose:
                    print("S_{{*,n}} for n = {} calculated".format(n))

            if verbose: print("\nDone\n")
            return out_dict, out2_dict

        else:
            if verbose: print("\nDone\n")
            return out_dict

    def aggregate_S_card(self, t_eval, n_min=3, n_max=10, cutoff=0.98, *, 
        S_eps = 1.0e-6, verbose=False, **kwargs):
        '''
        Compute S_{*,n} starting from n = 1 until a cutoff value of n = n_cut
        is reached. The value of n_cut can be specified explicitly or be
        determined implicitly by requiring that terms sum of computed S_{*,n}
        is a sufficiently large portion of S_*.
        
        NOTE that the sum of S_{*,n} is compared to S_* only at the last time
        point (for the case where t_eval is a numpy array).
        
        Arguments:
            t_eval: the time at which the total number of fertilization is 
                counted, as well as the time by which tracked fertilization 
                has occured. Should be a scalar or a 1-axis numpy array.
            n_min: the minimum number of n (total number of fertilization)
                to compute until the calculation terminates. Must be a 
                positive scalar integer.
            n_max: the maximum number of n (total number of fertilization) 
                to compute before the calculation terminates. Must be a 
                positive scalar integer (pick a sufficiently large value
                to ensure that the cutoff criterion is being used for 
                termination).
            cutoff: the cutoff value to use for terminating computation. By 
                definition, calculation is terminated if the computed S_{*,n} 
                (up to fixed n >= n_min) is at least cutoff * S_*. More 
                precisely, since both S_{*,n} and S_* are n_lamda-by-n_gamma 
                matrices, calculation is terminated only if S_{*,n} / S_* > 
                cutoff for ALL entries in the matrix (pick cutoff > 1 to 
                ensure that n_max criterion is being used for termination).
                
        Keyword only arguments:
            S_eps: the value below which an entry of S_* is being treated as 
                zero and the ratio of S_{*,n} / S_* is DEFINED to be 1.0.
                Should be a floating point scalar.
            verbose: whether to print out additional information during the 
                computation. Can be True or False.
        Remaining keyword arguments are passed to scipy.integrate.quad_vec(),
        which is used under the hood when calculating S_* and S_{*,n}
        
        Returns: a dictionary whose keys are the values of n and whose 
        correponding values are the calculated values of S_{*,n}. If t is a
        scalar, S_{*,n} is provided as an (n_lambda, n_gamma)-dimensional 
        array. If t is a 1-axis numpy array, S_{*,n} is provided in the form
        of an (n_lambda, n_gamma, t_size)-dimensional numpy array. See the 
        documentation of .S_card() for details.
        '''

        if np.isscalar(t_eval):
            t_eval = np.array([t_eval])
            t_shape = list()
        else:
            t_shape = list(t_eval.shape)
        t_eval = t_eval.ravel()
        
        out_dict = dict()
        total_mn = np.zeros((self.nlamda, self.ngamma))
        total_all = self.S_any(t_eval[-1], **kwargs)[0]
        
        for n in range(1, n_max + 1):

            out = self.S_card(t_eval, n, **kwargs)
            total_mn += out[:, :, -1]

            out_dict[n] = out.reshape(
                [self.nlamda, self.ngamma] + t_shape
            )

            r = np.min(np.divide(
                total_mn, total_all, 
                out=np.ones_like(total_all), where=(total_all > S_eps)
            ))

            if verbose:
                print("Results for n = {} aggregrated".format(n))
                print("  ratio of computed/total = {:.5f}".format(r))

            if (n >= n_min) and (r > cutoff):
                if verbose:
                    print("\nAggregation terminated at n = {}".format(n))
                break

        else: # else of for
            if verbose:
                print("\nMaximum n of {} reached. Terminating.".format(n_max))

        return out_dict

    def aggregate_S_ord(self, t_eval, m_min=3, m_max=10, cutoff=0.98, *, 
        S_eps = 1.0e-6, verbose=False, **kwargs):
        '''
        Compute S_{m,*} starting from m = 1 until a cutoff value of m = m_cut
        is reached. The value of m_cut can be specified explicitly or be
        determined implicitly by requiring that terms sum of computed S_{m,*}
        is a sufficiently large portion of S_*.
        
        NOTE that the sum of S_{m,*} is compared to S_* only at the last time
        point (for the case where t_eval is a numpy array).
        
        Arguments:
            t_eval: the time at which the total number of fertilization is 
                counted, as well as the time by which tracked fertilization 
                has occured. Should be a scalar or a 1-axis numpy array.
            m_min: the minimum number of m (order of fertilization) to 
                compute until the calculation terminates. Must be a positive
                scalar integer.
            m_max: the maximum number of m (order of fertilization) to compute
                before the calculation terminates. Must be a positive scalar
                integer (pick a sufficiently large value to ensure that the 
                cutoff criterion is being used for termination).
            cutoff: the cutoff value to use for terminating computation. By 
                definition, calculation is terminated if the computed S_{m,*} 
                (up to fixed m >= m_min) is at least cutoff * S_*. More 
                precisely, since both S_{m,*} and S_* are n_lamda-by-n_gamma 
                matrices, calculation is terminated only if S_{m,*} / S_* > 
                cutoff for ALL entries in the matrix (pick cutoff > 1 to 
                ensure that m_max criterion is being used for termination).
                
        Keyword only arguments:
            S_eps: the value below which an entry of S_* is being treated as 
                zero and the ratio of S_{*,n} / S_* is DEFINED to be 1.0.
                Should be a floating point scalar.
            verbose: whether to print out additional information during the 
                computation. Can be True or False.
        Remaining keyword arguments are passed to scipy.integrate.quad_vec(),
        which is used under the hood when calculating S_* and S_{m,*}
        
        Returns: a dictionary whose keys are the values of m and whose 
        correponding values are the calculated values of S_{m,*}. If t is a
        scalar, S_{m,*} is provided as an (n_lambda, n_gamma)-dimensional 
        array. If t is a 1-axis numpy array, S_{m,*} is provided in the form
        of an (n_lambda, n_gamma, t_size)-dimensional numpy array. See the 
        documentation of .S_ord() for details.
        '''

        if np.isscalar(t_eval):
            t_eval = np.array([t_eval])
            t_shape = list()
        else:
            t_shape = list(t_eval.shape)
        t_eval = t_eval.ravel()

        out_dict = dict()
        total_mn = np.zeros((self.nlamda, self.ngamma))
        total_all = self.S_any(t_eval[-1], **kwargs)[0]

        for m in range(1, m_max + 1):

            out = self.S_ord(t_eval, m, **kwargs)
            total_mn += out[:, :, -1]

            out_dict[m] = out.reshape(
                [self.nlamda, self.ngamma] + t_shape
            )

            r = np.min(np.divide(
                total_mn, total_all, 
                out=np.ones_like(total_all), where=(total_all > S_eps)
            ))

            if verbose:
                print("Results for m = {} aggregrated".format(m))
                print("  ratio of computed/total = {:.5f}".format(r))

            if (m >= m_min) and (r > cutoff):
                if verbose:
                    print("\nAggregation terminated at m = {}".format(m))
                break

        else: # else of for
            if verbose:
                print("\nMaximum m of {} reached. Terminating.".format(m_max))

        return out_dict

__all__ = [
    "make_const_func",
    "NotYetSolvedError", 
    "FertilizationModel"
]