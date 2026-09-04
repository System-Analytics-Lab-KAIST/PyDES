import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, truncnorm, invgamma
from scipy.stats import norm as gaussian, chisquare, kstest, expon
from scipy.stats import multivariate_t, multivariate_normal
from scipy.optimize import root_scalar, minimize
from scipy.special import expit, logsumexp
import time
# import pymc as pm
# import arviz as az

def make_dist_generator(stream_name, dist_name, *param):
    rng = RndNumGen(stream_name)
    rng.set_distribution(dist_name, *param)
    return rng

def _make_raw_stream(seed):
    stream = np.random.default_rng(seed)  # standard Numpy random stream
    # stream = Generator(PCG64DXSM(seed))  # PCG64DXSM is more stable in parallel computing
    return stream

def _set_seed(stream, seed):
    temp_new_stream = _make_raw_stream(seed)
    stream.bit_generator.state = temp_new_stream.bit_generator.state


class RndNumGen:
    stream_pool = {}
    # CRN_Sep = True  # for CRN test
    AT_ST = False   # for antithetic test
    AT_IAT = False  # for antithetic test

    @classmethod
    def is_initialized(cls):
        return len(cls.stream_pool) > 0

    @classmethod
    def _get_seeds(cls, master_seed=None):
        max_stream = 100
        ss = SeedSequence(master_seed)
        cls.seeds = ss.generate_state(max_stream)

    @classmethod
    def init_rnd_generators(cls, master_seed=None):
        cls.stream_pool = {}
        cls._get_seeds(master_seed)
        cls.stream0 = cls.get_stream(None)        
        cls.rng0 = RndNumGen(None)

    @classmethod
    def set_master_seed(cls, master_seed=None):
        if not cls.is_initialized():
            cls.init_rnd_generators(master_seed=master_seed)
            return
        cls._get_seeds(master_seed)
        for L in cls.stream_pool.values():
            stream = L[0]
            n = L[2]
            L[1] = cls.seeds[n]  # new seed
            _set_seed(stream, L[1])

    @classmethod
    def make_stream(cls, stream_name):
        n = len(cls.stream_pool)  # use next seed
        seed = cls.seeds[n]
        # stream = np.random.default_rng(seed)
        stream = _make_raw_stream(seed)
        cls.stream_pool[stream_name] = [stream, seed, n]
        return stream

    @classmethod
    def get_stream(cls, stream_name):
        try:
            tup = cls.stream_pool[stream_name]
            return tup[0]
        except KeyError:
            if len(cls.stream_pool) >= len(cls.seeds):
                m = len(cls.seeds)
                print(f"# of RNG stream exceeds {m}. Are you sure?")
                cls.seeds = cls.ss.generate_state(m*2)
            return cls.make_stream(stream_name)

    @classmethod
    def reset_seed_stream(cls, stream_name):
        stream, seed = cls.stream_pool[stream_name]
        _set_seed(stream, seed)

    @classmethod
    def reset_seed_all(cls):
        for L in cls.stream_pool.values():
            _set_seed(L[0], L[1])

    # None for default shared RNG
    def __init__(self, stream_name=None):
        if not RndNumGen.is_initialized():
            RndNumGen.init_rnd_generators()
        # if not RndNumGen.CRN_Sep: 
        #     stream_name = None  # for testing CRN effect
        self.stream_name = stream_name
        self.stream = RndNumGen.get_stream(stream_name)

    def reset_seed(self, seed=None):
        L = RndNumGen.stream_pool[self.stream_name]
        if seed is None:
            seed = L[1]
        else:
            L[1] = seed
        _set_seed(self.stream, seed)

    def get_U01(self, size=None):
        return self.stream.random(size=size)

    def get_U01_stored(self, size=None):
        return self.stream.random(size=size)

    def get_N01(self, size=None):
        return self.stream.standard_normal(size=size)

    def get_N01_stored(self, size=None):
        return self.stream.standard_normal(size=size)

    def get_uniform(self, low, high, size=None):
        return self.stream.uniform(low, high, size=size)

    def get_uniform_stored(self, size=None):
        return self.stream.uniform(*self.params[:2], size=size)

    def get_unif_int(self, low, high, size=None):
        return self.stream.integers(low, high, size=size)

    def get_unif_int_stored(self, size=None):
        return self.stream.integers(*self.params[:2], size=size)

    def get_exponential(self, mean, size=None):
        return self.stream.exponential(mean, size=size)

    # def get_exponential_stored(self, size=None):
    #     return self.stream.exponential(self.params[0], size=size)

    def get_exponential_stored(self, size=None):
        is_IAT = (self.stream_name == "IAT")
        is_ST = (self.stream_name == "ST")
        AT = (is_IAT and self.AT_IAT) or (is_ST and self.AT_ST)
        u = self.stream.random(size=size)
        if AT: u = 1 - u
        return -self.params[0] * np.log(u)

    def get_triangular(self, low, mode, high, size=None):
        return self.stream.triangular(low, mode, high, size=size)

    def get_triangular_stored(self, size=None):
        return self.stream.triangular(*self.params[:3], size=size)

    def get_normal(self, mu, sigma, size=None):
        return self.stream.normal(mu, sigma, size=size)

    def get_normal_stored(self, size=None):
        return self.stream.normal(*self.params[:2], size=size)

    def get_trunc_normal(self, mu, sigma, low, high, size=None):
        z_low = (low - mu)/sigma
        z_high = (high - mu)/sigma
        return truncnorm.rvs(z_low, z_high, mu, sigma, size=size)

    def get_trunc_normal_stored(self, size=None):
        if (size is None or size == 1) and self.use_rejection:
            while True:
                x = self.stream.normal(*self.params[:2])
                if self.params[2] <= x < self.params[3]: 
                    return x  # rejection sampling
        # truncnorm.rvs() is much slower than rejection sampling
        return truncnorm.rvs(self.z_low, self.z_high, self.params[0], self.params[1], size=size)

    def get_gamma(self, k_shape, theta_scale, size=None):
        # mean = k_shape*theta_scale
        # var = mean*theta_scale
        return self.stream.gamma(k_shape, theta_scale, size=size)

    def get_gamma_stored(self, size=1):
        return self.stream.gamma(self.k_shape, self.theta_scale, size=size)

    def get_exp_gamma(self, alpha, exp_mean, k_shape, theta_scale, size=None):
        u = self.get_U01(size=size)
        b = (u < alpha)
        e = self.stream.exponential(exp_mean, size=size)
        g = self.stream.gamma(k_shape, theta_scale, size=size)
        return b*e + (1-b)*g 
        # if self.get_U01() < alpha:  # exponential
        #     return self.stream.exponential(exp_mean)
        # else: # gamma
        #     return self.stream.gamma(k_shape, theta_scale)

    def get_exp_gamma_stored(self, size=None):
        return self.get_exp_gamma(self.alpha, self.exp_mean, self.k_shape, self.theta_scale, size=size)

    def get_beta(self, alpha, beta, size=None):
        return self.stream.beta(alpha, beta, size=size)

    def get_beta_stored(self, size=None):
        return self.get_beta(*self.params[:2], size=size)

    # beta4 : loc + beta * (high-low)
    def get_beta4(self, alpha, beta, low=0, high=1, size=None):
        beta_sample = self.stream.beta(alpha, beta, size=size)
        return low + beta_sample * (high - low)  # min-max scaling 적용

    def get_beta4_stored(self, size=None):
        return self.get_beta4(*self.params[:4], size=size)

    def get_choice(self, choices, probs=None, size=None):
        x = self.stream.choice(choices, p=probs, size=size)
        return x

    def get_choice_stored(self, size=None):
        x = self.stream.choice(self.choices, p=self.probs, size=size)
        return x

    def set_gamma_param(self, mean, sigma):
        self.theta_scale = (sigma * sigma) / mean
        self.k_shape = mean / self.theta_scale

    def set_exp_gamma_param(self, alpha, exp_mean, g_mean, g_sigma):
        self.alpha = alpha
        self.exp_mean = exp_mean
        self.set_gamma_param(g_mean, g_sigma)

    def set_choice_param(self, choices, probs):
        self.choices, self.probs = choices, probs
        # self.n_choices = len(choices)

    def set_trunc_normal_param(self, mu, sigma, low, high):
        self.z_low = (low - mu)/sigma
        self.z_high = (high - mu)/sigma
        Z = gaussian.cdf(self.z_high) - gaussian.cdf(self.z_low)
        self.use_rejection = (Z >= 0.5)

    def set_distribution_tuple(self, dist_name, params):
        self.dist_name = dist_name
        self.params = params
        if dist_name == 'gamma':
            self.set_gamma_param(*params[:2])
        elif dist_name == 'choice':
            self.set_choice_param(*params[:2])
        elif dist_name == 'exp_gamma':
            self.set_exp_gamma_param(*params[:4])
        elif dist_name == 'trunc_normal':
            self.set_trunc_normal_param(*params[:4])
        try:
            self.get = getattr(self, f"get_{dist_name}_stored")  # get에 불릴 function 저장
            return self.get
        except Exception as e:
            print (f"Exception occured in set_distribution: {e}")
            return None

    def set_distribution(self, dist_name, *params):
        return self.set_distribution_tuple(dist_name, params)

    def get_mean_var(self):  # return mean, var
        def beta_mv(a, b):
            a_b = a + b
            m = a / a_b
            v = a*b / (a_b**2) / (a_b + 1)
            return m, v
        
        if self.dist_name == "U01":
            return 0.5, 1/12
        elif self.dist_name == "N01":
            return 0, 1
        elif self.dist_name == "uniform":
            w = (self.params[1] - self.params[0])
            return (self.params[0] + self.params[1])/2, w*w/12
        elif self.dist_name == "unif_int":
            a, b = self.params
            w = (b - a)
            return (a + b-1)/2, (w*w-1)/12
        elif self.dist_name == "exponential":
            m = self.params[0]
            return m, m*m
        elif self.dist_name == "triangular":
            a, b, c = self.params
            m = (a+b+c)/3
            v = (a*a + b*b + c*c - a*b - a*c - b*c)/18
            return m, v
        elif self.dist_name == "normal":
            return self.params[0], self.params[1]**2
        elif self.dist_name == "gamma":
            return self.params[0], self.params[1]**2
        elif self.dist_name == "exp_gamma":
            alpha, em, gm, gs = self.params
            m = alpha*em + (1-alpha)*gm
            v = alpha*em*em + (1-alpha)*gs*gs + alpha*(1-alpha)*(em-gm)**2
            return m, v
        elif self.dist_name == "beta":
            a, b = self.params
            return beta_mv(a, b)
        elif self.dist_name == "beta4":
            a, b, low, high = self.params
            w = high - low
            m, v = beta_mv(a, b)
            m = low + w*m
            v = w*w*v
            return m, v
        elif self.dist_name == "trunc_normal":
            mu, sig, low, high = self.params
            alpha = (low - mu)/sig
            beta = (high - mu)/sig
            Z = gaussian.cdf(beta) - gaussian.cdf(alpha)
            c = (gaussian.pdf(alpha) - gaussian.pdf(beta))/Z
            d = (beta*gaussian.pdf(beta) - alpha*gaussian.pdf(alpha))/Z
            m = mu + c*sig
            v = (sig*sig)*(1 - d - c*c)
            return m, v
        return 0, 0
        
        

    def test(self):  # test function to check correct implementation
        L = {
            "U01": None,
            "N01": None, 
            "uniform": (2, 5), 
            "unif_int": (1, 10),
            "exponential": 3.4,
            "triangular": (2, 6, 10),
            "normal": (5, 2),
            "gamma": (3, 1.5),
            "exp_gamma": (0.2, 1, 3, 1.5),
            "beta": (2, 3),
            "beta4": (2, 3, 4, 10),
            "trunc_normal": (2.0, 1.0, 1.9, 2.1),
            # "trunc_normal": (5, 2, 3, 8),
            # "choice": (("A", "B", "C"), (0.1, 0.2, 0.7)),
            # "choice": (4, (0.1, 0.2, 0.3, 0.4)),
        }

        n = 100000 # for vector output test  
        # n = None # for single output test
        st = time.time()
        for dist in L.keys():
            params = L[dist]
            if params is None or isinstance(params, (float, int, str, bool)):
                self.set_distribution(dist, params)
            else:
                self.set_distribution(dist, *params)
            x = self.get(size=n)
            m, v = self.get_mean_var()
            et = time.time()
            print (f"{dist}({params}), mean={m:.4f}, var={v:.4f}, s.mean={x.mean():.4f}, s.var={x.var():.4f}, elapsed={et-st:.3f}")
            st = et

    def test_AV(self):  # test function to check correct implementation
        n = 10000
        self.set_distribution("exponential", 1.0)
        self.AT_IAT = False
        x1 = self.get(size=n)
        self.set_seed()
        self.AT_IAT = True
        x2 = self.get(size=n)
        rho = np.corrcoef(x1, x2)
        print (f"antithetic rho = {rho}")


class InputAnalyzer:
    def __init__(self, rng=None):
        if rng is None or isinstance(rng, str):
            rng = RndNumGen(rng).stream
        self.rng = rng
        self.data = None
        self.x_min = 0
        self.x_max = 1

    def set_params(self, params):
        pass

    def __str__(self):
        return f"U(0,1)"

    def get_sample(self, n=1):
        self.data = self.rng.random(n)  # U(0,1)
        return self.data

    def get_pdf(self, x):
        return np.where((x >= 0) & (x <= 1), 1.0, 0.0)

    def get_cdf(self, x):
        return np.clip(x, 0.0, 1.0)

    def get_quantile(self, p):  
        def invert(p_val):
            if p_val <= 0:
                return self.x_min
            if p_val >= 1:
                return self.x_max
            res = root_scalar(lambda x: self.get_cdf(x) - p_val, bracket=bracket, method='brentq')
            return res.root
        
        bracket = (self.x_min, self.x_max)
        if np.isscalar(p):
            return invert(p)
        else:  # use interpolation
            # return np.array([invert(pv) for pv in p])
            grid_x = np.linspace(self.x_min, self.x_max, 1000)
            grid_y = self.get_cdf(grid_x).ravel()
            return np.interp(p, grid_y, grid_x)
        

    def get_mean(self):
        return 0.5

    def get_std(self):
        return np.sqrt(1/12)

    def fit_EM(self, X, max_iter=100, tol=1e-6):
        self.set_data(X)
        return

    @staticmethod
    def _get_ax(ax):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        return ax

    def set_data(self, X):
        self.data = X
        self.x_min = X.min()
        self.x_max = X.max()
        d = (self.x_max - self.x_min)/20
        self.x_min -= d
        self.x_max += d

    def _get_data(self, X):
        if X is None:
            X = self.data
        else:
            self.data = X
        return X
    
    def plot_histo(self, X=None, bins=30, ax=None):
        ax = self._get_ax(ax)
        X = self._get_data(X)
        ax.hist(X, bins=bins, density=True, alpha=0.4, color='gray', edgecolor='black', label='Data Histogram')
        return ax

    def plot_pdf(self, ax=None, color='blue', label='PDF'):
        ax = self._get_ax(ax)
        if self.data is None:
            x_min, x_max = self.x_min, self.x_max
        else:
            x_min, x_max = self.data.min() - 1.0, self.data.max() + 1.0
        x_grid = np.linspace(x_min, x_max, 1000)
        pdf = self.get_pdf(x_grid)
        ax.plot(x_grid, pdf, color=color, linewidth=2, label=label)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, linestyle=':', alpha=0.6)
        return ax, x_grid

    def plot_QQ(self, X=None, ax=None):
        X = self._get_data(X)
        X_sorted = np.sort(X)
        n = len(X_sorted)
        
        prob = (np.arange(1, n + 1) - 0.5) / n        
        theoretical_quantiles = self.get_quantile(prob)
        
        plt.figure(figsize=(6, 6))
        plt.scatter(theoretical_quantiles, X_sorted, alpha=0.7, edgecolors='k', label='Data Quantiles')
        
        # 기준선 (y = x reference line)
        min_val = min(theoretical_quantiles.min(), X_sorted.min())
        max_val = max(theoretical_quantiles.max(), X_sorted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Reference Line (y = x)')
        
        plt.xlabel('Theoretical Quantiles of fitted CDF')
        plt.ylabel('Sample Quantiles (Data)')
        plt.title('QQ Plot')
        plt.legend()
        plt.grid(True)
        plt.show()

    def n_params(self):
        return 0
    
    def chi_squared_test(self, X, bins=20, plot=True):
        bin_probs = np.linspace(0, 1, bins + 1)
        bin_edges = self.get_quantile(bin_probs)
        bin_edges[0] = min(bin_edges[0], self.x_min)
        bin_edges[-1] = max(bin_edges[-1], self.x_max)

        N = len(X)
        counts, _ = np.histogram(X, bins=bin_edges)
        N_bins = N / bins
        expected_counts = np.full(bins, N_bins)

        ddof = self.n_params() - 1   # reduction of dof (ddof in NumPy/SciPy)
        dof = bins - ddof
        chi2_stat, p_value = chisquare(f_obs=counts, f_exp=expected_counts, ddof=ddof)

        print("=== Chi-Squared Goodness-of-Fit Test ===")
        print(f"Chi-square Statistic: {chi2_stat:.4f}")
        print(f"Degrees of Freedom : {dof}")
        alpha = 0.05
        if p_value > alpha:
            print(f"p-value({p_value:.4f}) >= significance level({alpha:.3f}): cannot reject H0 (fitting is good)")
        else:
            print(f"p-value({p_value:.4f}) < significance level({alpha:.3f}): reject H0 (fitting is not good)")

        if plot:
            plt.figure(figsize=(10, 5))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            widths = np.diff(bin_edges)

            # 각 Bin의 관측 밀도/빈도 시각화
            plt.bar(bin_centers, counts, width=widths*0.9, alpha=0.6, color='skyblue', edgecolor='black', label='Observed Count')
            plt.axhline(N_bins, color='red', linestyle='--', linewidth=2, label=f'Expected Count ({N_bins:.0f})')

            plt.xlabel('Value (Bin widths vary based on quantiles)')
            plt.ylabel('Frequency')
            plt.title('Equal-Probability Binning: Observed vs Expected Counts')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()

    def ks_test(self, X, plot=True):
        ks_stat, p_value = kstest(X, self.get_cdf)
        print("=== Kolmogorov-Smirnov Test Results ===")
        print(f"KS Statistic (D) : {ks_stat:.4f}")
        print(f"P-value          : {p_value:.4f}")
        alpha = 0.05
        if p_value > alpha:
            print(f"p-value({p_value:.4f}) >= significance level({alpha:.3f}): cannot reject H0 (fitting is good)")
        else:
            print(f"p-value({p_value:.4f}) < significance level({alpha:.3f}): reject H0 (fitting is not good)")

        if plot:
            sorted_X = np.sort(X)
            ecdf = np.arange(1, len(X) + 1) / len(X)        # empirical CDF
            tcdf = self.get_cdf(sorted_X)                   # theoretical CDF

            diffs = np.abs(ecdf - tcdf)
            max_idx = np.argmax(diffs)
            max_x = sorted_X[max_idx]

            plt.figure(figsize=(9, 5))
            plt.plot(sorted_X, ecdf, label='Empirical CDF (Data)', color='blue', lw=2)
            plt.plot(sorted_X, tcdf, label='Theoretical CDF (GMM Fit)', color='red', linestyle='--', lw=2)

            # show max-diff position
            plt.vlines(x=max_x, ymin=min(ecdf[max_idx], tcdf[max_idx]),
                    ymax=max(ecdf[max_idx], tcdf[max_idx]),
                    color='green', linewidth=2.5, label=f'Max Dist = {ks_stat:.4f} at {max_x:.4f})')

            plt.title('Kolmogorov-Smirnov Test: EDF vs CDF')
            plt.xlabel('x')
            plt.ylabel('Cumulative Probability')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()

            
class GaussianMixture(InputAnalyzer):
    def __init__(self, K, means=None, stds=None, weights=None, rng=None):
        super().__init__(rng)
        self.set_params(K, means, stds, weights)

    def n_params(self):
        return self.K * 3

    def set_params(self, K, means=None, stds=None, weights=None):
        self.K = K
        self.means = np.array(means) if means is not None else np.array([0.0] * K)
        self.stds = np.array(stds) if stds is not None else np.array([1.0] * K)
        self.weights = np.array(weights) if weights is not None else np.array([1.0 / K] * K)

    def __str__(self):
        return f"GaussianMixture(K={self.K}, means={self.means}, stds={self.stds}, weights={self.weights})"

    def get_sample(self, n=1):
        stream = self.rng #.stream
        z = stream.choice(len(self.weights), p=self.weights, size=n)
        X = stream.normal(loc=self.means[z], scale=self.stds[z])
        return X

    def _mix(self, comp):
        weighted = comp * self.weights     
        total = np.sum(weighted, axis=-1)   
        return total, weighted

    def _pdf_cdf(self, X, pdf=True):
        if isinstance(X, np.ndarray) and X.shape[-1] != 1:
            X = np.expand_dims(X, axis=-1)  # X = X[..., np.newaxis]
        if pdf:
            comp = gaussian.pdf(X, loc=self.means, scale=self.stds)  # shape: (N, K)
        else:
            comp = gaussian.cdf(X, loc=self.means, scale=self.stds)  # shape: (N, K)
        weighted = comp * self.weights     # shape: (N, K)
        total = np.sum(weighted, axis=-1)   # shape: (N,) 
        return total, weighted, comp

    def get_pdf(self, X):
        total, _, _ = self._pdf_cdf(X, True)
        return total 

    def get_cdf(self, X):
        total, _, _ = self._pdf_cdf(X, False)
        return total 

    def get_mean(self):
        return self._mix(self.means)

    def get_std(self):
        v1 = self._mix(self.stds ** 2)
        mu = self.get_mean()
        v2 = self._mix((self.means - mu)**2)
        return np.sqrt(v1 + v2)
    
    def _init_params_random_sample(self, X):
        K = self.K
        indices = self.rng.choice(len(X), size=K, replace=False)
        self.means = X[indices].copy()
        overall_std = np.std(X)
        self.stds = np.full(K, overall_std if overall_std > 1e-3 else 1.0)
        self.weights = np.full(K, 1.0 / K)

    def fit_EM(self, X, max_iter=100, tol=1e-6):
        self.set_data(X)
        N = len(X)
        K = self.K
        prev_log_likelihood = None
        self._init_params_random_sample(X)
        X = X.reshape((-1,1))

        for iteration in range(max_iter):
            # ----------------------------------------------------
            # 1. E-step (Expectation): compute r (responsibility) of shape (N, K)
            # ----------------------------------------------------
            _, weighted_pdfs, _ = self._pdf_cdf(X, True)
            likelihood_per_sample = np.sum(weighted_pdfs, axis=1, keepdims=True)  # shape: (N, 1)
            likelihood_per_sample = np.maximum(likelihood_per_sample, 1e-12)
            r = weighted_pdfs / likelihood_per_sample  # shape: (N, K)

            # ----------------------------------------------------
            # Convergence Check: Log-Likelihood 계산
            # ----------------------------------------------------
            log_likelihood = np.sum(np.log(likelihood_per_sample))
            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood

            # ----------------------------------------------------
            # 2. M-step (Maximization): update parameters (means, stds, weights)
            # ----------------------------------------------------
            N_k = np.sum(r, axis=0)  # effective sample size for component k, shape: (K,)
            N_k = np.maximum(N_k, 1e-8)
            self.means = ((X.T @ r) / N_k).reshape(-1)
            diff = X - self.means.reshape(1, K)
            variance = np.sum(r * (diff ** 2), axis=0) / N_k
            self.stds = np.sqrt(np.maximum(variance, 1e-6)).reshape(-1)         
            self.weights = N_k / N
        print (f"EM converged in {iteration + 1} iterations, log-L = {log_likelihood:.6f}, pref log-L = {prev_log_likelihood:.6f}")


    def plot_pdf(self, ax=None, color='blue', label='PDF'):
        ax, x_grid = super().plot_pdf(ax=ax, color=color, label=label)
        _, comp_pdf, _ = self._pdf_cdf(x_grid, True)
        for k in range(self.K):
            comp_label=f"{label}_comp{k+1} (mu={self.means[k]:.2f}, sigma={self.stds[k]:.2f}, pi={self.weights[k]:.2f})"
            ax.plot(x_grid, comp_pdf[:,k], '--', linewidth=1, label=comp_label) 
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, linestyle=':', alpha=0.6)




class ExpGaussianMixture(InputAnalyzer):
    def __init__(self, exp_mean, mu, sigma, pi, rng=None):
        super().__init__(rng)
        self.set_params(exp_mean, mu, sigma, pi)

    def n_params(self):
        return 4

    def set_params(self, exp_mean, mu, sigma, pi):
        self.exp_mean = exp_mean
        self.mu = mu
        self.sigma = sigma
        self.pi = pi
        
    def __str__(self):
        return f"ExpGaussianMixture(exp_mean={self.exp_mean:.3f}, mu={self.mu:.3f}, sigma={self.sigma:.3f}, pi={self.pi:.3f})"

    def get_sample(self, n=1):
        stream = self.rng
        n_gaussian = stream.binomial(n, p=self.pi)
        n_exp = n - n_gaussian
        
        e_samples = stream.exponential(scale=self.exp_mean, size=n_exp)
        g_samples = stream.normal(loc=self.mu, scale=self.sigma, size=n_gaussian)
        
        # combine and shuffle
        samples = np.concatenate([e_samples, g_samples])
        np.random.shuffle(samples)
        return samples

    def _mix(self, e_comp, g_comp):
        return (1-self.pi)*e_comp + self.pi*g_comp

    def _pdf_cdf(self, X, pdf=True):
        if isinstance(X, np.ndarray) and X.shape[-1] != 1:
            X = np.expand_dims(X, axis=-1)  # X = X[..., np.newaxis]
        if pdf:
            e_comp = expon.pdf(X, scale=self.exp_mean)
            g_comp = gaussian.pdf(X, loc=self.mu, scale=self.sigma)
        else:
            e_comp = expon.cdf(X, scale=self.exp_mean)
            g_comp = gaussian.cdf(X, loc=self.mu, scale=self.sigma)
        total = self._mix(e_comp, g_comp)
        return total, e_comp, g_comp

    def get_pdf(self, X):
        total, _, _ = self._pdf_cdf(X, pdf=True)
        total = total.ravel()
        return total 

    def get_cdf(self, X):
        total, _, _ = self._pdf_cdf(X, pdf=False)
        total = total.ravel()
        return total 

    def get_mean(self):
        return self._mix(self.exp_mean, self.mu)

    def get_std(self):
        v1 = self.mix(self.exp_mean**2, self.std**2)
        v2 = self.pi*(1-self.pi)*(self.exp_mean - self.mu)**2
        return np.sqrt(v1 + v2)
    
    def _init_params_random_sample(self, X):
        pass


    def fit_EM(self, X, max_iter=100, tol=1e-6):
        self.set_data(X)
        N = len(X)
        prev_log_likelihood = None
        self._init_params_random_sample(X)
        X = X.reshape((-1,1))

        for iteration in range(max_iter):
            # ----------------------------------------------------
            # 1. E-step (Expectation): compute r (responsibility) of shape (N, K)
            # ----------------------------------------------------
            pdf, pdf_e, pdf_g = self._pdf_cdf(X, pdf=True)            
            pdf = np.maximum(pdf, 1e-12)
            r_g = (self.pi * pdf_g) / pdf
            r_e = 1 - r_g

            # ----------------------------------------------------
            # Convergence Check: Log-Likelihood
            # ----------------------------------------------------
            log_likelihood = np.sum(np.log(pdf))
            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood

            # ----------------------------------------------------
            # 2. M-step (Maximization): update parameters (means, stds, weights)
            # ----------------------------------------------------
            N_e = np.sum(r_e)    
            N_g = np.sum(r_g)
            self.pi = N_g / N
            
            # update Exponential mean
            if N_e > 1e-6:
                self.exp_mean = np.sum(r_e * X) / N_e
            
            # update Gaussian mu 및 sigma
            if N_g > 1e-6:
                self.mu = np.sum(r_g * X) / N_g
                var_g = np.sum(r_g * (X - self.mu) ** 2) / N_g
                self.sigma = np.sqrt(np.maximum(var_g, 1e-6))
        print (f"EM converged in {iteration + 1} iterations, log-L = {log_likelihood:.6f}, pref log-L = {prev_log_likelihood:.6f}")


    def plot_pdf(self, ax=None, color='blue', label='PDF'):
        ax, x_grid = super().plot_pdf(ax=ax, color=color, label=label)
        _, pdf_e, pdf_g = self._pdf_cdf(x_grid, True)
        comp_label=f"{label}_exp (mean={self.exp_mean:.2f}, weight={1-self.pi:.2f})"
        ax.plot(x_grid, (1-self.pi)*pdf_e, '--', linewidth=1, label=comp_label) 
        comp_label=f"{label}_gaussian (mean={self.mu:.2f}, sigma={self.sigma:.2f}, weight={self.pi:.2f})"
        ax.plot(x_grid, self.pi*pdf_g, '--', linewidth=1, label=comp_label) 
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, linestyle=':', alpha=0.6)


class AdvSampler:
    def __init__(self, p, q, rng=None):
        # p, q: distribution class in scipi.stats 
        # q.rvs(): get samples
        # p.logpdf(x), q.logpdf(x): get log of pdf
        self.rng = rng
        self.p = p  # target
        self.q = q  # proposal

    def prep_rejection_sampling(self, log_M):
        self.log_M = log_M

    def rejection_sampling(self, n_samples):
        accepted = []
        n_trial = 0
        while len(accepted) < n_samples:
            x = self.q.rvs(random_state=self.rng)  # sample beta from proposal q
            log_p = self.p.logpdf(x)
            log_q = self.q.logpdf(x)
            log_alpha = log_p - log_q - self.log_M  # log acceptance probability
            if log_alpha > 1e-8:        # Envelope check
                raise RuntimeError(f"Envelope violated: log_alpha={log_alpha:.3e}")

            n_trial += 1
            u = self.rng.random()
            if np.log(u) < log_alpha:
                if n_samples == 1:
                    return x, n_trial
                accepted.append(x)
            # if n_trial % 100 == 0:
            #     print (len(accepted))
        samples = np.asarray(accepted)
        return samples, n_trial


    def importance_sampling(self, n_samples):
        x = self.q.rvs(size=n_samples, random_state=self.rng)
        log_p = self.p.logpdf(x)
        log_q = self.q.logpdf(x)
        log_w = log_p - log_q
        log_w -= logsumexp(log_w)  # normalized weights
        w = np.exp(log_w) * n_samples
        # print(f"effective sample size = {self.ESS_IS(w):.3f}")    
        return x, w      

    @staticmethod
    def ESS_IS(w):
        return (w.sum()**2) / (w*w).sum() / len(w)


    @staticmethod
    def ESS_MCMC(x):
        def autocorr(x, lag):
            x = np.asarray(x)
            x = x - x.mean(axis=0)
            xx = (x * x).sum(axis=0)
            xx_lag = (x[:-lag] * x[lag:]).sum(axis=0)
            # xx = (x * x).mean(axis=0)
            # xx_lag = (x[:-lag] * x[lag:]).mean(axis=0)
            rho = xx_lag / xx
            return rho

        print ("ESS")
        x = np.asarray(x)
        N, dim = x.shape

        max_lag = min(N//10, 500)
        rhoL = np.zeros((max_lag, dim))
        for lag in range(max_lag):
            rhoL[lag] = autocorr(x, lag+1)

        ess = np.zeros(dim)
        for c in range(dim):
            # P0 = rho_0 + rho_1 = 1 + rho_1
            pair_sum = 1.0 + rhoL[0, c]
            if pair_sum <= 0:
                ess[c] = N
                continue
            P_sum = pair_sum
            # P1 = rho_2 + rho_3
            # P2 = rho_4 + rho_5
            # ...
            for k in range(1, len(rhoL) - 1, 2):
                pair_sum = rhoL[k, c] + rhoL[k + 1, c]
                if pair_sum <= 0:
                    break
                P_sum += pair_sum
            tau = 2 * P_sum - 1
            ess[c] = N / tau
            print(f"component={c}, stop lag={k}, tau={tau:.2f}, ESS={ess[c]:.2f}")
        return ess


    def prep_MCMC(self, x0, burn_in, RW=True, step_size=1.0):
        self.x_cur = x0
        self.dim = len(x0)
        self.logp = self.p.logpdf(x0)
        self.RW = RW
        if RW: # random walk
            self.logq = 0.0
            self.step_size = step_size
        else:
            self.logq = self.q.logpdf(x0)
        self.MCMC(burn_in)  

    def check_accept(self, x_new, logp_new, logq_new=0.0):
        if self.RW:  # symmetric random walk sampler
            log_alpha = logp_new - self.logp
        else: # independent sampler
            log_alpha = logp_new - logq_new - self.logp + self.logq

        u = self.rng.random()
        if np.log(u) < log_alpha:
            self.x_cur = x_new
            self.logp = logp_new
            self.logq = logq_new
            return True
        return False

    def MCMC(self, n_sample):
        if n_sample > 1:
            samples = np.empty((n_sample, self.dim))
            n_accept = 0
        logq_new = 0.0
        for k in range(n_sample):
            x_new = self.q.rvs(random_state=self.rng)  # sample beta from proposal q
            if self.RW:
                x_new = self.x_cur + self.step_size * x_new
            else:
                logq_new = self.q.logpdf(x_new)
            logp_new = self.p.logpdf(x_new)
            accept = self.check_accept(x_new, logp_new, logq_new)
            if n_sample == 1:  # single sample
                return self.x_cur
            if accept: 
                n_accept += 1
            samples[k] = self.x_cur
            if (k+1) % 1000 == 0:
                print(f"iter={k+1}, n_acccept={n_accept}")
        return samples, n_accept     

    # def prep_pymc(self, pm_model):
    #     self.pm_model = pm_model

    # @staticmethod
    # def MCMC_pymc(pm_model, n_sample, tune=1000):
    #     n_chain = 4
    #     n_per_chain = n_sample // n_chain
    #     with pm_model:
    #         trace = pm.sample(draws=n_per_chain, tune=tune, chains=n_chain, target_accept=0.9)
    #     # s = az.summary(trace)
    #     # print (f"ESS bulk={s['ess_bulk'].mean()/n_sample:.3f}, tail={s['ess_tail'].mean()/n_sample:.3f}")
    #     # print (s)
    #     # print ("ESS\n", az.ess(trace))
    #     # az.plot_trace(trace, var_names=["beta"])
    #     # plt.show()

    #     samples = trace.posterior["beta"].values
    #     dim = samples.shape[-1]
    #     samples = samples.reshape((-1, dim))
    #     return samples




class BayesModel:
    def log_likelihood(self, beta):
        pass  # to be redefined

    # precision matrix estimated from data
    def data_precision(self, beta_map):
        pass  # to be redefined

    # precision matrix if prior
    def prior_precision(self):
        pass

    def __init__(self, prior):
        self.prior = prior
        self.log_post = self.logpdf

    @staticmethod
    def make_Gaussian_prior(dim, mean=0, tau=1):
        mean = np.zeros(dim) + mean
        cov = tau*tau * np.eye(dim)
        prior = multivariate_normal(mean=mean, cov=cov)
        return prior, mean, cov

    def make_proposal(self, q_option="default"):
        beta_map, cov_post = self.get_MAP_Cov()
        if q_option == "default":
            q = self.prior
        else:
            if q_option == "norm_post":
                q = multivariate_normal(mean=beta_map, cov=cov_post)
            else: # option == "t-post"
                df, scale = 5, 1
                # shape = scale**2 * (df - 2) / df * cov_post
                shape = scale**2 * cov_post
                q = multivariate_t(loc=beta_map, shape=shape, df=df)
        return q, beta_map, cov_post

    def logpdf(self, beta):
        return self.log_likelihood(beta) + self.prior.logpdf(beta)

    def get_MAP_Cov(self):
        result = minimize(  # find MAP
            fun=lambda beta: -self.logpdf(beta),
            x0=np.zeros(self.dim),
            method="BFGS"
        )
        if not result.success:
            print("Warning:", result.message)
        beta_map = result.x

        # Laplace approx. of covariance of log posterior
        H = self.data_precision(beta_map)
        H += self.prior_precision()

        cov_post = np.linalg.inv(H)
        return beta_map, cov_post
    

    def get_log_M(self, q, beta_map, n_start=100):
        # starting points for multi-start
        starts = [beta_map]
        for _ in range(n_start - 1):
            starts.append(q.rvs())

        best_log_M = -np.inf
        best_beta = None

        for x0 in starts:
            result = minimize(
                lambda beta: q.logpdf(beta) - self.logpdf(beta),
                x0,
                method="BFGS"
            )

            value = self.logpdf(result.x) - q.logpdf(result.x)
            if value > best_log_M:
                best_log_M = value
                best_beta = result.x

        # small numerical margin
        log_M = best_log_M + 1e-4
        print("log_M =", log_M)
        print("beta_max =", best_beta)
        return log_M


class BayesLogReg(BayesModel):
    @staticmethod
    def make_logreg_data(rng, dim=20, n=1000):
        beta_true = np.linspace(1, 2, dim)
        X = rng.normal(size=(n, dim))
        z = beta_true @ X.T         # logit
        p = expit(z)                # sigmoid
        y = rng.binomial(1, p)      # class label from Bernoulli
        return X, y

    def __init__(self, dim, tau_prior, X, y):
        self.dim = dim
        self.prior, self.prior_mean, self.prior_cov = self.make_Gaussian_prior(self.dim, tau=tau_prior)
        super().__init__(self.prior)
        self.X, self.y = X, y
        self.dim = X.shape[1]
        self.tau = tau_prior
        self.prior_prec = np.linalg.inv(self.prior_cov)



    def logit(self, beta):
        return beta @ self.X.T

    def log_likelihood(self, beta):
        z = self.logit(beta)
        return np.sum(self.y * z - np.logaddexp(0, z), axis=-1)

    # precision matrix estimated from data
    def data_precision(self, beta_map):
        z = self.logit(beta_map)
        p = expit(z)    # sigmoid --> prob.
        w = p * (1 - p) # 
        H = self.X.T @ (w[:, None] * self.X)
        return H

    # precision matrix if prior
    def prior_precision(self):
        return self.prior_prec

    def make_pymc_model(self):
        pm_model = pm.Model()
        with pm_model:
            beta = pm.Normal("beta", mu=0, sigma=self.tau, shape=self.dim)  # prior
            p = pm.math.sigmoid(self.X @ beta)
            y_obs = pm.Bernoulli("y_obs", p=p, observed=self.y)  # observation likelihood
        return pm_model



class BayesGMM (GaussianMixture):

    def __init__(self, rng):
        means = [1, 5]
        stds = [1, 1.5]
        weights = [0.3, 0.7]
        super().__init__(2, means, stds, weights, rng)

        self.rng = rng
        self.n_data = 1000
        self.dim = 5   # pi, mu0, mu1, sig0, sig1
        self.y = self.get_sample(self.n_data)
        self.a_pr, self.b_pr = 1, 1
        self.m_pr = np.array([0, 0])
        self.V_pr = np.array([1, 1])
        self.alpha_pr = np.array([1, 1])
        self.beta_pr = np.array([1, 1])
        # parameter & latent variables
        self.pi = 0.5
        self.mu = np.zeros(self.K)
        self.sigma2 = np.ones(self.K)
        self.z = np.zeros(self.n_data, dtype=int)

    def nn(self, k):
        return (self.z == k).sum()

    def yz(self, k):
        return self.y[self.z == k]

    def compute_p(self):
        p0 = (1-self.pi) * gaussian.pdf(self.y, self.mu[0], np.sqrt(self.sigma2[0]))
        p1 = self.pi * gaussian.pdf(self.y, self.mu[1], np.sqrt(self.sigma2[1]))
        p_sum = p0 + p1
        return [p0/p_sum, p1/p_sum]

    def fc_sample_z(self):
        p = self.compute_p()
        self.z = self.rng.binomial(1, p[1])

    def fc_sample_pi(self):
        n0 = self.nn(0)
        n1 = self.nn(1)
        self.pi = self.rng.beta(self.a_pr + n1, self.b_pr + n0)

    def fc_sample_mu(self, k):
        V_fc = 1 / (1/self.V_pr[k] + self.nn(k)/self.sigma2[k])
        yz = self.yz(k)
        m_fc = V_fc * (self.m_pr[k]/self.V_pr[k] + yz.sum()/self.sigma2[k])
        self.mu[k] = self.rng.normal(m_fc, np.sqrt(V_fc))

    def fc_sample_sigma2(self, k):
        alpha_fc = self.alpha_pr[k] + self.nn(k) / 2
        yz = self.yz(k)
        beta_fc = self.beta_pr[k] + np.sum((yz - self.mu[k])**2) / 2
        self.sigma2[k] = invgamma.rvs(a=alpha_fc, scale=beta_fc, random_state=self.rng)

    def param_vector(self):
        # v = np.concatenate([self.pi, self.mu, self.sigma2])
        v = np.hstack([self.pi, self.mu, self.sigma2])
        return v

    def Gibbs(self, n_sample):
        if n_sample > 1:
            samples = np.empty((n_sample, self.dim))

        for k in range(n_sample):
            self.fc_sample_z()
            self.fc_sample_pi()
            self.fc_sample_mu(0)
            self.fc_sample_sigma2(0)
            self.fc_sample_mu(1)
            self.fc_sample_sigma2(1)
            v = self.param_vector()
            if n_sample == 1:  # single sample
                return v
            samples[k] = v
        return samples


    
def get_rnd_gen(sn):
    if not RndNumGen.is_initialized():
        RndNumGen.init_rnd_generators(42)
    else:
        RndNumGen.reset_seed_all()
    rng = RndNumGen(sn)
    return rng


def test_adv_sampling():
    RndNumGen.set_master_seed(42)
    rng = RndNumGen().stream

    n_data = 1000
    n_dim = 5
    X, y = BayesLogReg.make_logreg_data(rng, dim=n_dim, n=n_data)
    tau = 2.0
    bayes_model = BayesLogReg(n_dim, tau_prior=tau, X=X, y=y)

    # method = "RS"
    # method = "IS"
    method = "MCMC"
    # method = "PyMC"
    # option = "default"
    # option = "norm_post"
    option = "t_post"

    proposal, beta_map, _ = bayes_model.make_proposal(q_option=option)
    sampler = AdvSampler(bayes_model, proposal, rng)

    st = time.time()
    n_sample = 10000
    if method == "RS":
        log_M = bayes_model.get_log_M(proposal, beta_map)
        sampler.prep_rejection_sampling(log_M)
        samples, n_trial = sampler.rejection_sampling(n_sample)
        print("Posterior mean   :", samples.mean(axis=0))
        print(f"acceptance rate = {n_sample/n_trial:.3f}")
    elif method == "IS":
        samples, w = sampler.importance_sampling(n_sample)
        print("Posterior mean   :", (samples*w.reshape((-1,1))).mean(axis=0))
        print(f"effective sample size = {sampler.ESS_IS(w):.3e}")    
    elif method == "MCMC":
        RW = (option == "default")
        sampler.prep_MCMC(beta_map, burn_in=1000, RW=RW, step_size=0.05)
        samples, n_accept = sampler.MCMC(n_sample)
        # samples = MCMC_pymc(tau, X, y, n_samples)
        print("Posterior mean   :", samples.mean(axis=0))
        ess = sampler.ESS_MCMC(samples)
        print (f"MCMC accept={n_accept}, ESS_mean={ess.mean():.3f}, ESS_min={ess.min():.3f}, argmin={ess.argmin()}")    
    elif method == "PyMC":
        pm_model = bayes_model.make_pymc_model()
        samples = sampler.MCMC_pymc(pm_model, n_sample)
        ess = sampler.ESS_MCMC(samples)
        print (f"MCMC ESS_mean={ess.mean():.3f}, ESS_min={ess.min():.3f}, argmin={ess.argmin()}")    

    et = time.time()
    print (f"time for {len(samples)} samples = {et-st:.3f}")


def test_Gibbs():
    RndNumGen.set_master_seed(42)
    rng = RndNumGen().stream

    gmm = BayesGMM(rng)

    st = time.time()
    samples = gmm.Gibbs(1000)  # burn-in
    n_sample = 10000
    samples = gmm.Gibbs(n_sample)
    print("Posterior mean   :", samples.mean(axis=0))
    ess = AdvSampler.ESS_MCMC(samples)
    print (f"Gibbs, ESS_mean={ess.mean():.3f}, ESS_min={ess.min():.3f}")    
    et = time.time()
    print (f"time for {len(samples)} samples = {et-st:.3f}")

    gmm.fit_EM(gmm.y, max_iter=1000, tol=1e-6)
    print (gmm)


if __name__ == "__main__":
    # test_adv_sampling()
    test_Gibbs()
    exit(0)

    def plot_pdf_cdf():
        x_min, x_max = 0, 5
        mu = 1
        x_grid = np.linspace(x_min, x_max, 1000)
        pdf = expon.pdf(x_grid, scale=mu)
        plt.plot(x_grid, pdf, color='blue', linewidth=2, label="pdf")
        cdf = expon.cdf(x_grid, scale=mu)
        plt.plot(x_grid, cdf, color='red', linewidth=2, label="cdf")
        plt.legend(fontsize=10, loc='best')
        plt.show()

    def goodness_of_fit(m_fit, X, ax):
        if ax is not None:
            m_fit.plot_pdf(ax, color='blue', label='Fitted PDF')
            m_fit.plot_QQ(X)      
        m_fit.chi_squared_test(X)
        m_fit.ks_test(X)           

    def fit_GMM(X, K, ax=None):
        gmm_fit = GaussianMixture(K)
        gmm_fit.fit_EM(X, max_iter=1000, tol=1e-6)
        print (gmm_fit)
        goodness_of_fit(gmm_fit, X, ax)
        return gmm_fit
    
    def test_GMM():
        rng = get_rnd_gen("GM").stream
        weights = [0.2, 0.5, 0.3]        # 각 성분별 확률 (합 = 1.0)
        means = [-5.0, 0.0, 5.0]         # 각 성분의 평균
        stds = [0.5, 1.0, 1.5]           # 각 성분의 표준편차    
        gmm_true = GaussianMixture(3, means=means, stds=stds, weights=weights, rng=rng)
        print (gmm_true)
        X = gmm_true.get_sample(1000)
        ax = gmm_true.plot_histo(X, bins=50)
        gmm_true.plot_pdf(ax, color='red', label='True PDF')
        gmm_fit = fit_GMM(X, 3, ax)
        plt.show()


    def fit_EGM(X, ax=None):
        egm_fit = ExpGaussianMixture(0.1, 0, 1, 0.5)
        egm_fit.fit(X, max_iter=1000, tol=1e-6)
        print (egm_fit)
        goodness_of_fit(egm_fit, X, ax)
        return egm_fit

    def test_EGM():
        rng = get_rnd_gen("EG")
        exp_mean = 0.3
        mu, sigma = 3.0, 1.0
        pi = 0.6
        egm_true = ExpGaussianMixture(exp_mean, mu, sigma, pi, rng=rng)
        print (egm_true)
        X = egm_true.get_sample(1000)
        ax = egm_true.plot_histo(X, bins=50)
        # egm_true.plot_pdf(ax, color='red', label='True PDF')
        # gmm_fit = fit_GMM(X, 3, ax)
        egm_fit = fit_EGM(X, ax)
        plt.show()

    # plot_pdf_cdf()
    test_GMM()
    # test_EGM()
