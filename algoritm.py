import numpy as np

class RFF:
    """
    Random Fourier features for Gaussian kernel

    Approximates
        k(x,y) = exp(-gamma*||x-y||^2)

    by an explicit random feature map z(x) s.t.
        z(x)^T z(y) = k(x,y)

    """

    def __init__(self, D, gamma, random_state = None):
        """
        Parameters:
            D = int
                # of sampled fourier frequencies omega_j

            gamma = float
                Bandwidth parameter of the Gaussian kernel

                k(x,y) = exp(-gamma*||x-y||^2)
            
            random_state = int or None
                Reproducibility.
        """
        self.D = D
        self.gamma = gamma
        self.random_state = random_state

        self.omega = None

    def fit(self, X):
        """
        Sample the RFF

        If X has shape (N, d) then each omega_j must belong to R^d

        For Gausian kernel
            k(x,y) = exp(-gamma*||x-y||^2)
        
        corresponding fourier dist. is Gaussian
            omega ~ N(0, 2*gamma*I_d)
        """

        rng = np.random.default_rng(self.random_state)

        d = X.shape[1] # number of input dimensions

        # Draw D random frequency vectors omega_j in R^d
        self.omega = rng.normal(loc = 0.0, scale = np.sqrt(2*self.gamma), size = (self.D, d))

        return self

    def transform(self, X):
        """
        Apply the Random fourier feature map

        Algoritm uses
            z(x) = 1/sqrt(D) (cos(omega_1) x, ..., cos(omega_D) x, sin(omega_1), ..., sin(omega_D) X)
        
        For N observations, returned Z matrix has shape 
            (N, 2D)
        """

        if self.omega is None:
            raise ValueError("Model must be fitted before transform")

        proj = X @ self.omega.T # compute omega_j^T x_i for each observation i and freq. j.

        cos_features = np.cos(proj)
        sin_features = np.sin(proj)

        Z = np.concatenate([cos_features, sin_features], axis = 1)

        Z /= np.sqrt(self.D) # Normalize

        return Z

    def fit_transform(self, X):
        """
        sample omega and transforms X
        """
        self.fit(X)
        return self.transform(X)