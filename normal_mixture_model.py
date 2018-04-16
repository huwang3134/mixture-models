import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from netflix_loader import *

class NormalMixtureModel:
    def __init__(self, data, num_components=2, mu_inits=None, sigmasq_inits=None, alpha_inits=None, max_val=5):
        self.Y = np.array(data)
        self.mu = mu_inits
        if mu_inits is None:
            self.mu = []
            for _ in range(num_components):
                self.mu.append(0)
        self.sigmasq = sigmasq_inits
        if sigmasq_inits is None:
            self.sigmasq = []
            for _ in range(num_components):
                self.sigmasq.append(1)
        if alpha_inits is None:
            self.alpha = []
            for _ in range(num_components):
                self.alpha.append(float(1)/num_components)
        self.num_components = num_components
        self.max_val = max_val

    def _compute_I(self, mu, sigmasq, alpha):
        I = np.zeros((len(self.Y), self.num_components))
        # while np.min(np.sum(I, axis=0)) <= 1:
        component_scores = np.zeros((len(self.Y), self.num_components))
        for j in range(self.num_components):
            component_scores[:,j] = alpha[j]*norm.pdf(self.Y, loc=mu[j], scale=np.sqrt(sigmasq[j]))
        component_scores = np.transpose(np.transpose(component_scores)/np.sum(component_scores, axis=1))
        I = np.array([np.random.multinomial(1, components, 1)[0] for components in component_scores])
        return I

    def gibbs_sample(self, chain_length=200):
        # mu = deepcopy(self.mu)
        # mu0 = np.random.uniform()*self.max_val
        # mu = np.array([mu0, np.random.uniform(mu0, self.max_val, 1)])
        mu = np.array([1, 5]);
        sigmasq = deepcopy(self.sigmasq)
        alpha = deepcopy(self.alpha)
        mu_chains = []
        mu_chains.append(np.array(mu))
        sigmasq_chains = []
        sigmasq_chains.append(np.array(sigmasq))
        alpha_chains = []
        alpha_chains.append(np.array(alpha))

        nu0 = np.ones((self.num_components))
        sigmasq0 = np.ones((self.num_components))
        k0 = np.ones((self.num_components))*0.1
        mu0 = np.ones((self.num_components))*2.5

        I = self._compute_I(mu, sigmasq, alpha)
        for i in range(chain_length):
            print(i, 'mu', mu_chains[-1])
            print(i, 'sigmasq', sigmasq_chains[-1])
            print(i, 'alpha', alpha_chains[-1])
            mu_chains.append(np.zeros((self.num_components)))
            sigmasq_chains.append(np.zeros((self.num_components)))
            alpha_chains.append(np.zeros((self.num_components)))
            y_lens = []
            for j in range(self.num_components):
                y_j = []
                for k, y in enumerate(self.Y):
                    if I[k][j] == 1:
                        y_j.append(y)
                y_j = np.array(y_j)
                mu_j_prev = mu_chains[-2][j]
                muj_stacked = np.array([mu_j_prev]*len(y_j))
                s_squared = nu0[j]*sigmasq0[j]+np.sum((y_j-muj_stacked)**2)
                sigmasq_j = 1.0/np.random.gamma(shape=0.5*(len(y_j)+nu0[j]), scale=1.0/(0.5*s_squared), size=1)
                # sigmasq_j = 1/np.random.gamma(shape=0.5*(len(y_j)+2-1), scale=1.0/(0.5*(len(y_j)+2-1)*np.std(y_j, ddof=1)**2), size=1)
                mean_mu_j = (k0[j]*mu0[j]+len(y_j)*np.mean(y_j))/(k0[j]+len(y_j))
                stdev_mu_j = np.sqrt(sigmasq_j/(k0[j]+len(y_j)))
                mu_j = np.random.normal(loc=mean_mu_j, scale=stdev_mu_j, size=1)
                y_lens.append(len(y_j))
                mu_chains[-1][j] = mu_j
                sigmasq_chains[-1][j] = sigmasq_j
            alpha_chains[-1] = np.random.dirichlet(alpha=np.array(y_lens)+1, size=1)[0]
            I = self._compute_I(mu_chains[-1], sigmasq_chains[-1], alpha_chains[-1])
        return {'mu': mu_chains, 'sigmasq': sigmasq_chains, 'alpha': alpha_chains}

    def _gamma_gradient(self, mu, rho, gamma, I):
        n = np.shape(I)[0]
        return (np.sum(I[:,0])+1)-(n+2)*np.exp(gamma)/(1+np.exp(gamma))

    def _mu_gradient(self, mu, rho, gamma, I):
        res = np.zeros((self.num_components))
        mu_stacked = np.vstack([self.mu]*np.shape(I)[0])
        y_stacked = np.transpose(np.vstack([self.Y, self.Y]))
        mat = np.dot(np.transpose(I), y_stacked-mu_stacked)/np.exp(rho)
        return np.array([mat[0][0], mat[1][1]])

    def _rho_gradient(self, mu, rho, gamma, I):
        mu_stacked = np.vstack([self.mu]*np.shape(I)[0])
        y_stacked = np.transpose(np.vstack([self.Y, self.Y]))
        squares_stacked = (y_stacked-mu_stacked)**2
        mat = np.dot(np.transpose(I), squares_stacked)
        return -0.5*np.sum(I, axis=0)+np.array([mat[0][0], mat[1][1]])/(2*np.exp(rho))-1

    def _log_posterior(self, mu, sigmasq, alpha, I):
        res = 0.0
        print('log args', mu, sigmasq, alpha, np.sum(I[:,0]), np.sum(I[:,1]))
        for j in range(self.num_components):
            res += np.log((1.0/sigmasq[j])**(0.5*np.sum(I[:,j])))
            res -= 0.5*np.dot(I[:,j], (self.Y-mu[j])**2)/sigmasq[j]
        print('log pre alpha', res)
        for j in range(self.num_components-1):
            res += np.sum(I[:,j])*np.log(alpha[j])
        res += np.sum(I[:,-1])*np.log((1-np.sum(alpha)))
        print('log', res)
        return res

    def _log_posterior2(self, mu, rho, gamma, I):
        gamma_terms = -(np.shape(I)[0]+2)*np.log(1+np.exp(gamma))+gamma*(1+np.sum(I[:,0]))
        rho_terms = -0.5*np.sum(np.dot(rho, np.transpose(I)))
        mu_stacked = np.vstack([mu]*np.shape(I)[0])
        y_stacked = np.transpose(np.vstack([self.Y, self.Y]))
        squares_stacked = (y_stacked-mu_stacked)**2
        mu_terms_mat = np.dot(np.transpose(I), squares_stacked)
        mu_terms = mu_terms_mat[0][0]/(-2*np.exp(rho[0]))+mu_terms_mat[1][1]/(-2*np.exp(rho[1]))
        print('res', gamma_terms+rho_terms+mu_terms)
        return gamma_terms+rho_terms+mu_terms-np.sum(rho)

    def check_valid(self, mu, sigmasq, alpha):
        for j in range(self.num_components):
            if j < self.num_components-1:
                if alpha[j] < 0 or alpha[j] > 1:
                    return False
                if sigmasq[j] < 0:
                    return False
        return True

    def hmc(self, chain_length=1000, L=5, epsilon=0.0001):
        num_components = self.num_components
        M_mu = np.eye(num_components)*0.1
        M_rho = np.eye(num_components)*0.1
        M_gamma = 1.0
        phi_mu = None
        phi_rho = None
        phi_gamma = None
        chains = {'mu': [np.array([4, 3])], 'rho': [np.ones((num_components))], 'gamma': [0.5]}
        I = self._compute_I(chains['mu'][-1], np.exp(chains['rho'][-1]), [np.exp(chains['gamma'][-1])/(1+np.exp(chains['gamma'][-1])), 1.0/(1+np.exp(chains['gamma'][-1]))])
        for i in range(1, chain_length):
            phi_mu = np.random.multivariate_normal(mean=np.zeros((num_components)), cov=M_mu)
            phi_rho = np.random.multivariate_normal(mean=np.zeros((num_components)), cov=M_rho)
            phi_gamma = np.random.normal(loc=0, scale=np.sqrt(M_gamma))
            phi_mu_init = deepcopy(phi_mu)
            phi_rho_init = deepcopy(phi_rho)
            phi_gamma_init = phi_gamma
            mu = deepcopy(chains['mu'][-1])
            rho = deepcopy(chains['rho'][-1])
            gamma = chains['gamma'][-1]
            print(i, 'mu', mu)
            print(i, 'rho', rho)
            print(i, 'gamma', gamma)
            for _ in range(L):
                phi_mu = phi_mu + 0.5*epsilon*self._mu_gradient(mu, rho, gamma, I)
                phi_rho = phi_rho+0.5*epsilon*self._rho_gradient(mu, rho, gamma, I)
                phi_gamma = phi_gamma+0.5*epsilon*self._gamma_gradient(mu, rho, gamma, I)
                mu = mu+epsilon*np.dot(np.linalg.inv(M_mu), phi_mu)
                rho = rho+epsilon*np.dot(np.linalg.inv(M_rho), phi_rho)
                gamma = gamma+epsilon*phi_gamma/M_gamma
                """if not self.check_valid(mu, sigmasq, alpha):
                    phi_mu = -1*phi_mu
                    phi_sigmasq = -1*phi_sigmasq
                    phi_alpha = -1*phi_alpha"""
                phi_mu = phi_mu + 0.5*epsilon*self._mu_gradient(mu, rho, gamma, I)
                phi_rho = phi_rho+0.5*epsilon*self._rho_gradient(mu, rho, gamma, I)
                phi_gamma = phi_gamma+0.5*epsilon*self._gamma_gradient(mu, rho, gamma, I)
            log_r = self._log_posterior2(mu, rho, gamma, I)
            log_r += np.log(multivariate_normal.pdf(phi_mu, mean=np.zeros((self.num_components)), cov=M_mu))
            log_r += np.log(multivariate_normal.pdf(phi_rho, mean=np.zeros((self.num_components)), cov=M_rho))
            log_r += np.log(norm.pdf(phi_gamma, loc=0, scale=np.sqrt(M_gamma)))
            log_r -= self._log_posterior2(chains['mu'][-1], chains['rho'][-1], chains['gamma'][-1], I)
            log_r -= np.log(multivariate_normal.pdf(phi_mu_init, mean=np.zeros((self.num_components)), cov=M_mu))
            log_r -= np.log(multivariate_normal.pdf(phi_rho_init, mean=np.zeros((self.num_components)), cov=M_rho))
            log_r -= np.log(norm.pdf(phi_gamma_init, loc=0, scale=np.sqrt(M_gamma)))
            print(phi_mu_init, phi_rho_init, phi_gamma_init)
            print(phi_mu, phi_rho, phi_gamma)
            print('log_r', log_r)
            r = np.exp(log_r)
            print(r)
            print('proposal mu', mu)
            sigmasq = np.exp(rho)
            print('proposal sigmasq', sigmasq)
            alpha = [np.exp(gamma)/(1+np.exp(gamma)), 1/(1.0+np.exp(gamma))]
            print('proposal alpha', alpha)
            if np.random.uniform() < r:
                chains['mu'].append(mu)
                chains['rho'].append(rho)
                chains['gamma'].append(gamma)
            else:
                chains['mu'].append(deepcopy(chains['mu'][-1]))
                chains['rho'].append(deepcopy(chains['rho'][-1]))
                chains['gamma'].append(deepcopy(chains['gamma'][-1]))
            I = self._compute_I(chains['mu'][-1], np.exp(chains['rho'][-1]), [np.exp(chains['gamma'][-1])/(1+np.exp(gamma)), 1/(1.0+np.exp(gamma))])
        alpha0 = np.exp(chains['gamma'])/(1+np.exp(chains['gamma']))
        alpha1 = 1.0-alpha0
        alpha_chains = np.transpose(np.vstack([alpha0, alpha1]))
        output_chains = {'mu': chains['mu'], 'sigma': np.log(chains['rho']), 'alpha': alpha_chains}
        return pd.DataFrame.from_dict(output_chains)


    def run_and_plot(self):
        """chains = {'mu': [], 'sigmasq': [], 'alpha': []}
        y_predicted = []
        group0 = []
        group1 = []
        length = 0
        while length < 100:
            res = self.gibbs_sample(chain_length=100)
            chains['mu'].append(res['mu'][-1])
            chains['sigmasq'].append(res['sigmasq'][-1])
            chains['alpha'].append(res['alpha'][-1])
            if np.random.uniform() < res['alpha'][-1][0]:
                val = np.random.normal(loc=chains['mu'][-1][0], scale=np.sqrt(chains['sigmasq'][-1][0]))
                y_predicted.append(val)
                group0.append(val)
            else:
                val = np.random.normal(loc=chains['mu'][-1][1], scale=np.sqrt(chains['sigmasq'][-1][1]))
                y_predicted.append(val)
                group1.append(val)
            length += 1
        y_predicted = np.array(y_predicted)
        group0 = np.array(group0)
        group1 = np.array(group1)
        plt.hist(group0, color='blue', label='group0')
        plt.hist(group1, color='red', label='group1')
        plt.figure(112)
        plt.hist(y_predicted)"""
        # chains = self.gibbs_sample(chain_length=10000)
        chains = self.hmc(chain_length=1000)
        mu0_chain = np.array(chains['mu'])[:,0]
        alpha0_chain = np.array(chains['alpha'])[:,0]
        plt.figure(111)
        plt.plot(range(len(mu0_chain)), mu0_chain)
        plt.xlabel('Iteration')
        plt.ylabel('\mu_0')
        plt.title('\mu_0 vs. Iteration')
        plt.figure(112)
        plt.acorr(mu0_chain, maxlags=1000)
        plt.figure(113)
        plt.acorr(alpha0_chain, maxlags=1000)
        plt.show(block=True)

if __name__ == '__main__':
    # df = pd.read_csv('tmdb-5000-movie-dataset/tmdb_5000_movies.csv', header=0)
    netflix_loader = NetflixLoader()
    netflix_loader.load_file('netflix/combined_data_1.txt')
    # netflix_loader.load_file('netflix/combined_data_2.txt')
    data = np.array(netflix_loader.df['avg_rating'])
    nmm = NormalMixtureModel(data=data, num_components=2)
    nmm.run_and_plot()
