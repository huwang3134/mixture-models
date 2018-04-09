import numpy as np
from scipy.stats import dirichlet
import pandas as pd
from netflix_loader import NetflixLoader
import matplotlib.pyplot as plt

class DirichletMixture:
    def __init__(self, data, num_components=2):
        self.num_components = num_components
        self.Y = np.array(data)
        self.num_categories = np.shape(self.Y)[1]
        self.num_examples = np.shape(self.Y)[0]

    def _sample_I(self, alpha, beta):
        component_scores = np.zeros((self.num_examples, self.num_components))
        for i in range(self.num_examples):
            component_scores[i,:] = dirichlet.pdf(np.transpose(beta), self.Y[i]+1)*alpha
        component_scores = np.transpose(np.transpose(component_scores)/np.sum(component_scores, axis=1))
        I = np.array([np.random.multinomial(1, components, 1)[0] for components in component_scores])
        nan_indicators = np.isnan(component_scores)
        nan_rows = (np.sum(nan_indicators, axis=1) > 0)
        num_nan_rows = np.sum(nan_rows)
        I[nan_rows] = np.random.multinomial(1, [1.0/self.num_components]*self.num_components, num_nan_rows)
        return I

    def gibbs(self, chain_length=1000):
        alpha = np.ones((self.num_components))/self.num_components
        alpha[-1] = 0.7
        alpha[:-1] = (1-alpha[-1])/self.num_components
        beta = np.ones((self.num_components, self.num_categories))/self.num_categories
        chains = {'alpha': [alpha], 'beta': [beta]}
        I = self._sample_I(alpha, beta)
        print(I[:20,:])
        for t in range(chain_length):
            print('alpha', chains['alpha'][-1])
            print('beta', chains['beta'][-1])
            alpha_t = np.random.dirichlet(np.sum(I, axis=0)+1, size=1)[0]
            beta_t = np.zeros((self.num_components, self.num_categories))
            counts = np.dot(np.transpose(I), self.Y)
            for j in range(self.num_components):
                beta_t[j,:] = np.random.dirichlet(counts[j,:]+1, size=1)[0]
            chains['alpha'].append(alpha_t)
            chains['beta'].append(beta_t)
            I = self._sample_I(alpha, beta)
        chains['alpha'] = np.array(chains['alpha'])
        chains['beta'] = np.array(chains['beta'])
        return chains

    def run_and_plot(self):
        chains = self.gibbs()
        alpha0_chain = chains['alpha'][:,0]
        plt.figure(111);
        plt.plot(range(len(alpha0_chain)), alpha0_chain)
        plt.figure(112)
        plt.acorr(alpha0_chain, maxlags=500)
        plt.show()

if __name__ == '__main__':
    netflix_loader = NetflixLoader()
    netflix_loader.load_file('netflix/combined_data_1.txt')
    data = netflix_loader.ratings
    model = DirichletMixture(data=data)
    model.run_and_plot()
