import numpy as np
import pymc3 as pm
import random

class Sampler(object):
    def __init__(self):
        self.tree = None
        self.algoritmos = None
        self.instancias = None
        # self.parameters_algos = None
        np.random.seed(123)

    def pass_info(self, tree, algoritmos, instancias):
        self.tree = tree
        self.algoritmos = algoritmos
        self.instancias = instancias
        # self.parameters_algos = parameters_algos

    def simulations(self, total, __alg=None, sampl_alg_sum=None):

        self.tree.refreshSimulations()

        mcmc_sampl_mean = dict()
        mcmc_sampl_sd = dict()
        if sampl_alg_sum:
            # print("simulaciones sesgadas")
            
            for x in range(total):
                for k in self.algoritmos:
                    alg = self.algoritmos[k]
                    mcmc_sampl_mean[alg], mcmc_sampl_sd[alg] = Sampler.randomParameters(alg.tmpParameters)
                self.simulation(mcmc_sampl_mean, mcmc_sampl_sd, __alg, sampl_alg_sum)
        else:
            # print("simulaciones no sesgadas:")
            for x in range(total):
                for k in self.algoritmos:
                    alg = self.algoritmos[k]
                    mcmc_sampl_mean[alg], mcmc_sampl_sd[alg] = Sampler.randomParameters(alg.tmpParameters)
                self.simulation(mcmc_sampl_mean, mcmc_sampl_sd)

    def simulation(self, mcmc_sampl_mean, mcmc_sampl_sd, __alg=None, sampl_alg_sum=None):
        n = self.tree.root
        simul_mean = {}
        while n is not None:
            n.addSimulationVisit()
            if n.is_leaf == False:
              total = n.total_instances
              if n.alg1 not in simul_mean:
                simul_mean[n.alg1] = self.simul_mean(n.alg1, total, mean=mcmc_sampl_mean[n.alg1], sd=mcmc_sampl_sd[n.alg1])
              
              if n.alg2 not in simul_mean:
                simul_mean[n.alg2] = self.simul_mean(n.alg2, total, mean=mcmc_sampl_mean[n.alg2], sd=mcmc_sampl_sd[n.alg2])                       
              
              simulated_mean1 = simul_mean[n.alg1]
              simulated_mean2 = simul_mean[n.alg2]
              #print("compare:",simulated_mean1,simulated_mean2)

              if simulated_mean1 - n.delta_sig > simulated_mean2:
                  #n.p1 = n.p1+1
                  n = n.left
              else:
                  #n.p2 = n.p2+1
                  n = n.right
            else: break #leaf node

    def sampleoDeSumas(self):
        sampledSums = dict()
        for k in self.algoritmos:
            alg = self.algoritmos[k]
            sampledSums[alg, 5], sampledSums[alg, 95] = self.sampledSum(alg, 5, 95)
        return sampledSums


    @staticmethod
    def randomParameters(samples): 
        #if len(self.sampledParameters) ==0: self.sampleParameters()
        index = random.randint(0, len(samples[0])-1)
        return samples[0][index], samples[1][index]


    @staticmethod
    def sampleParameters(data):
        # https://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
        # print(data)
        # print("sampling parameters given data")
        np.random.seed(123)

        # extracted means and sigmas from
        __medias = []
        __sigmas = [] 
        # with suppress_stdout:
        with pm.Model():
            mu = pm.Normal('mu', np.mean(data), 1)
            sigma = pm.Uniform('sigma', lower=0.001, upper=2)

            returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)

            step = pm.Metropolis()
            trace = pm.sample(250, step, cores=4, progressbar=True, tune=500)

            for t in trace:
                __medias.append(t["mu"])
                __sigmas.append(t["sigma"])
        ret = __medias, __sigmas

        return ret


    def sampledSum(self, alg, alpha=None, beta=None, c=None):
        id_alg = alg
        medias, standart_deviations = alg.tmpParameters
        sampledSums = []
        total = len(self.instancias)
        if c==None: c = total - len(alg.result_list())
        if alpha is not None:
            for i in range(0, len(medias)):
                sampled_sum_i = np.random.normal(c*medias[i], np.sqrt(c)*standart_deviations[i])
                sampledSums.append(sampled_sum_i)
            sampledSums.sort()
            alpha_sum = sampledSums[(len(sampledSums)*alpha)//100]
            beta_sum = sampledSums[(len(sampledSums)*beta)//100]
            return alpha_sum, beta_sum
        else:
            randomIndex = np.random.randint(len(medias))
            return medias[randomIndex], sigmas[randomIndex]

    def simul_mean(self, alg, total, compl_sum=None, sd=None, mean=None):
        # print("simulando media para comparar")
        if compl_sum:
            mean = (sum(alg.result_list()) + compl_sum)/total*1.0
            # print("simulated Mean: "+str(__sum))
            return mean
        elif sd and mean:
            # print("sd: "+str(sd))
            # print("mean: "+str(mean))
            remaining = total - len(alg.result_list())
            # print("remaining: "+str(remaining)) 
            __sum = np.random.normal(remaining * mean, np.sqrt(remaining) * sd)
            mean = (sum(alg.result_list()) + __sum)/total*1.0
            # print("simulated Mean: "+str(__sum))
            return mean
