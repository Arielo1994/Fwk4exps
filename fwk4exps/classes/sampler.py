import numpy as np


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
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        # print("algoritmos:", self.algoritmos)
        self.tree.refreshSimulations()

        mcmc_sampl_mean = dict()
        mcmc_sampl_sd = dict()
        if sampl_alg_sum:
            # print("simulaciones sesgadas")
            for x in range(1, total + 1):
                for k in self.algoritmos:
                    alg = self.algoritmos[k]
                    hash_alg = hash(alg)
                    mcmc_sampl_mean[hash(alg)], mcmc_sampl_sd[hash(alg)] = alg.randomSampledParameters()
                self.simulation(mcmc_sampl_mean, mcmc_sampl_sd, __alg, sampl_alg_sum)
        else:
            # print("simulaciones no sesgadas:")
            for x in range(1, total + 1):
                for k in self.algoritmos:
                    alg = self.algoritmos[k]
                    mcmc_sampl_mean[hash(alg)], mcmc_sampl_sd[hash(alg)] = alg.randomSampledParameters()
                self.simulation(mcmc_sampl_mean, mcmc_sampl_sd)

    def simulation(self, mcmc_sampl_mean, mcmc_sampl_sd, __alg=None, sampl_alg_sum=None):
        n = self.tree.root
        while n is not None:
            n.addSimulationVisit()
            if n.is_leaf == False:
              total = n.total_instances
              if __alg:
                  if n.alg1 == __alg:
                      simulated_mean1 = self.simul_mean(__alg, total, compl_sum=sampl_alg_sum)
                      simulated_mean2 = self.simul_mean(n.alg2, total, mean=mcmc_sampl_mean[hash(n.alg2)], sd=mcmc_sampl_sd[hash(n.alg2)])                
                  elif n.alg2 == __alg:
                      simulated_mean1 = self.simul_mean(n.alg1, total, mean=mcmc_sampl_mean[hash(n.alg1)], sd=mcmc_sampl_sd[hash(n.alg1)])
                      simulated_mean2 = self.simul_mean(__alg, total, compl_sum=sampl_alg_sum)
                  else:
                      simulated_mean1 = self.simul_mean(n.alg1, total, mean=mcmc_sampl_mean[hash(n.alg1)], sd=mcmc_sampl_sd[hash(n.alg1)])
                      simulated_mean2 = self.simul_mean(n.alg2, total, mean=mcmc_sampl_mean[hash(n.alg2)], sd=mcmc_sampl_sd[hash(n.alg2)])
              else:
                  simulated_mean1 = self.simul_mean(n.alg1, total, mean=mcmc_sampl_mean[hash(n.alg1)], sd=mcmc_sampl_sd[hash(n.alg1)])
                  simulated_mean2 = self.simul_mean(n.alg2, total, mean=mcmc_sampl_mean[hash(n.alg2)], sd=mcmc_sampl_sd[hash(n.alg2)])
          
              if simulated_mean1 - n.delta_sig > simulated_mean2:
                  n.p1 = n.p1+1
                  n = n.left
              else:
                  n.p2 = n.p2+1
                  n = n.right

    def sampleoDeSumas(self):
        sampledSums = dict()
        for k in self.algoritmos:
            alg = self.algoritmos[k]
            sampledSums[alg, 5], sampledSums[alg, 95] = self.sampledSum(alg, 5, 95)
        return sampledSums

    def sampledSum(self, alg, alpha=None, beta=None):
        id_alg = alg
        medias, standart_deviations = alg.sampledParameters
        data = alg.result_list()
        sampledSums = []
        total = len(self.instancias)
        c = total - len(data)
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
            __sum = (sum(alg.result_list()) + compl_sum)/total*1.0
            # print("simulated Mean: "+str(__sum))
            return __sum  # + delta
        elif sd and mean:
            # print("sd: "+str(sd))
            # print("mean: "+str(mean))
            remaining = total - len(alg.result_list())
            # print("remaining: "+str(remaining)) 
            __sum = np.random.normal(remaining * mean, np.sqrt(remaining) * sd)
            __sum = (sum(alg.result_list()) + __sum)/total*1.0
            # print("simulated Mean: "+str(__sum))
            return __sum
