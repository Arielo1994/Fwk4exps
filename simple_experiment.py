import fwk4exps.speculative_monitor as fwk

f4e = fwk.SpeculativeMonitor(cpu_count=7)

bsg_path = '../Metasolver/BSG_CLP'


def experimentalDesign():
    print("experimental design")
    params = {"a": 0.0, "b": 0.0, "g": 0.0, "p": 0.0}
    S = fwk.Strategy('BSG_CLP', bsg_path, '--alpha {a} --beta {b} --gamma {g} -p {p} -t 5 --min_fr=0.98', params)
    print("S:", S)
    params_S2 = {"a": 1.0, "b": 0.0, "g": 0.0, "p": 0.0}
    S2 = fwk.Strategy('BSG_CLP', bsg_path, '--alpha {a} --beta {b} --gamma {g} -p {p} -t 5 --min_fr=0.98', params_S2)
    S2.params = {"a": 1.0, "b": 0.0, "g": 0.0, "p": 0.0}
    print("S2", S2)

    S3 = f4e.bestStrategy(S, S2)
    print("lalalalaal")
    print("The best found parameter values are: ", S.params)

    f4e.terminate()

f4e.speculative_execution(experimentalDesign, '../Metasolver/extras/fw4exps/instancesBR.txt')
