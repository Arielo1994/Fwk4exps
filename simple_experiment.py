import fwk4exps.speculative_monitor as fwk

f4e = fwk.SpeculativeMonitor(cpu_count=2)

bsg_path = 'python opt_test.py'


def experimentalDesign():
    print("experimental design2")
    params = {"x": 0.0, "y": 0.0}
    S = fwk.Strategy('opt_test', bsg_path, '{x} {y}', params)
    print("S:", S)
       
    params_S2 = {"x": 1.0, "y": 0.0}
    S2 = fwk.Strategy('opt_test', bsg_path, '{x} {y}', params_S2)
    print("S2", S2)

    S3 = f4e.bestStrategy(S, S2)

    print("The best found parameter values are: ", S.params)

    f4e.terminate()

f4e.speculative_execution(experimentalDesign, 'instances.txt')
