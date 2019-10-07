import fwk4exps.speculative_monitor as f4e


def esperimentalDesign():

    S = f4e.Strategy('BSG_CLP', '-a {a} -b {b} -g {g} -p {p}')
    S.params = {"a": 0.0, "b": 0.0, "g": 0.0, "p": 0.0}

    a = f4e.best_param_value(S, "a", [0.0, 1.0, 2.0, 4.0, 8.0])
    S.set_param("a", a)
    b = f4e.best_param_value(S, "b", [0.0, 0.5, 1.0, 2.0, 4.0])
    S.set_param("b", b)
    g = f4e.best_param_value(S, "g", [0.0, 0.1, 0.2, 0.3, 0.4])
    S.set_param("g", g)
    p = f4e.best_param_value(S, "p", [0.0, 0.1, 0.2, 0.3, 0.4])
    S.set_param("p", p)

    print("The best found parameter values are: " + S.params)

f4e.speculativeExecution(experimentalDesign, '../Metasolver/extras/fw4exps/instancesBR.txt')
