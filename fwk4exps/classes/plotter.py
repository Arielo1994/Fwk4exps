class Plotter(object):
    def __init__(self):
        pass
    
    def clearAnimationData():
        open('fwk4exps/dataQanimation.txt', 'w').close()
        open('fwk4exps/dataEQanimation.txt', 'w').close()
        open('fwk4exps/histogram.info', 'w').close()
        open('fwk4exps/end.txt', 'w').close()

    def animateQuality(i, ax):
        global Qann_list, __numOfExecutions, the_end

        # for i, a in enumerate(Qann_list):
        #
        #    a.remove()
        # Qann_list[:] = []

        graph_data_q = open('fwk4exps/dataQanimation.txt', 'r').read()
        graph_data_eq = open('fwk4exps/dataEQanimation.txt', 'r').read()
        lines_q = graph_data_q.split('\n')
        lines_eq = graph_data_eq.split('\n')
        xs = []
        ys = []
        us = []
        vs = []
        ws = []
        for line in lines_q:
            if len(line) > 1:
                # x, y, z = line.split(',')
                x, y = line.split(',')
                xs.append(float(x))
                ys.append(float(y))
        for line in lines_eq:
            if len(line) > 1:
                # x, y, z = line.split(',')
                u, v, w = line.split(',')
                us.append(float(u))
                vs.append(float(v))
                ws.append(float(w))
        ax.clear()
        ax.plot(xs, ys, label="realquality")
        ax.plot(us, vs, label="expected_pess_quality")
        ax.plot(us, ws, label="expected_opt_quality")

        ax.legend()
        # for i in range(0,len(xs)):
        # ann=ax.annotate(__numOfExecutions, (xs[len(xs)-1], ys[len(xs)-1]))
        # Qann_list.append(ann)

    def animate_histograms(i, ax):
        global algoritmos
        ax.clear()
        # histogram_info = None
        if os.stat('fwk4exps/histogram.info').st_size!=0:
            
            with open('fwk4exps/histogram.info', 'rb') as histogram_info_file:
                histogram_info = pickle.load(histogram_info_file)
                for key in histogram_info:
                    bins = np.linspace(0.8, 1, 100)
                    __label = key  # algoritmos[key].to_string()
                    ax.hist(histogram_info[key], bins, alpha=0.3, label=__label)
                    ax.legend(loc='upper right')

        ax.set_title("posterior distribution of the means")
        ax.set_ylabel("frequency")
        ax.set_xlabel("magnitude")

    def save_histograms_info():
        global algoritmos
        open('fwk4exps/histogram.info', 'w').close()
        histogram_info = dict()
        for k in algoritmos:
            alg = algoritmos[k]
            if alg.sampledParameters:
                histogram_info[alg.to_string()] = alg.sampledParameters[0]
        with open('fwk4exps/histogram.info', 'wb') as histogram_info_file:
            pickle.dump(histogram_info, histogram_info_file)

    def plotter_function():
        global quality_animation, parameter_histogram, quality_frame, histogram_frame
        clearAnimationData()
        # global Qax1
        quality_frame = plt.figure()
        Qax1 = quality_frame.add_subplot(1, 1, 1)    
        quality_animation = animation.FuncAnimation(quality_frame, animateQuality, fargs=[Qax1], interval=4000)

        histogram_frame = plt.figure()
        ax = histogram_frame.add_subplot(1,1,1)     
        parameter_histogram = animation.FuncAnimation(histogram_frame, animate_histograms, fargs=[ax], interval=4000)
        #livegraphQuality()
        #plot_histograms()
        plt.show()   

