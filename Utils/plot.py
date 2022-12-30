import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def percentify(value, max):
    return round(value / max * 100)

# Generate smooth curvess
def smoothify(yInput, depth):
    x = np.array(range(0, depth))
    y = np.array(yInput)
    # define x as 600 equally spaced values between the min and max of original x
    x_smooth = np.linspace(x.min(), x.max(), 600) 
    # define spline with degree k=3, which determines the amount of wiggle
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    # Return the x and y axis
    return x_smooth, y_smooth

def zipf_plot(combined_dict:dict, depth:int, w:int, h:int):
    
    f = plt.figure()
    f.set_figwidth(w)
    f.set_figheight(h)

    ziffianCurveValues = [100/i for i in range(1, depth+1)]
    x, y = smoothify(ziffianCurveValues, depth)
    xAxis = [str(number) for number in range(1, depth+1)]
    plt.plot(x, y, label='Ziffian Curve', ls=':', color='grey')

    for i in combined_dict:
        text_length = sum(combined_dict[i].values())
        maxValue = list(combined_dict[i].values())[0]
        yAxis = [percentify(value, maxValue) for value in list(combined_dict[i].values())[:depth]]
        x, y = smoothify(yAxis, depth)
        plt.plot(x, y, label = i + f' [{text_length}]', lw=1, alpha=0.5)

    plt.xticks(range(0, depth), xAxis)
    plt.legend()
    plt.show()

def plot_words_distribution(p:list, q:list, x_axis:list, w:int, h:int, n:int):
    print('P=%.3f Q=%.3f' % (sum(p), sum(q)))

    f = plt.figure()
    f.set_figwidth(w)
    f.set_figheight(h)

    # plot first distribution
    plt.subplot(2,1,1)
    plt.bar(x_axis[:n], p[:n])
    # plot second distribution
    plt.subplot(2,1,2)
    plt.bar(x_axis[:n], q[:n])
    # show the plot
    plt.show()