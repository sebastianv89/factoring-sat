from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('classic')

def collect_data(n=35):
    timing = defaultdict(list)
    fname = 'timing_long_{}.txt'.format(n)
    with open(fname.format(n)) as f:
        for line in f:
            if line.startswith('#'):
                continue
            (_n, p, q, _pq, _s) = map(int, line.split()[:5])
            t = float(line.split()[5])
            timing[(p,q)].append(t)
    return timing

def sift(timing):
    '''Find 'interesting' semiprimes'''
    means = {}
    for pq in timing:
        means[pq] = np.mean(timing[pq])
    sorted_means = sorted(means.items(), key = lambda x: x[1])
    return {
        'min' : sorted_means[0][0],
        'median' : sorted_means[len(sorted_means)//2][0],
        'max' : sorted_means[-1][0]
    }

def plot_data(timing, interesting, n=35):
    a = np.array([timing[pq] for pq in timing])
    seeds = len(a[0])
    min_seeds = np.amin(a, axis=1)
    max_seeds = np.amax(a, axis=1)
    median_seeds = np.median(a, axis=1)
    mean_seeds = np.mean(a, axis=1)

    fig, axes = plt.subplots(2, 2)
    #fig.suptitle('{} seeds'.format(seeds))

    axes[0,0].hist(median_seeds, bins=20)
    axes[0,0].set_title('median')
    axes[0,0].set_xlabel('$T(n)$: time in seconds')
    axes[0,0].set_ylabel('# solutions')

    axes[0,1].hist(mean_seeds, bins=20)
    axes[0,1].set_title('mean')
    axes[0,1].set_xlabel('$T(n)$: time in seconds')
    axes[0,1].set_ylabel('# solutions')

    axes[1,0].hist(min_seeds, bins=20)
    axes[1,0].set_title('minimum')
    axes[1,0].set_xlabel('$T(n)$: time in seconds')
    axes[1,0].set_ylabel('# solutions')

    axes[1,1].hist(max_seeds, bins=20)
    axes[1,1].set_title('maximum')
    axes[1,1].set_xlabel('$T(n)$: time in seconds')
    axes[1,1].set_ylabel('# solutions')

    plt.tight_layout()
    plt.savefig('dist_{}.pdf'.format(n))


    for xlog in [False, True]:
        fig, axes = plt.subplots(3, 1)
        (p,q) = interesting['min']
        easiest = timing[(p,q)]
        if xlog:
            _, bins = np.histogram(np.log10(easiest), bins='auto')
            axes[0].hist(easiest, bins=10**bins)
            axes[0].set_xscale('log')
            #axes[0].set_xlim((10**-3, 10**1))
        else:
            axes[0].hist(easiest, bins=25)
        axes[0].set_title('easiest semiprime (${}={}*{}$)'.format(p*q, p, q))
        axes[0].set_xlabel('$T(n)$: time in seconds')
        axes[0].set_ylabel('# solutions')

        (p,q) = interesting['median']
        average = timing[(p,q)]
        if xlog:
            _, bins = np.histogram(np.log10(average), bins='auto')
            axes[1].hist(average, bins=10**bins)
            axes[1].set_xscale('log')
            #axes[1].set_xlim((10**-3, 10**1))
        else:
            axes[1].hist(average, bins=25)
        axes[1].set_title('median semiprime (${}={}*{}$)'.format(p*q, p, q))
        axes[1].set_xlabel('$T(n)$: time in seconds')
        axes[1].set_ylabel('# solutions')

        (p,q) = interesting['max']
        hardest = timing[(p,q)]
        if xlog:
            _, bins = np.histogram(np.log10(hardest), bins='auto')
            axes[2].hist(hardest, bins=10**bins)
            axes[2].set_xscale('log')
            #axes[2].set_xlim((10**-3, 10**1))
        else:
            axes[2].hist(hardest, bins=25)
        axes[2].set_title('hardest semiprime (${}={}*{}$)'.format(p*q, p, q))
        axes[2].set_xlabel('$T(n)$: time in seconds')
        axes[2].set_ylabel('# solutions')

        plt.tight_layout()
        plt.savefig('zoomed{}_{}.pdf'.format('_log' if xlog else '', n))

def main():
    for bitsize in [30, 35]:
        timing = collect_data(n=bitsize)
        plot_data(timing, sift(timing), n=bitsize)

if __name__ == '__main__':
    main()

