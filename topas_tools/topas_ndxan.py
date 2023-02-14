import argparse
import json
from ast import literal_eval


def parse_str_int_float(item):
    """returns string, int or float either from a list or not"""

    # recursion like a boss
    if isinstance(item, list) and len(item) == 1:
        return parse_str_int_float(item[0])
    elif isinstance(item, list):
        return [parse_str_int_float(i) for i in item]

    try:
        return literal_eval(item)
    except ValueError:
        return item
    except:
        return item


def pprint(d):
    """pretty printing using the json module"""
    print(json.dumps(d, sort_keys=True, indent=4))


def gen_lines(f):
    """return all the lines in every single file"""
    yield from f


def get_lines(lines, options):
    # Indexing_Solutions_With_Zero_Error_2
    # Indexing_Solutions_2

    start = False
    for line in lines:

        if not line:
            continue
        elif line.startswith('Indexing_Solutions'):
            start = not start
        elif ')' in line and start:
            # split on ' and return list of rest, except for the number
            yield line.split("'")[0].replace(')', '').split()
        elif 'Zero' in line and not start:
            options.zero_corr = True
        elif start and line.startswith('}'):
            break


def cosort(*lsts):
    """Takes a few lists and sorts them based on the values of the first list."""
    tmp = list(zip(*lsts))
    tmp.sort()
    return list(zip(*tmp))


def plot_3d(iterator, x_key, y_key, z_key, title="plot", picker=['spgr', 'num']):
    """3d plot: takes 3 keys and an iterator and will plot the values of the keys given"""
    args = [x_key, y_key, z_key] + picker

    xyz_gen = (tuple([d[arg] for arg in args]) for d in iterator)
    x, y, z, spgr, num = list(zip(*xyz_gen))

    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    def onpick(event):
        ind = event.ind

        pckx = np.take(x,    ind)
        pcky = np.take(y,    ind)
        pckz = np.take(z,    ind)
        pckspgr = np.take(spgr, ind)
        pcknum = np.take(num,  ind)

        print()
        for n in range(len(ind)):
            print('idx: {}, {}: {}, {}: {}, {}: {}, spgr: {}, #{}'.format(
                ind[n], x_key, pckx[n], y_key, pcky[n], z_key, pckz[n], pckspgr[n].split('.')[0], pcknum[n]))
        if len(ind) > 5:
            print('number of points:', len(ind))

    def onkeypress(event):
        if event.key == 'x':
            ax.view_init(0, 0)
            plt.draw()
        if event.key == 'y':
            ax.view_init(0, -90)
            plt.draw()
        if event.key == 'z':
            ax.view_init(90, -90)
            plt.draw()

    fig = plt.figure()
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('key_press_event', onkeypress)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_zlabel(z_key)
    ax.set_title(title)

    ax.scatter(x, y, z, c='r', marker='.', picker=True)
    plt.show()


def plot_2d(iterator, x_key, y_key, picker=['spgr', 'num']):
    """plots given keys in a 2D scatter"""
    args = [x_key, y_key] + picker

    xy_gen = (tuple([d[arg] for arg in args]) for d in iterator)

    x, y, spgr, num = list(zip(*xy_gen))

    import matplotlib.pyplot as plt
    import numpy as np

    def onpick(event):
        ind = event.ind

        pckx = np.take(x,    ind)
        pcky = np.take(y,    ind)
        pckspgr = np.take(spgr, ind)
        pcknum = np.take(num,  ind)

        print()
        for n in range(len(ind)):
            print('idx: {}, {}: {}, {}: {}, spgr: {}, #{}'.format(
                ind[n], x_key, pckx[n], y_key, pcky[n], pckspgr[n].split('.')[0], pcknum[n]))
        if len(ind) > 5:
            print('number of points:', len(ind))

    # col = ax.scatter(x, y, 100*s, c, picker=True) => s,c add color/size to
    # points, maybe nice for best solutions from sflog ??

    fig = plt.figure()
    fig.canvas.mpl_connect('pick_event', onpick)

    ax = fig.add_subplot(111)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)

    ax.scatter(x, y, c='r', marker='.', picker=True)
    plt.show()


def plot_1d(iterator, x_key, sort=None, title="Plot", picker=['spgr', 'num']):
    """plots given keys in a 2D scatter"""
    args = [x_key] + picker

    x_gen = (tuple([d[arg] for arg in args]) for d in iterator)

    x, spgr, num = list(zip(*x_gen))

    if sort:
        x, num, spgr = cosort(x, num, spgr)

    import matplotlib.pyplot as plt
    import numpy as np

    def onpick(event):
        ind = event.ind

        pckx = np.take(x,    ind)
        pckspgr = np.take(spgr, ind)
        pcknum = np.take(num,  ind)

        print()
        for n in range(len(ind)):
            print('idx: {}, {}: {}, spgr: {}, #{}'.format(
                ind[n], x_key, pckx[n], pckspgr[n].split('.')[0], pcknum[n]))
        if len(ind) > 5:
            print('number of points:', len(ind))

    # col = ax.scatter(x, y, 100*s, c, picker=True) => s,c add color/size to
    # points, maybe nice for best solutions from sflog ??

    fig = plt.figure()
    fig.canvas.mpl_connect('pick_event', onpick)

    ax = fig.add_subplot(111)
    ax.set_ylabel(x_key)
    ax.set_xlabel('idx')
    ax.set_title(title)

    ax.plot(x, c='r', marker='.', linewidth=0, picker=True)
    plt.show()


def histogram(iterator, x_key, title="Histogram"):
    """plots a histogram of the specified keyword"""
    x = [d[x_key] for d in iterator]

    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n, bins, patches = ax.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

    mu = np.mean(x)    # mean
    sigma = np.std(x)     # standard deviation

    y = mlab.normpdf(bins, mu, sigma)

    ax.plot(bins, y, 'r', linewidth=1)
    ax.set_title(title)

    # print '{:12} - {:12} : {:12}'.format('begin','end','val')
    # for i in range(len(n)):
    #   begin = bins[i]
    #   end   = bins[i+1]
    #   val   = n[i]
    #   print '{:12f} - {:12f} : {:12f}'.format(begin,end,val)

    print("mean:", mu)
    print("sigma:", sigma)

    plt.grid(True)

    plt.show()


def counter(iterator, key):
    """counts the frequency of the value for a given key"""
    freqs = {}
    vals = (d[key] for d in iterator)

    for val in vals:
        if val in freqs:
            freqs[val] += 1
        else:
            freqs[val] = 1

    pprint(freqs)


def table_out(iterator, keywords, out=None):
    """Lists output of specified keywords to stdout (or outfile if an open file object specified)"""
    header = '{:<10}'*len(keywords)
    print(header.format(*keywords), file=out)

    fmt = ''.join([f'{{{keyword}:<10}}' for keyword in keywords])

    for d in iterator:
        print(fmt.format(**d), file=out)


def gen_filter(iterator, args):
    """generator that filters dictionaries based on the input
    args is a list containing the keyword, operator and value"""
    assert len(args) == 3

    kw = args[0]
    op = args[1]
    val = parse_str_int_float(args[2])

    operators = {'gt': '>', 'ge': '>=', 'eq': '==',
                 'ne': '!=', 'le': '<=', 'lt': '<'}

    def conditional(d, kw, val): return eval(f'd[kw] {operators[op]} val')

    for d in iterator:
        if conditional(d, kw, val):
            yield d


def main():
    usage = """superanalyser [KEYWORD ...]
                     [-h] [-v] [-c] [-s N] [-t] [-m] [-i] [-o]
                     [-e KW OP VAL] [-x FILE]"""

    description = """Notes:
- Requires numpy and matplotlib for plotting.
- Keywords/values are case sensitive
- Based on a stripped down version of superanalyser.py, works the same way"""

    parser = argparse.ArgumentParser(usage=usage,
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("keywords",
                        type=str, metavar="KEYWORD", nargs='*',
                        help="Keywords for values to analyse. The outcome depends on the number of keywords supplied. With no optional arguments, 1 keyword: histogram; 2 keywords: 2d scatter; 3 keywords: 3d scatter. Run with arguments '-s 0' to see what kind of keywords would be available.")

    parser.add_argument("-x", "--index",
                        action="store", type=str, dest="index_out",
                        help="Output from indexing routine in Topas. Default filename: indexing.ndx")

    parser.add_argument("-c", "--count",
                        action="store_true", dest="count",
                        help="Counts occurances of ARG1 and exits.")

    parser.add_argument("-s", "--show",
                        action="store", type=int, dest="show", metavar='N',
                        help="Generates and prints a single entry with index N and exits.")

    parser.add_argument("-t", "--table",
                        action="store_true", dest="table",
                        help="Prints the values for the given keywords to STDOUT. (default: False)")

    parser.add_argument("-e", "--filter",
                        action="append", dest="filter", nargs=3, metavar='PAR',
                        help="Filters the entries where the value of the keyword does or does not follow the given (in)equality. Order should be 'keyword operator value', where the operator can be gt : >, ge : >=, eq : ==, ne : !=, le : <=, lt : <. Multiple conditionals can be given by chaining them, e.g.: -e spgr eq cmcm -e rsf le 30.")

    parser.add_argument("-m", "--fast",
                        action="store_true", dest="fastmode",
                        help="Turns on fast mode. This will significantly reduce the time taken to parse all the logs, because it only looks for the specified keywords and those needed for --filter. Time taken decreases by up to 500%%.")  # escape % as %%

    parser.add_argument("-i", "--hist",
                        action="store_true", dest="hist",
                        help="Shows a histogram and plots the probability distribution of the first keyword given.")

    parser.add_argument("-o", "--sort",
                        action="store_true", dest="sort",
                        help="Turns on sorting for the 1D plot. Only works for 1D plot. Note: This messes with the indexes, so the idx value can't be used with --show")

    parser.add_argument("-l", "--title",
                        action="store", type=str, dest="title",
                        help="Title for the plot.")

    parser.set_defaults(count=False,
                        table=False,
                        show=None,
                        filter=[],
                        fastmode=False,
                        sort=False,
                        hist=False,
                        debug=False,
                        title='',

                        index_out='indexing.ndx',
                        zero_corr=False)

    options = parser.parse_args()
    args = options.keywords
    print(options, args)

    f = open(options.index_out)

    print(f.name, 'opened')

    lines = gen_lines(f)
    lines = get_lines(lines, options)
    lines = (parse_str_int_float(line) for line in lines)

    if options.zero_corr:
        colnames = ('num', 'spgr', 'status', 'unindexed', 'volume',
                    'zero_corr', 'gof', 'a', 'b', 'c', 'A', 'B', 'C')
    else:
        colnames = ('num', 'spgr', 'status', 'unindexed',
                    'volume', 'gof', 'a', 'b', 'c', 'A', 'B', 'C')

    dicts = (dict(list(zip(colnames, line))) for line in lines)

    # for d in dicts:
    #   pprint(d)
    #   raw_input()

    iterator = dicts

    if options.show is not None:
        for x in range(options.show):
            next(iterator)
        pprint(next(iterator))

    elif options.table:
        table_out(iterator, args)

    elif options.count:
        counter(iterator, args[0])

    elif options.hist:
        histogram(iterator, args[0])

    elif len(args) == 1:
        plot_1d(iterator, args[0])

    elif len(args) == 2:
        plot_2d(iterator, args[0], args[1])

    elif len(args) == 3:
        plot_3d(iterator, args[0], args[1], args[2], title=title)

    elif len(args) > 3:
        table_out(iterator, args)

    else:
        s = 0
        for item in iterator:
            s += 1
        print(s)


if __name__ == '__main__':
    main()
