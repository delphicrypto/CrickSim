import pickle
import random
import numpy as np
import networkx as nx
DEBUG = False


class Trace_Calls: 
#Credit goes to Richard Pattis for his Illustrate_Recursive class.
#Source:https://www.ics.uci.edu/~pattis/ICS-33/lectures/decoratorspackages.txt
    def __init__(self,f):
        self.f = f
        self.calls = 0
        self.trace = False
        self.record = []


    def illustrate(self,*args,**kargs):
        
        self.indent = 0
        self.trace = True
        answer = self.__call__(*args,**kargs)
        self.trace = False
        return answer
    
    # def __call__(self,*args,**kargs):  # bundle arbitrary arguments to this call
    #     self.calls += 1
    #     return self.f(*args,**kargs)  # unbundle arbitrary arguments to call f

    def display_records(self):
        return self.record

    def __call__(self,*args,**kargs):

        if self.trace:
            if self.indent == 0:
                print('Starting recursive illustration'+30*'-')
            print (self.indent*"."+"calling", self.f.__name__+str(args)+str(kargs))
            self.indent += 2
        self.calls += 1
        answer = self.f(*args,**kargs)
        if answer != None:
            self.record.append(answer)
        if self.trace:
            self.indent -= 2
            print (self.indent*"."+self.f.__name__+str(args)+str(kargs)+" returns", answer)
            if self.indent == 0:
                print('Ending recursive illustration'+30*'-')
        return answer
    def called(self):
        return self.calls

    def get_recursive_calls(self):
        return self.calls - 1
    
    def reset(self):
        self.calls = 0
        self.record = []


def trace(f): #Visualize recursive calls
    trace.recursive_calls = 0
    trace.depth = 0


    def _f(*args, **kwargs):


        print("  " * trace.depth, ">", f.__name__, args, kwargs)
        if trace.depth >= 1:
            trace.recursive_calls += 1
        trace.depth += 1
        res = f(*args, **kwargs)
        trace.depth -= 1
        print("  " * trace.depth, "<", res)
        print("recursive calls so far: {}".format(trace.recursive_calls))
        return res
    return _f
@Trace_Calls
def bron_kerbosch(R, P, X, graph, find_pivot=False):
    if len(P) == 0:
        if len(X) == 0:
            yield R
            # return R
    else:
        frontier = set(P)
        if find_pivot:
            #print("found_pivot")
            u = find_max_pivot(graph, P, X)
            #print(set(P), set(graph[u]))
            frontier = set(P) - set(graph[u])
        for v in frontier:
            yield from bron_kerbosch(
                R.union({v}),
                P.intersection(set(graph[v])),
                X.intersection(set(graph[v])),
                graph,
                find_pivot)

            P.remove(v)

            X = X.union({v})

def find_max_pivot(graph, P, X):
    nodes = list(P.union(X))
    u = nodes[0]
    max_intersection = len(set(graph[nodes[0]]).intersection(P))
    for n in nodes:
        if len(set(graph[n]).intersection(P)) > max_intersection:
            u = n
            max_intersection = len(set(graph[n]).intersection(P))

    return u


def bk_initial_call(graph,pivot=False, visualize=False):
    f = bron_kerbosch(set(), set(graph.keys()), set(), graph, pivot)
    # for clique in f:
    current_best = 0
    while True:
        clique = next(f)
        clique_size = len(clique)
        if current_best < clique_size:
            current_best = clique_size
            print(f"new best: {current_best}")
            print(f"new best: {clique}")
def clique_solver(graph):
    return bron_kerbosch(set(), set(graph.keys()), set(), graph, False)


def N(v, g):
    # for i, n_v in enumerate(g[v]):
    #     print(i, n_v)
    #print("{}->{}".format(v,[n_v for i, n_v in enumerate(g[v]) if n_v]))

    return [n_v for i, n_v in enumerate(g[v]) if n_v]


def is_clique(subgraph, check, graph):
    test = subgraph.copy()
    test.add(check)
    for n in test:
        if (test-{n}).issubset(set(graph[n])):
            continue
        else:
            return False
    return True

def clique_finder_rand(graph,alpha=1e50):
    start = random.randint(0, len(graph))
    nodes = list(range(0,len(graph)))
    random.shuffle(nodes)
    cur_max = 1
    cur_cli = set()
    for i, start in enumerate(nodes):
        stack = [start]
        subgraph = {start}
        visited = set()
        k = 0
        while stack:
            if random.random() > 1/np.exp(k/alpha):
                print(f"breaking {k}")
                break
            k += 1
            check = stack.pop()
            visited.add(check)
            if is_clique(subgraph, check, graph):
                subgraph.add(check)
            for nei in graph.neighbors(check):
                if nei not in visited:
                    stack.append(nei)
            if len(subgraph) > cur_max:
                cur_max = len(subgraph)
                cur_cli = subgraph
            yield cur_cli 

def nx_find_cliques(graph):
    return nx.find_cliques(graph)
complete_graph_4 = {
    1 : [2, 3, 4, 5],
    2 : [1,3,4, 5],
    3 : [1,2,4, 5],
    4 : [1,2,3, 5],
    5 : [1, 2, 3, 4]
}

if __name__ == '__main__':
    print("building graph")
    n_nodes = 100
    max_deg = 99
    # graph = {i: random.sample(set(list(range(n_nodes))) - {i}, random.randint(1, max_deg)) for i in range(n_nodes)}
    graph = nx.gnp_random_graph(100, 0.8)
    # graph = pickle.load(open("graph.pickle", "rb"))
    #print(set([1,2,3,4]))
    # print(set(test_graph.keys()))
    # init_bkb(test_graph)
    #init_bkb(frucht, pivot=True)
    cl = clique_finder(graph)
    cur_max = 0
    while True:
        try:
            m = next(cl)
        except StopIteration:
            break
        if m > cur_max:
            cur_max = m
    print(cur_max)
    f = nx.find_cliques(graph)
    cur_max = 0
    while True:
        l = len(next(f))
        if l > cur_max:
            print(l)
            cur_max = l
    # # print(cl)
    # bk_initial_call(random_graph, 5)
    # print("Pivot")

    # bk_initial_call(complete_graph_4, pivot=True, visualize=True)
