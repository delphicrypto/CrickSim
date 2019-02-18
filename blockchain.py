import pickle
import sys
import hashlib
import random
import time

import matplotlib.pyplot as plt
import networkx as nx

from mnist import NN_optimize
from mnist import param_update
from clique import clique_finder_rand

n_nodes = 10000
max_deg= 9000

print("building graph")
# graph = {i: random.sample(set(list(range(n_nodes))) - {i}, random.randint(1, max_deg)) for i in range(n_nodes)}
graph = nx.fast_gnp_random_graph(1000, 0.8)
# graph = pickle.load(open("graph.pickle", "rb"))
print("graph built")

# Hashing in hexadecimal
def BCHash(x):
    #check that this hash is legit..
    y = int(hashlib.sha256(bytes(x, 'utf-8')).hexdigest(), 16)
    return y
    #return ''.join(format(ord(i), 'b') for i in y)

class Miner:
    def __init__(self, best_score=0, problem='clique'):
        self.best_score = best_score
        self.solver = clique_finder_rand(graph)
        self.solve_time = 0

    def solHash(self, bestSol, BTC_time):
        """
            Solve until you beat the best.
            Return time.
        """
        timeT = 0
        score = self.best_score
        while(score <= bestSol):
            start = time.time()
            clique = next(self.solver)
            timeT += time.time() - start
            score = len(clique)
            if timeT > BTC_time:
                return None, timeT
        return score, timeT

#list of miner objects
class Block:
    def __init__(self, txs, prevHash, nonce, sol=None, score=None):
        self.txs = txs
        self.prevHash = prevHash
        self.nonce = nonce
        self.solution = sol
        self.score = score

    def __str__(self):
        return str(self.__dict__)

def createBlock(bC, txs, score = None, sol=None):
    pHash = BCHash(str(bC[-1]))
    return Block(txs, pHash, 1, sol=sol, score = score)

def mine(block, difficulty):
    while(BCHash(str(block)) > difficulty):
        # print(block.nonce)
        # print(BCHash(str(block)))
        block.nonce = block.nonce + 1

def miningComp(num_miners, bC, difficulty):
    minerResults = []
    timeResults = []
    for i in range(num_miners):
        #treat transaction as a random number
        trans = random.random()
        cc = createBlock(bC, trans)
        start = time.time()
        mine(cc, difficulty)
        stop = time.time() - start
        minerResults.append(cc)
        timeResults.append(stop)
    # print(nonceResults)
    return min(timeResults), minerResults[timeResults.index(min(timeResults))]

def solComp(sol_miners, bC, CAPdifficulty, bestSol, BTC_time):
    minerResults = []
    timeResults = []
    for miner in sol_miners:
        found_sol = False
        #if dont have a solution that already beats the best, do optimization
        if miner.best_score <= bestSol:
            score, timesol = miner.solHash(bestSol, BTC_time)
            #if got improvement before BTC_time record it
            if score:
                miner.best_score = score
                found_sol = True

        if found_sol:
            trans = random.random()
            start = time.time()
            cc = createBlock(bC, trans, miner.best_score, sol=True)
            stop = time.time() - start
            mine_time = mine(cc, CAPdifficulty)

            timeResults.append(timesol+stop)
            minerResults.append(cc)
        else:
            #no solution found, set time to infinity and block to None
            timeResults.append(sys.maxsize)
            minerResults.append(None)

    return min(timeResults), minerResults[timeResults.index(min(timeResults))]

def get_times(tslist, tblist):
    Tstar = sum(tslist) + sum(tblist)
    #average solving time
    if len(tslist) == 0:
        ts_star = sys.maxsize
    else:
        ts_star = sum(tslist) / (len(tslist) + 0.1)

    #average btc time
    if len(tblist) == 0:
        tb_star = sys.maxsize
    else:
        tb_star = sum(tblist) / (len(tblist) + 0.1)
    return Tstar, ts_star, tb_star

#updating the number of miners
def get_num_miners(total, ts_star, tb_star):
    num_BTC_miners = int(total * tb_star / (ts_star + tb_star))
    num_PAC_miners = int(total * ts_star / (ts_star + tb_star))
    print(f"Number of PAC miners: {num_PAC_miners}")
    print(f"Number of BTC miners: {num_BTC_miners}")
    return num_PAC_miners, num_BTC_miners

def update_BTC_diff(difficulty, b, eta, eta_star, T, T_star):
    return difficulty *(
        (b+(1-b)*eta) / (b+(1-b)*eta_star) * (T_star/T)
            )

def difficulty_scale(new_diff, old_diff, min_factor=1/4, max_factor=4):
    """
        Make sure difficulty not changing too hard.

    """
    ratio = new_diff / old_diff
    if ratio < min_factor:
        return old_diff * min_factor
    elif ratio > max_factor:
        return old_diff * max_factor
    else:
        return new_diff

def mine_blocks():
    #initialize blockchain as list
    blockChain = []
    #initial difficulty
    difficulty_BTC = BCHash("difficulty")
    difficulty_BTC = 1e75
    #make it 10x easier to mine with solution initially
    difficulty_PAC = difficulty_BTC * 10
    #frequency at which difficulty is updated
    update_freq = 10
    #wanted average time to mine blocks before update in seconds
    T = update_freq
    #solution advantagee - eta
    eta = 1/5
    #initial best solution
    bestSol = 1
    #time lists
    tblist = []
    tslist = []

    #genesis block
    genBlock = Block(0, 0, 0)
    blockChain.append(genBlock)

    #initial number of miners
    num_miners = 10
    num_sol_miners = 10
    total = num_miners + num_sol_miners

    sol_miners = [Miner() for _ in range(num_sol_miners)]

    data = {k: [] for k in
        ['eta_star', 'T_star', 'score', 'sol', 'db', 'dr',
            'tb_star', 'ts_star']}

    eta_star = 0
    tb_star = 0
    ts_star = 0
    T_star = 0

    while len(blockChain) < 5 * update_freq:
        print(f"Blockhain height: {len(blockChain)}")
        #update difficulty based on nonce
        if not len(blockChain) % update_freq:
            b=0
            for block in blockChain[-update_freq:]:
                if block.score == None:
                    b+=1/update_freq
            T_star, ts_star, tb_star = get_times(tslist, tblist)
            #keep eta_star between 0 and 1
            eta_star = min(ts_star/tb_star, 1)
            tblist = []
            tslist = []
            print("UPDATING DIFFICULTY")
            print(f"T_star: {T_star}")
            print(f"tb_star: {tb_star}")
            print(f"ts_star: {ts_star}")
            print(f"eta_star: {eta_star}")
            print(f"b: {b}")

            difficulty_BTC_new = update_BTC_diff(difficulty_BTC, b, eta, eta_star, T, T_star)
            difficulty_BTC_new = difficulty_scale(difficulty_BTC_new, difficulty_BTC)
            print(f"BTC difficulty updated by factor: {difficulty_BTC_new/difficulty_BTC}")

            difficulty_PAC_new = 1/( (eta/difficulty_BTC_new) - (eta_star/difficulty_BTC) + (1/difficulty_PAC) )
            if difficulty_PAC_new < 0:
                difficulty_PAC_new = difficulty_BTC_new * eta
            print(f"PAC difficulty updated by factor: {difficulty_PAC_new/difficulty_PAC}")
            difficulty_PAC = difficulty_PAC_new
            difficulty_BTC = difficulty_BTC_new

            print(f"BTC difficulty update: {difficulty_BTC}")
            print(f"reduced difficulty update: {difficulty_PAC}")

            # num_miners, num_sol_miners = get_num_miners(total, ts_star, tb_star)

        #treat transactions as a random number
        print("DOING BTC RACE")
        time_btc, winBlock = miningComp(num_miners, blockChain, difficulty_BTC)

        print(f"DOING PAC RACE, best score: {bestSol}")
        time_pac, winSolBlock = solComp(sol_miners, blockChain, difficulty_PAC, bestSol, time_btc)

        print(f"btc: {time_btc}, pac: {time_pac}")
        if time_btc < time_pac:
            blockChain.append(winBlock)
            tblist.append(time_btc)
            data['sol'].append(0)
            print(">>> BTC WINS")
        else:
            blockChain.append(winSolBlock)
            print(">>> PAC WINS")
            bestSol = winSolBlock.score
            tslist.append(time_pac)
            data['sol'].append(1)

        # for i,b in enumerate(blockChain):
            # if not i % update_freq:
                # print("="*30)
            # print(b)

        #record stats
        data['db'].append(difficulty_BTC)
        data['dr'].append(difficulty_PAC)
        data['eta_star'].append(eta_star)
        data['T_star'].append(T_star)
        data['score'].append(bestSol)
        data['tb_star'].append(tb_star)
        data['ts_star'].append(ts_star)

    return data
def norm(d):
    mn = min(d)
    mx = max(d)
    return [(x-mn) / (mx-mn) for x in d]

if __name__ == '__main__':
    data = mine_blocks()
    plt.plot(norm(data['score']), label='score')
    plt.plot(norm(data['db']), label='db')
    plt.plot(norm(data['dr']), label='dr')
    plt.legend()
    plt.show()
    print(data)
