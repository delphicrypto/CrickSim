import pickle
import logging
import sys
import hashlib
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

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
    # y = hashlib.sha256(bytes(x, 'utf-8')).hexdigest()
    y = int(hashlib.sha256(bytes(x, 'utf-8')).hexdigest(), 16)
    return y
    #return ''.join(format(ord(i), 'b') for i in y)

class Miner:
    def __init__(self, best_score=0, problem='clique'):
        self.best_score = best_score
        self.solver = clique_finder_rand(graph)
        self.solve_time = 0

    def solHash(self, best_sol, BTC_time):
        """
            Solve until you beat the best.
            Return time.
        """
        timeT = 0
        score = self.best_score
        while(score <= best_sol):
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

def diff_check(hashed, difficulty):
    return hashed[:difficulty] == '0' * difficulty

def mine(block, epsilon):
    # while not diff_check(BCHash(str(block)), int(difficulty)):
    while(BCHash(str(block)) > epsilon):
        # print(block.nonce)
        # print(BCHash(str(block)))
        block.nonce = block.nonce + 1

def miningComp(num_miners, bC, difficulty):
    minerResults = []
    timeResults = []
    for i in tqdm(range(num_miners)):
        #treat transaction as a random number
        trans = random.random()
        cc = createBlock(bC, trans)
        start = time.time()
        mine(cc, diff_to_eps(difficulty))
        stop = time.time() - start
        minerResults.append(cc)
        timeResults.append(stop)
    # print(nonceResults)
    return min(timeResults), minerResults[timeResults.index(min(timeResults))]

def solComp(sol_miners, bC, CAPdifficulty, best_sol, BTC_time):
    minerResults = []
    timeResults = []
    for miner in tqdm(sol_miners):
        found_sol = False
        #if dont have a solution that already beats the best, do optimization
        if miner.best_score <= best_sol:
            score, timesol = miner.solHash(best_sol, BTC_time)
            #if got improvement before BTC_time record it
            if score:
                miner.best_score = score
                found_sol = True

        if found_sol:
            trans = random.random()
            start = time.time()
            cc = createBlock(bC, trans, miner.best_score, sol=True)
            mine_time = mine(cc, CAPdifficulty)
            stop = time.time() - start

            timeResults.append(timesol+stop)
            minerResults.append(cc)
        else:
            #no solution found, set time to infinity and block to None
            timeResults.append(sys.maxsize)
            minerResults.append(None)

    print(timeResults)
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
        (b+(1-b)*eta_star) / (b+(1-b)*eta) * (T/T_star)
            )

def update_PAC_diff(difficulty, eta, eta_star, db, db_prime):
    return eta*db_prime - eta_star*db + difficulty

def difficulty_scale(new_diff, old_diff, min_factor=1/2, max_factor=2):
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

# figure out largest integer and put it into these functions.
def eps_to_diff( eps ):
    return 1 * 10 ** 64/eps

def diff_to_eps (diff):
    return 1 * 10 ** 64/diff

def get_new_difficulties(block_chain, tslist, tblist, db, dr, update_freq, eta, eta_star, T, T_star):
        num_btc_blocks = len([b for b in block_chain[-update_freq:] if b.score == None]) / update_freq

        T_star, ts_star, tb_star = get_times(tslist, tblist)
        #keep eta_star between 0 and 1
        eta_star = min(ts_star/tb_star, 1)
        tblist, tslist = [], []

        db_prime = update_BTC_diff(db, num_btc_blocks, eta, eta_star, T, T_star)
        db_prime = difficulty_scale(db_prime, db)
        print(f"BTC difficulty updated by factor: {db_prime/db}")

        dr_prime = update_PAC_diff(dr, eta, eta_star, db, db_prime)
        if dr_prime <= 0:
            dr_prime = db_prime * eta
        print(f"PAC difficulty updated by factor: {dr_prime/dr}")
        dr = difficulty_scale(dr_prime, dr)
        eps_r = diff_to_eps(dr)
        db = db_prime
        eps_b = diff_to_eps(db)
        print(f"BTC difficulty update: {db}")
        print(f"reduced difficulty update: {dr}")
        return db, dr

def mine_blocks(eps_b=None, eps_r=None, 
                update_freq=5,
                T=0.01, eta=1/200, best_sol=200,
                num_miners=10, num_sol_miners=5,
                mode='v1', run_id="r0"):
    """
        Iterator over blocks.

        Params
            :log open file pointer for writing logs.
    """
    metrics = ['eta_star', 'T_star', 'best_sol',
             'db', 'dr',
             'tb_star', 'ts_star', 'block_height']

    #initialize blockchain as list
    block_chain = []
    #initial difficulty
    if not eps_b:
        eps_b = BCHash("difficulty")
    # eps_b = 2e71
    db = eps_to_diff(eps_b)
    #make it 10x easier to mine with solution initially
    if not eps_r:
        eps_r = eps_b * 10
    dr = eps_to_diff(eps_r)
    #initial best solution

    #time lists
    tblist = []
    tslist = []

    #genesis block
    genBlock = Block(0, 0, 0)
    block_chain.append(genBlock)
    #initial number of miners
    total = num_miners + num_sol_miners

    sol_miners = [Miner() for _ in range(num_sol_miners)]

    eta_star, tb_star, ts_star, T_star, block_height = (0,) * 5

    while True:
        #update difficulty based on nonce
        if mode == "v1":
            if not len(block_chain) % update_freq:
                db, dr = get_new_difficulties(block_chain, tblist, tslist, db, dr, update_freq, eta, eta_star, T, T_star)
                tslist = []
                tblist = []
        elif mode == 'v2':
            if not len(tslist) % update_freq and len(tslist) != 0:
                dr = difficulty_scale(dr * T / sum(tslist), dr)
                tslist = []
            if not len(tblist) % update_freq and len(tblist) != 0:
                db = difficulty_scale(db * T / sum(tblist), db)
                tblist = []

        #treat transactions as a random number
        print("DOING BTC RACE")
        time_btc, winBlock = miningComp(num_miners, block_chain, db)

        print(f"DOING PAC RACE, best score: {best_sol}")
        time_pac, winSolBlock = solComp(sol_miners, block_chain, dr, best_sol, time_btc)

        if time_btc < time_pac:
            block_chain.append(winBlock)
            tblist.append(time_btc)
            print(">>> BTC WINS")
        else:
            block_chain.append(winSolBlock)
            print(">>> PAC WINS")
            best_sol = winSolBlock.score
            tslist.append(time_pac)

        d = {}
        for m in metrics:
            d[m] = locals()[m]
        yield d

def simulation(max_height, **params):
    height = 0
    mining = mine_blocks(**params)
    while height < max_height:
        state = next(mining)
        height +=1

if __name__ == '__main__':
    simulation(200, mode = 'v2')
    # sys.exit()
    data = mine_blocks(num_sol_miners=5, num_miners=5 )

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plt.plot(data['db'], label='db')
    plt.plot(data['dr'], label='dr')
    ax.set_yscale('log')
    plt.legend()
    plt.show()
