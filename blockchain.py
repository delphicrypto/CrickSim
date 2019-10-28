import pickle
import logging
import sys
import hashlib
import random
from collections import defaultdict
import time

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from clique import clique_finder_rand

n_nodes = 10000
max_deg= 9000

# print("building graph")
# graph = {i: random.sample(set(list(range(n_nodes))) - {i}, random.randint(1, max_deg)) for i in range(n_nodes)}
graph = nx.fast_gnp_random_graph(1000, 0.8)
# graph = pickle.load(open("graph.pickle", "rb"))
# print("graph built")

def animate(i):
    graph_data = open('data.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    ax1.clear()
    ax1.plot(xs, ys)

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
    # for i in tqdm(range(num_miners)):
    for i in range(num_miners):
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
    # for miner in tqdm(sol_miners):
    for miner in sol_miners:
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
            mine_time = mine(cc, diff_to_eps(CAPdifficulty))
            stop = time.time() - start

            timeResults.append(timesol+stop)
            minerResults.append(cc)
        else:
            #no solution found, set time to infinity and block to None
            timeResults.append(sys.maxsize)
            minerResults.append(None)

    # print(timeResults)
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
    # print(f"Number of PAC miners: {num_PAC_miners}")
    # print(f"Number of BTC miners: {num_BTC_miners}")
    return num_PAC_miners, num_BTC_miners

def update_BTC_diff(difficulty, b, eta, eta_star, T, T_star):
    return difficulty *(
        (b+(1-b)*eta_star) / (b+(1-b)*eta) * (T/T_star)
            )

def update_PAC_diff(difficulty, eta, eta_star, db, db_prime):
    return eta*db_prime - eta_star*db + difficulty

def difficulty_scale(new_diff, old_diff, max_factor=2):
    """
        Make sure difficulty not changing too hard.

    """
    ratio = new_diff / old_diff
    if ratio < 1/max_factor:
        return old_diff * (1/max_factor)
    elif ratio > max_factor:
        return old_diff * max_factor
    else:
        return new_diff

# figure out largest integer and put it into these functions.
def eps_to_diff( eps ):
    return 1 * 10 ** 64/eps

def diff_to_eps (diff):
    return 1 * 10 ** 64/diff

def get_new_difficulties(block_chain, tslist, tblist, db, dr, update_freq,
                        eta, T, max_factor=2):
        num_btc_blocks = len([b for b in block_chain[-update_freq:] if b.score == None]) / update_freq

        T_star, ts_star, tb_star = get_times(tslist, tblist)
        #keep eta_star between 0 and 1
        eta_star = min(ts_star/tb_star, 1)

        db_prime = update_BTC_diff(db, num_btc_blocks, eta, eta_star, T, T_star)
        db_prime = difficulty_scale(db_prime, db, max_factor=max_factor)
        # print(f">>> BTC difficulty updated by factor: {db_prime/db}")

        dr_prime = update_PAC_diff(dr, eta, eta_star, db, db_prime)
        if dr_prime <= 0:
            dr_prime = db_prime * eta
        dr = difficulty_scale(dr_prime, dr, max_factor=max_factor)
        # print(f">>> PAC difficulty updated by factor: {dr_prime/dr}")
        # print(f">>> BTC difficulty update: {db}")
        # print(f">>> reduced difficulty update: {dr}")
        return db_prime, dr_prime, eta_star, T_star, tb_star, ts_star

def mine_blocks(eps_b=None, eps_r=None,
                update_freq=5,
                v2_update_freq=5,
                T=0.01, eta=1/10, best_sol=0,
                num_miners=10, num_sol_miners=5,
                max_factor=10, bounce=False,
                mode='v2', run_id="r0",
                btc_miner_schedule=None,
                pac_miner_schedule=None,
                verbose=True):
    """
        Iterator over blocks.

        Params
            :log open file pointer for writing logs.
    """
    metrics = ['eta_star', 'T_star', 'best_sol',
             'db', 'dr',
             'tb_star', 'ts_star', 'block_height',
             'T', 'sol_blocks', 'btc_blocks']

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

    #count number of blocks for each type
    sol_blocks, btc_blocks = (0,0)

    sol_miners = [Miner() for _ in range(num_sol_miners)]

    eta_star, tb_star, ts_star, T_star, block_height = (0,) * 5

    blocks_since_sol = 0
    num_defacto =0

    do_pac_sched = not pac_miner_schedule is None
    do_btc_sched = not btc_miner_schedule is None

    if do_btc_sched:
        btc_update_todo, btc_update_todo_num = next(btc_miner_schedule)
    if do_pac_sched:
        pac_update_todo, pac_update_todo_num = next(pac_miner_schedule)

    while True:

        #update number of miners
        if do_btc_sched and block_height == btc_update_todo:
            num_miners = btc_update_todo_num
            btc_update_todo, btc_update_todo_num = next(btc_miner_schedule)
        if do_pac_sched and block_height == pac_update_todo:
            if num_sol_miners < pac_update_todo_num:
                # print(">>> Adding SOL miners")
                sol_miners.extend([Miner()] * (pac_update_todo_num - num_sol_miners))
            elif num_sol_miners > pac_update_todo_num:
                # print(">>> Removing SOL miners")
                sol_miners = sol_miners[:pac_update_todo_num]
                pac_update_todo, pac_update_todo_num = next(pac_miner_schedule)
            else:
                pass

            num_sol_miners = len(sol_miners)

        # print(f"tblist {tblist}")
        # print(f"tslist {tslist}")
        #update difficulty based on nonce
        if mode == "v1":
            if not len(block_chain) % update_freq and len(block_chain) != 0:
                db, dr, eta_star, T_star, tb_star, ts_star  = get_new_difficulties(
                                            block_chain, tblist, tslist,
                                            db, dr,
                                            update_freq, eta, T,
                                            max_factor=max_factor)
                tslist = []
                tblist = []
        elif mode == 'v2':
            if not len(tslist) % v2_update_freq and len(tslist) != 0:
                dr = difficulty_scale(dr * (T * v2_update_freq)/ sum(tslist), dr)
                tslist = []
                # print(">>> DR UPDATE")
            elif not blocks_since_sol % v2_update_freq and blocks_since_sol != 0:
                dr = 1/max_factor * dr
                num_defacto += 1
                # print(">>> DR UPDATE DE FACTO")
            else:
                pass

            #BTC UPDATE
            if not len(tblist) % update_freq and len(tblist) != 0:
                db = difficulty_scale(db * (T * update_freq) / sum(tblist), db,
                                        max_factor=max_factor)
                tblist = []
                # print(">>> DB UPDATE")
            else:
                pass
        else:
            pass

        v1_bounce = (mode == 'v1' and blocks_since_sol > 100) and bounce
        v2_bounce = (mode == 'v2' and num_defacto == 2) and bounce
        if v1_bounce or v2_bounce:
            # print("BOUNCING")
            best_sol = 0
            sol_miners = [Miner() for _ in range(num_sol_miners)]
            num_defacto = 0

        #treat transactions as a random number
        # print("DOING BTC RACE")
        time_btc, winBlock = miningComp(num_miners, block_chain, db)

        # print(f"DOING PAC RACE, best score: {best_sol}")
        time_pac, winSolBlock = solComp(sol_miners, block_chain, dr, best_sol, time_btc)

        if time_btc < time_pac:
            block_chain.append(winBlock)
            tblist.append(time_btc)
            blocks_since_sol += 1
            btc_blocks += 1
            # print(">>> BTC WINS")
        else:
            block_chain.append(winSolBlock)
            # print(">>> PAC WINS")
            sol_blocks += 1
            blocks_since_sol = 0
            best_sol = winSolBlock.score
            num_defacto = 0
            tslist.append(time_pac)

        block_height += 1
        d = {}
        for m in metrics:
            d[m] = locals()[m]
        yield d

def simulation(max_height, **params):
    height = 0
    mining = mine_blocks(**params)
    states = defaultdict(list)
    while height < max_height:
        state = next(mining)
        states['height'].append(height)
        for k, v in state.items():
            states[k].append(v)
        height +=1
            
    return states

def plotter(states, *args, show=False, save=None, log=False):
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    for a in args:
        ax.plot([s[a] for s in states], label=a)
    if log:
        ax.set_yscale('log')
    plt.legend()
    if save:
        plt.savefig(save, format='pdf')
    if show:
        plt.show()
if __name__ == '__main__':
    # data = simulation(500, eta=1/200, mode = 'v2', bounce=False, update_freq=10,
                        # pac_miner_schedule=iter([(50, 200)]))
    data = simulation(100, eta=1/200, mode = 'v2', bounce=False, update_freq=10, v2_update_freq=5)
    pickle.dump(data, open("Data/1000_blocks_v2.p", "wb"))
    plotter(data, 'dr', 'db', log=True, show=False, save="v1.pdf")
    plotter(data, 'best_sol', log=False, show=False, save="v1_scores.pdf")
    plotter(data, 'T_star', 'T', log=False, show=False, save="v1_T.pdf")
    plotter(data, 'sol_blocks', 'btc_blocks', log=False, show=False, save="v1_solblocks.pdf")
    sys.exit()
    data = mine_blocks(num_sol_miners=5, num_miners=5 )

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plt.plot(data['db'], label='db')
    plt.plot(data['dr'], label='dr')
    ax.set_yscale('log')
    plt.legend()
    plt.show()
