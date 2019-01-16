import hashlib
import random
import time

from mnist import NN_optimize
from mnist import param_update 

# Hashing in hexadecimal
def BCHash(x):
    #check that this hash is legit..
    y = int(hashlib.sha256(bytes(x, 'utf-8')).hexdigest(), 16)
    return y/(10**50)
    #return ''.join(format(ord(i), 'b') for i in y)

class Miner:
    def __init__(self, best_score=0):
        self.best_score = best_score
class Block:
    def __init__(self, txs, prevHash, nonce, sol=None, score=None):
        self.txs = txs
        self.prevHash = prevHash
        self.nonce = nonce
        self.solution = sol
        self.score = score

    def __str__(self):
        return str(self.__dict__)

def createBlock(bC, txs, score = None):
    pHash = BCHash(str(bC[-1]))
    return Block(txs, pHash, 1, score = score)

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

def solHash(bestSol):
    opter = NN_optimize({'max_iter': 10}, param_update)
    score, timeT = next(opter)
    while(score < bestSol):
      score, timeadd = next(opter)
      timeT += timeadd
    return score, timeT

#list of miner objects
def solComp(sol_miners, bC, CAPdifficulty, bestSol):
    minerResults = []
    timeResults = []
    for miner in sol_miners:
        #if they have a solution that already beats the best, skip optimization
        timesol = 0
        score = miner.best_score
        if miner.best_score <= bestSol:
            score, timesol = solHash(bestSol)
            miner.best_score = score

        trans = random.random()
        cc = createBlock(bC, trans, score)
        start = time.time()
        mine(cc, CAPdifficulty)
        stop = time.time() - start
        timeResults.append(timesol+stop)
        minerResults.append(cc)

    return min(timeResults), minerResults[timeResults.index(min(timeResults))]



def mine_blocks():
    #initialize blockchain as list
    blockChain = []
    #initial difficulty
    difficulty = BCHash("difficulty")
    PAC_difficulty = difficulty * 10
	#total number of hashes attempted before difficulty is ajusted
    total_nonce = 0
    #frequency at which difficulty is updated
    update_freq = 5
    #wanted average time to mine blocks before update
    T = update_freq
    #solution advantagee - eta
    eta = 1/5
    #initial best solution
    bestSol = 0
    #time lists
    tblist = []
    tslist = []

    #genesis block
    genBlock = Block(0, 0, 0)
    blockChain.append(genBlock)

    num_miners = 2
    num_sol_miners = 2

    sol_miners = [Miner() for _ in range(num_sol_miners)]

    while len(blockChain) < 6 * update_freq:
        print(f"Blockhain height: {len(blockChain)}")
        #update difficulty based on nonce
        if not len(blockChain) % update_freq:
            b=0
            for block in blockChain[-update_freq:]:
                if block.score == None:
                    b+=1/update_freq
            T_star = sum(tslist) + sum(tblist)
            ts_star = sum(tslist)/(len(tslist) + 0.1)
            tb_star = sum(tblist)/(len(tblist) +0.1)
            eta_star = ts_star/tb_star
            tblist = [] 
            tslist = []
            print("UPDATING DIFFICULTY")
            print(f"T_star: {T_star}")
            print(f"tb_star: {tb_star}")
            print(f"ts_star: {ts_star}")
            print(f"eta_star: {eta_star}")
            print(f"b: {b}")
            difficulty2 = difficulty *(
                (b+(1-b)*eta)/(b+(1-b)*eta_star)*T_star/T)
            
            #CAP_difficulty = int(int(difficulty, 16) / (total_nonce * update_freq))
            PAC_difficulty = 1/(eta/difficulty2-eta_star/difficulty+1/PAC_difficulty)

            difficulty = difficulty2
            print(f"BTC difficulty update: {difficulty}")
            print(f"reduced difficulty update: {difficulty}")
            total_nonce = 0

        #treat transactions as a random number
        print("DOING BTC RACE")
        time_btc, winBlock = miningComp(num_miners, blockChain, difficulty)

        print(f"DOING PAC RACE, best score: {bestSol}")
        time_pac, winSolBlock = solComp(sol_miners, blockChain, PAC_difficulty, bestSol)

        print(f"btc: {time_btc}, pac: {time_pac}")
        if time_btc < time_pac:
            blockChain.append(winBlock)
            tblist.append(time_btc)
        else:
            blockChain.append(winSolBlock)
            print("PAC WINS")
            bestSol = winSolBlock.score
            tslist.append(time_pac)

        for b in blockChain:
            print(b)

        #loop over blocks
        total_nonce += winBlock.nonce
    pass

if __name__ == '__main__':
    mine_blocks()
