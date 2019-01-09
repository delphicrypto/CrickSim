import hashlib
import random

# Hashing in hexadecimal
def BCHash(x):
    #check that this hash is legit..
    y = int(hashlib.sha256(bytes(x, 'utf-8')).hexdigest(), 16)
    return hex(y)
    #return ''.join(format(ord(i), 'b') for i in y)

class Block:
    def __init__(self, txs, prevHash, nonce, sol=None, score=None):
        self.txs = txs
        self.prevHash = prevHash
        self.nonce = nonce
        self.solution = sol
        self.score = score 

    def __str__(self):
        return str(self.__dict__)

def createBlock(bC, txs):
    pHash = BCHash(str(bC[-1]))
    return Block(txs, pHash, 0)

def mine(block, difficulty):
    while(BCHash(str(block)) > difficulty):
        print(block.nonce)
        print(BCHash(str(block)))
        block.nonce = block.nonce + 1

def miningComp(numMiners, bC, difficulty):
    minerResults = []
    nonceResults = []
    for i in range(numMiners):
        #treat transaction as a random number
        trans = random.random()
        cc = createBlock(bC, trans)
        mine(cc, difficulty)
        minerResults.append(cc)
        nonceResults.append(cc.nonce)
    print(nonceResults)
    return minerResults[nonceResults.index(min(nonceResults))]

def mine_blocks():
    #initialize blockchain as list
    blockChain = []
    R = 10 #number of nonces/block on average
    #number of nonce ~ time -> equivalent to 10min/block in BTC

    #initial difficulty
    difficulty = BCHash("difficulty")
	#total number of hashes attempted before difficulty is ajusted
    total_nonce = 0
    #frequency at which difficulty is updated
    update_freq = 10

    #genesis block
    genBlock = Block(0, 0, 0)
    blockChain.append(genBlock)

    while len(blockChain) < 50:
        print(f"Blockhain height: {len(blockChain)}")
        #update difficulty based on nonce
        if not len(blockChain) % update_freq:
            T = R * update_freq
            difficulty = int(int(difficulty, 16) * (total_nonce ) / ( T ))
            #CAP_difficulty = int(int(difficulty, 16) / (total_nonce * update_freq))
            print(f"difficulty update: {difficulty}")

        #train NN
        # if score > prev_score
        #mine(block, CAP_diff)
        #else mine(block, BTC_diff)

        #treat transactions as a random number
        numMiners = 1
        winBlock = miningComp(numMiners, blockChain, difficulty)

        #impkement multiple miners as a list where the smallest timestamp is
        #taken as 'winning miner'
        #loop over blocks
        total_nonce += winBlock.nonce
        blockChain.append(winBlock)
    # print(int(difficulty, 16))
    difficulty = int(int(difficulty, 16) / (total_nonce * 200))
    difficulty = hex(difficulty)
    pass

    # 'pseudo' threading for solving problem + classical hashing

    # sequential mining - nips = neurips

    # time stamp every nonce and iteration for problem solving
        #smallest time (tmin) stamp adds to blockchain
        #others revert to state at tmin

    print(difficulty)

if __name__ == '__main__':
    # BCHash(str(
    mine_blocks()


