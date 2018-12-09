import hashlib

def BCHash( x ):
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
        # return f"txs = {self.txs}, prevHash = {self.prevHash}, nonce = {self.nonce}"

def createBlock(bC, txs):
    pHash = BCHash( str(bC[-1]) )
    return Block(txs, pHash, 0)

def mine(block, difficulty):
    while(BCHash(str(block)) > difficulty):
        block.nonce = block.nonce + 1
        print("HI")
        print(block.nonce)
        print(BCHash(str(block)))
    # return None


def mine_blocks():
    time = 0
    total_nonce = 0

    blockChain = []
    difficulty = BCHash("difficulty")
    update_freq = 10

    genBlock = Block(0, 0, 0)
    blockChain.append(genBlock)

    while len(blockChain) < 50:
        print(len(blockChain))
        #update difficulty based on nonce
        if not len(blockChain) % update_freq:
            BTC_difficulty = int(int(difficulty, 16) / (total_nonce * update_freq * 2))
            CAP_difficulty = int(int(difficulty, 16) / (total_nonce * update_freq))
            print(f"difficulty update")

        #train NN
        # if score > prev_score
        #mine(block, CAP_diff)
        #else mine(block, BTC_diff)

        cc = createBlock( blockChain, 1 )
        mine(cc, difficulty)
        total_nonce += cc.nonce
        blockChain.append(cc)
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
    mine_blocks()


