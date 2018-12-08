blockChain = [] 
difficulty = hash("difficulty")
print(difficulty)


class Block:
	def __init__( self, txs, prevHash, nonce ):
		self.txs = txs
		self.prevHash = prevHash
		self.nonce = nonce

#def mine(block):
#	while(hash(block) > difficulty):
#		block.nonce = block.nonce + 1
#		print(block.nonce)
#		print(hash(block))
#	blockChain.append(block)

cc = Block( 1, 10, -12 )
print(hash(cc) < difficulty)

print(difficulty)

#mine(cc)
#print(blockChain[0].nonce)
#print(hash(blockChain[0]))


