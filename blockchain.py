import hashlib




def BCHash( x ):
	y = hashlib.sha256(bytes(x, 'utf-8')).hexdigest()
	return ''.join(format(ord(i), 'b') for i in y)



class Block:
	def __init__( self, txs, prevHash, nonce ):
		self.txs = txs
		self.prevHash = prevHash
		self.nonce = nonce

	def __str__(self):
		return f"txs = {self.txs}, prevHash = {self.prevHash}, nonce = {self.nonce}"

def createBlock( bC, txs):
	pHash = BCHash( str(bC[-1]) )
	return Block(txs, pHash, 0)

def mine(block):
	while(BCHash(str(block)) > difficulty):
		block.nonce = block.nonce + 1
		print(block.nonce)
		print(BCHash(str(block)))
	blockChain.append(block)




if __name__ == '__main__':
	
	blockChain = [] 
	difficulty = BCHash("difficulty")

	genBlock = Block(0, 0, 0)
	blockChain.append(genBlock)

	cc = createBlock( blockChain, 1 )

	mine(cc)
	print(blockChain[0].nonce)
	print(blockChain[1].nonce)
	#print(hash(blockChain[0]))
	
	print(str(cc))

	print(difficulty<BCHash(str(cc)))


