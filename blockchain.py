import hashlib

blockChain = [] 
difficulty = hashlib.sha224(bytes("difficulty", 'utf-8')).hexdigest()


def BCHash( x ):
	y = hashlib.sha256(bytes(x, 'utf-8')).hexdigest()
	return ''.join(format(ord(i), 'b') for i in y)

difficulty = BCHash("difficulty")

class Block:
	def __init__( self, txs, prevHash, nonce ):
		self.txs = txs
		self.prevHash = prevHash
		self.nonce = nonce

	def __str__(self):
		return f"txs = {self.txs}, prevHash = {self.prevHash}, nonce = {self.nonce}"

def mine(block):
	while(BCHash(str(block)) > difficulty):
		block.nonce = block.nonce + 1
		print(block.nonce)
		print(BCHash(str(block)))
	blockChain.append(block)

cc = Block( 1, 10, 0 )

mine(cc)
print(blockChain[0].nonce)
#print(hash(blockChain[0]))


print(str(cc))

print(difficulty<BCHash(str(cc)))


