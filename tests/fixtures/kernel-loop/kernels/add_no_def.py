# Test fixture for tt-orchard's kernel loop: a kernel.py with no function decorated with ttl.operation at all. The
# check must end no_kernel with the plain message, which names no function.
import ttl
import ttnn

TILE_SIZE = 32
