### 1

No, there is no date reuse in matrix addition.

### 2

...

### 3

Maybe using not ready  or changed data in calculation. 

### 4

Shared memory is visible for all threads in block, while registers is private for threads.

### 5

$$1-\frac{1}{32*32}\approx 99.9\%$$

### 6

$$1000*512=512000$$

### 7

1000

### 8

#### a

N

#### b

$$\frac{N}{T}$$

### 9

$$computational\ intensity = 36FLOP/7*4B\approx 1.29FLOP/B$$

#### a

$$max\ FLOPS = 100GB/s*1.29FLOP/B=129GFLOPS<Peak\ FLOPS$$

memory-bound

#### b

$$max\ FLOPS = 250GB/s*1.29FLOP/B=322.5GFLOPS>Peak\ FLOPS$$

compute-bound

### 10

Is BLOCK_WIDTH confused with BLOCK_SIZE?

#### a

To avoid RAW conflict, all threads in one block must be in one warp. So BLOCK_SIZE = 1,2,3,4,5.

#### b

add `__syncthreads();` between line 9 and 10.

### 11

#### a

1024

#### b

1024

#### c

8

#### d

8

#### e

(128+1)*4B=516B

#### f

$$\frac{10OP}{5*4B}=0.5OP/B$$

### 12

#### a

To achieve full occupancy,

$$block\_num = \frac{2048}{64}=32,register\_num=27*2K=54K<64K,shared\_memory=32*4KB=128KB>96KB$$

It can't achieve full occupancy, and the limiting factor is shared_memory.

#### b

To achieve full occupancy,

$$block\_num = \frac{2048}{256}=8,register\_num=31*2K=62K<64K,shared\_memory=8*8KB=64KB<96KB$$

It can achieve full occupancy.
