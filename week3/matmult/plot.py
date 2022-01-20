import math
from turtle import color
import matplotlib.pyplot as plt
import os


gpu1 = [
    (96.000,      8.258),  # matmult_gpu1
    (384.000,      8.348),  # matmult_gpu1
    (1536.000,      8.229),  # matmult_gpu1
    (6144.000,      8.252),  # matmult_gpu1
    (24576.000,      8.257),  # matmult_gpu1
    (98304.000,      7.459),  # matmult_gpu1
]

gpu2 = [
    (96.000,   1637.156),  # matmult_gpu2
    (384.000,  10410.631),  # matmult_gpu2
    (1536.000,  51227.755),  # matmult_gpu2
    (6144.000, 101681.872),  # matmult_gpu2
    (24576.000, 174897.587),  # matmult_gpu2
    (98304.000, 225317.178),  # matmult_gpu2
    (393216.000, 252452.039),  # matmult_gpu2
    (1572864.000, 266700.821),  # matmult_gpu2
    (6291456.000, 275470.784),  # matmult_gpu2
]

gpu3 = [
    (96.000,   1365.766),  # matmult_gpu3
    (384.000,   8082.956),  # matmult_gpu3
    (1536.000,  38493.043),  # matmult_gpu3
    (6144.000,  87536.831),  # matmult_gpu3
    (24576.000, 170468.633),  # matmult_gpu3
    (98304.000, 222526.332),  # matmult_gpu3
    (393216.000, 254053.516),  # matmult_gpu3
    (1572864.000, 276127.967),  # matmult_gpu3
    (6291456.000, 284121.337),  # matmult_gpu3
]

gpu4_8 = [
    (96.000,   1462.324),  # matmult_gpu4
    (384.000,   8802.727),  # matmult_gpu4
    (1536.000,  42735.273),  # matmult_gpu4
    (6144.000, 120859.188),  # matmult_gpu4
    (24576.000, 283347.272),  # matmult_gpu4
    (98304.000, 422783.390),  # matmult_gpu4
    (393216.000, 585018.081),  # matmult_gpu4
    (1572864.000, 714427.962),  # matmult_gpu4
    (6291456.000, 790384.917),  # matmult_gpu4
]

gpu4_16 = [
    (96.000,   1188.859),  # matmult_gpu4
    (384.000,   6749.982),  # matmult_gpu4
    (1536.000,  31021.209),  # matmult_gpu4
    (6144.000,  99854.908),  # matmult_gpu4
    (24576.000, 277866.804),  # matmult_gpu4
    (98304.000, 520602.690),  # matmult_gpu4
    (393216.000, 725536.279),  # matmult_gpu4
    (1572864.000, 921617.727),  # matmult_gpu4
    (6291456.000, 1064386.273),  # matmult_gpu4]
]

gpu4_32 = [
    (96.000,   1212.282),  # matmult_gpu4
    (384.000,   6889.679),  # matmult_gpu4
    (1536.000,  31172.928),  # matmult_gpu4
    (6144.000,  99600.993),  # matmult_gpu4
    (24576.000, 310028.414),  # matmult_gpu4
    (98304.000, 598681.264),  # matmult_gpu4
    (393216.000, 900969.171),  # matmult_gpu4
    (1572864.000, 1184933.617),  # matmult_gpu4
    (6291456.000, 1391190.644),  # matmult_gpu4]
]

gpu4_64 = [
    (96.000,    900.782),  # matmult_gpu4
    (384.000,   4624.747),  # matmult_gpu4
    (1536.000,  20154.649),  # matmult_gpu4
    (6144.000,  69357.705),  # matmult_gpu4
    (24576.000, 228737.327),  # matmult_gpu4
    (98304.000, 329881.417),  # matmult_gpu4
    (393216.000, 453853.401),  # matmult_gpu4
    (1572864.000, 518081.044),  # matmult_gpu4
    (6291456.000, 564632.194),  # matmult_gpu4
]

gpu5 = [
    (96.000,   1781.128),  # matmult_gpu5
    (384.000,  12215.695),  # matmult_gpu5
    (1536.000,  64766.347),  # matmult_gpu5
    (6144.000, 155873.888),  # matmult_gpu5
    (24576.000, 342075.354),  # matmult_gpu5
    (98304.000, 586934.586),  # matmult_gpu5
    (393216.000, 861884.575),  # matmult_gpu5
    (1572864.000, 1134304.262),  # matmult_gpu5
    (6291456.000, 1367879.165),  # matmult_gpu5
]

gpulib = [
    (96.000,    618.446),  # matmult_gpulib
    (384.000,   4892.276),  # matmult_gpulib
    (1536.000,  34535.110),  # matmult_gpulib
    (6144.000, 136196.716),  # matmult_gpulib
    (24576.000, 477913.430),  # matmult_gpulib
    (98304.000, 1100821.543),  # matmult_gpulib
    (393216.000, 2095962.012),  # matmult_gpulib
    (1572864.000, 3592969.106),  # matmult_gpulib
    (6291456.000, 5801759.704),  # matmult_gpulib
]

lib = [
    (96.000,  63636.636),  # matmult_lib
    (384.000, 357438.440),  # matmult_lib
    (1536.000, 627873.718),  # matmult_lib
    (6144.000, 827463.918),  # matmult_lib
    (24576.000, 921798.079),  # matmult_lib
    (98304.000, 922478.605),  # matmult_lib
    (393216.000, 1017679.699),  # matmult_lib
    (1572864.000, 1006362.823),  # matmult_lib
    (6291456.000, 964348.262),  # matmult_lib
]

gpu1m = [math.log(m/1000, 2) for m, p in gpu1]
gpu1p = [p/1000 for m, p in gpu1]
gpu2m = [math.log(m/1000, 2) for m, p in gpu2]
gpu2p = [p/1000 for m, p in gpu2]
gpu3m = [math.log(m/1000, 2) for m, p in gpu3]
gpu3p = [p/1000 for m, p in gpu3]
gpu4_8m = [math.log(m/1000, 2) for m, p in gpu4_8]
gpu4_8p = [p/1000 for m, p in gpu4_8]
gpu4_16m = [math.log(m/1000, 2) for m, p in gpu4_16]
gpu4_16p = [p/1000 for m, p in gpu4_16]
gpu4_32m = [math.log(m/1000, 2) for m, p in gpu4_32]
gpu4_32p = [p/1000 for m, p in gpu4_32]
gpu4_64m = [math.log(m/1000, 2) for m, p in gpu4_64]
gpu4_64p = [p/1000 for m, p in gpu4_64]
gpu5m = [math.log(m/1000, 2) for m, p in gpu5]
gpu5p = [p/1000 for m, p in gpu5]
gpulibm = [math.log(m/1000, 2) for m, p in gpulib]
gpulibp = [p/1000 for m, p in gpulib]
libm = [math.log(m/1000, 2) for m, p in lib]
libp = [p/1000 for m, p in lib]

plt.figure(1)
plt.title('BLAS vs GPU1 vs GPU2')
plt.scatter(gpu1m, gpu1p, marker='<', color='r')
plt.scatter(gpu2m, gpu2p, marker='>', color='yellow')
plt.scatter(libm, libp, marker='s', color='green')
plt.ylabel("Gflops/s")
plt.xlabel("Memory  $log_{2}(MB)$")
plt.grid(axis='y')
plt.legend(['GPU1', 'GPU2', 'BLAS'])
plt.savefig(f'BLAS_vs_GPU1_vs_GPU2.png')
# plt.show()

plt.figure(2)
plt.title('BLAS vs GPU3 vs GPU4_8')
plt.scatter(gpu3m, gpu3p, marker='<', color='r')
plt.scatter(gpu4_8m, gpu4_8p, marker='>', color='yellow')
plt.scatter(libm, libp, marker='s', color='green')
plt.ylabel("Gflops/s")
plt.xlabel("Memory  $log_{2}(MB)$")
plt.grid(axis='y')
plt.legend(['GPU3', 'GPU4', 'BLAS'])
plt.savefig(f'BLAS_vs_GPU3_vs_GPU4_8.png')
# plt.show()

plt.figure(3)
plt.title('BLAS vs GPU3 vs GPU4_16')
plt.scatter(gpu3m, gpu3p, marker='<', color='r')
plt.scatter(gpu4_16m, gpu4_16p, marker='>', color='yellow')
plt.scatter(libm, libp, marker='s', color='green')
plt.ylabel("Gflops/s")
plt.xlabel("Memory  $log_{2}(MB)$")
plt.grid(axis='y')
plt.legend(['GPU3', 'GPU4', 'BLAS'])
plt.savefig(f'BLAS_vs_GPU3_vs_GPU4_16.png')
# plt.show()

plt.figure(4)
plt.title('BLAS vs GPU3 vs GPU4_32')
plt.scatter(gpu3m, gpu3p, marker='<', color='r')
plt.scatter(gpu4_32m, gpu4_32p, marker='>', color='yellow')
plt.scatter(libm, libp, marker='s', color='green')
plt.ylabel("Gflops/s")
plt.xlabel("Memory  $log_{2}(MB)$")
plt.grid(axis='y')
plt.legend(['GPU3', 'GPU4', 'BLAS'])
plt.savefig(f'BLAS_vs_GPU3_vs_GPU4_32.png')
# plt.show()

plt.figure(5)
plt.title('BLAS vs GPU3 vs GPU4_64')
plt.scatter(gpu3m, gpu3p, marker='<', color='r')
plt.scatter(gpu4_64m, gpu4_64p, marker='>', color='yellow')
plt.scatter(libm, libp, marker='s', color='green')
plt.ylabel("Gflops/s")
plt.xlabel("Memory  $log_{2}(MB)$")
plt.grid(axis='y')
plt.legend(['GPU3', 'GPU4', 'BLAS'])
plt.savefig(f'BLAS_vs_GPU3_vs_GPU4_64.png')
# plt.show()

plt.figure(6)
plt.title('BLAS vs GPU5')
plt.scatter(gpu5m, gpu5p, marker='<', color='r')
plt.scatter(libm, libp, marker='s', color='green')
plt.ylabel("Gflops/s")
plt.xlabel("Memory  $log_{2}(MB)$")
plt.grid(axis='y')
plt.legend(['GPU5', 'BLAS'])
plt.savefig(f'BLAS_vs_GPU5.png')
# plt.show()

plt.figure(7)
plt.title('BLAS vs GPUlib vs GPU5')
plt.scatter(gpu5m, gpu5p, marker='s', color='r')
plt.scatter(gpulibm, gpulibp, marker='>', color='yellow')
plt.scatter(libm, libp, marker='<', color='green')
plt.ylabel("Gflops/s")
plt.xlabel("Memory  $log_{2}(MB)$")
plt.grid(axis='y')
plt.legend(['GPU5', 'CUBLAS', 'BLAS'])
plt.savefig(f'BLAS_vs_GPU5_vs_CUBLAS.png')
# plt.show()

plt.figure(8)
plt.title('BLAS vs GPUlib vs GPU5 vs GPU4_32 vs GPU3')
plt.scatter(gpu5m, gpu5p, marker='+', color='r')
plt.scatter(gpu4_32m, gpu4_32p, marker='x', color='blue')
plt.scatter(gpu3m, gpu3p, marker='s', color='black')
plt.scatter(gpulibm, gpulibp, marker='>', color='yellow')
plt.scatter(libm, libp, marker='<', color='green')
plt.ylabel("Gflops/s")
plt.xlabel("Memory  $log_{2}(MB)$")
plt.grid(axis='y')
plt.legend(['GPU5', 'GPU4', 'GPU3', 'CUBLAS', 'BLAS'])
plt.savefig(f'BLAS_vs_CUBLAS_vs_GPU5_vs_GPU4_vs_GPU3.png')
# plt.show()


plt.figure(9)
plt.title('BLAS vs GPU3 vs GPU4_8 vs GPU4_16 vs GPU4_32 vs GPU4_64')
plt.scatter(gpu3m, gpu3p, marker='<', color='r')
plt.scatter(gpu4_8m, gpu4_8p, marker='>', color='blue')
plt.scatter(gpu4_16m, gpu4_16p, marker='>', color='black')
plt.scatter(gpu4_32m, gpu4_32p, marker='>', color='yellow')
plt.scatter(gpu4_64m, gpu4_64p, marker='>', color='pink')
plt.scatter(libm, libp, marker='s', color='green')
plt.ylabel("Gflops/s")
plt.xlabel("Memory  $log_{2}(MB)$")
plt.grid(axis='y')
plt.legend(['GPU3', 'GPU4_8', 'GPU4_16', 'GPU4_32', 'GPU4_64', 'BLAS'])
plt.savefig(f'BLAS_vs_GPU3_vs_GPU4_8_vs_GPU4_16_vs_GPU4_32_vs_GPU4_64.png')
# plt.show()
