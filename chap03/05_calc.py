import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def kernel(asm):
    # [メモリ->VPM]:16要素*2行分読み込む
    setup_dma_load(nrows = 2)
    start_dma_load(uniform)
    wait_dma_load()

    # [VPM->レジスタ]:16要素*2行分読み込む
    setup_vpm_read(nrows = 2)
    mov(r0, vpm)
    mov(r1, vpm)

    # 演算:r0とr1を足してvpmに書き込む
    setup_vpm_write()
    fadd(vpm, r0, r1)

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    list_a = np.full(16, 1.0).astype('float32')
    list_b = np.full(16, 2.0).astype('float32')
    
    # 配列の結合
    inp = drv.copy(np.r_[list_a, list_b])
    out = drv.alloc(16, 'float32')

    print(' list_a '.center(80, '='))
    print(list_a)
    print(' list_b '.center(80, '='))
    print(list_b)

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[inp.address, out.address]
    )

    print(' out '.center(80, '='))
    print(out)