import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver


def astype_int24(array):
    array = np.left_shift(array, 8)
    array = np.right_shift(array, 8)
    return array


@qpu
def kernel(asm):
    # VPM使うことは確定なので最初にセットアップしておく
    setup_vpm_write()

    # [メモリ->VPM]:16要素*1行分読み込む
    setup_dma_load(nrows = 1)
    start_dma_load(uniform)
    wait_dma_load()

    # [VPM->レジスタ]:16要素*1行分読み込む
    setup_vpm_read(nrows = 1)
    mov(r0, vpm)

    fadd(r0, r0, 1.0)

    # 以下2行をどこに挟むかによって結果が変わる
    jmp(L.end)
    nop(); nop(); nop();

    fadd(r0, r0, 1.0)
    fadd(r0, r0, 1.0)
    fadd(r0, r0, 1.0)
    fadd(r0, r0, 1.0)

    L.end
    mov(vpm, r0)

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    list_a = drv.alloc(16, 'float32')
    list_a[:] = 0.0

    out = drv.alloc(16, 'float32')

    print(' list_a '.center(80, '='))
    print(list_a)

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[list_a.address, out.address]
    )

    print(' out '.center(80, '='))
    print(out)