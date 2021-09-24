#coding:utf-8
import concurrent.futures
import sys
import io
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from videocore.assembler import qpu
from videocore.driver import Driver

from time import sleep, clock_gettime, CLOCK_MONOTONIC
import picamera.array
from picamera import PiCamera

sys.path.append("../00_utils/")
import hdmi
import camera
from fps import FPS

def setCamera(w, h):
  camera = PiCamera()
  camera.resolution = (w, h)

  return camera

def mask(idx):
    values = [1]*16
    values[idx] = 0
    return values

@qpu
def histogram(asm):
    IN_ADDR   = 0 #インデックス
    OUT_ADDR  = 1
    IO_ITER   = 2
    THR_ID    = 3
    THR_NM    = 4
    COMPLETED = 0 #セマフォ用

    R = 30

    ldi(null,mask(IN_ADDR),set_flags=True)  # r2にuniformを格納
    mov(r2,uniform,cond='zs')
    ldi(null,mask(OUT_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(IO_ITER),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_ID),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_NM),set_flags=True)
    mov(r2,uniform,cond='zs')

    imul24(r3,element_number,4)
    rotate(broadcast,r2,-IN_ADDR)
    iadd(r0,r5,r3) # r0:IN_ADDR

    ldi(r1, 0) # r1 にヒストグラムの値を用意する

    L.loop

    ldi(broadcast,16*4)
    for i in range(R):
        mov(tmu0_s,r0) #ra
        iadd(r0, r0, r5,sig='load tmu0')
        mov(ra[i], r4) \
          .mov(tmu0_s,r0) #rb
        iadd(r0, r0, r5,sig='load tmu0')
        mov(rb[i], r4)


    for histogram in range(16):
        ldi(r3, 0) # r3 で計算する
        ldi(broadcast, 16) # 16 段階なので 256/16 で 16 を r5 に用意しておく

        # レジスタファイルa, bを全部見る
        for i in range(R):
            isub(ra[i], ra[i], r5, set_flags=True) # 16段階なので16を引く
            iadd(r3, r3, 1, cond='ns') # マイナスになっている要素だけ r3 に 1 を足す
            isub(rb[i], rb[i], r5, set_flags=True)
            iadd(r3, r3, 1, cond='ns')

        # r3 の16要素を r1 のhistogram番目の要素に足していく
        ldi(null, mask(histogram), set_flags=True) # histogram 番目の要素だけ Z フラグを立てる
        for i in range(16):
            rotate(broadcast, r3, -i, set_flags=False) # r3 のi番目の要素を r5 に書き込む
            iadd(r1, r1, r5, cond='zs', set_flags=False) # r1 の histogram 番目の要素 と r5(r3 のi番目の要素)を足す


    ldi(null,mask(IO_ITER),set_flags=True)
    isub(r2,r2,1,cond='zs')
    jzc(L.loop)
    nop()
    nop()
    nop()


    rotate(broadcast,r2,-OUT_ADDR)
    mutex_acquire()
    setup_vpm_write(mode='32bit horizontal',Y=0,X=0)

    mov(vpm,r1)

    setup_dma_store(mode='32bit horizontal',Y=0,nrows=1)
    start_dma_store(r5)
    wait_dma_store()

    mutex_release()


#====semaphore=====
    sema_up(COMPLETED)
    rotate(broadcast,r2,-THR_ID)
    iadd(null,r5,-1,set_flags=True)
    jzc(L.skip_fin)
    nop()
    nop()
    nop()
    rotate(broadcast,r2,-THR_NM)
    iadd(r0, r5, -1,set_flags=True)
    L.sem_down
    jzc(L.sem_down)
    sema_down(COMPLETED)    # すべてのスレッドが終了するまで待つ
    nop()
    iadd(r0, r0, -1)

    interrupt()

    L.skip_fin

    exit(interrupt=False)


with Driver() as drv:

    DISPLAY_W, DISPLAY_H = hdmi.getResolution()

    # 画像サイズ
    W=320
    H=360

    ALIGNMENT = 32

    ALIGNED_W = math.ceil(W / ALIGNMENT) * ALIGNMENT
    ALIGNED_H = math.ceil(H / ALIGNMENT) * ALIGNMENT
    W = ALIGNED_W

    print(W, H)
    print(DISPLAY_W, DISPLAY_H)

    WINDOW_W = DISPLAY_W // 3
    WINDOW_H = min(int((WINDOW_W / W) * H), DISPLAY_H)

    # cameraセットアップ
    cam = camera.setCamera(ALIGNED_W, ALIGNED_H)
    cam.framerate = 30
    overlay_dstimg = camera.PiCameraOverlay(cam, 3)
    overlay_dstimg2 = camera.PiCameraOverlay(cam, 4)
    cam.start_preview(fullscreen=False, window=(0, 0, WINDOW_W, WINDOW_H))

    # 画面のクリア
    back_img = Image.new('RGBA', (DISPLAY_W, DISPLAY_H), 0)
    hdmi.printImg(back_img, *hdmi.getResolution(), hdmi.PUT)

    n_threads=12
    SIMD=16
    R=30
    HISTOGRAM_ELEMS=16

    th_H    = int(H/n_threads) #1スレッドの担当行
    th_ele  = int(H*W/n_threads) #1スレッドの担当要素
    io_iter = int(th_ele/(R*2*SIMD)) #何回転送するか
    th_ele = io_iter * R * 2 * SIMD

    print(th_H, th_ele, io_iter)

    IN  = drv.alloc((H,W),'int32')
    OUT = drv.alloc((n_threads, SIMD),'int32')
    OUT[:] = 0

    uniforms=drv.alloc((n_threads,5),'uint32')
    for th in range(n_threads):
      uniforms[th,0]=IN.address + (th_ele * 4 * th)
      uniforms[th,1]=OUT.address + (SIMD * 4 * th)
    uniforms[:,2]=int(io_iter)
    uniforms[:,3]=np.arange(1,(n_threads+1))
    uniforms[:,4]=n_threads

    code=drv.program(histogram)

    def capture_thread():
      input_img_RGB = camera.capture2PIL(cam, stream, (ALIGNED_W, ALIGNED_H))
      input_img = input_img_RGB.convert('L')
      pil_img = input_img.resize((W, H))

      IN[:] = np.asarray(pil_img, dtype='uint32')[:]

    def gpu_thread():
      drv.execute(
          n_threads= n_threads,
          program  = code,
          uniforms = uniforms
      )

    def fps_thread():
      print(f'{fps.update():.3f} FPS')

    graph_width = W / HISTOGRAM_ELEMS

    def image_out_thread():
      draw_img = Image.new('RGB', (W, H), (0, 0, 0))
      draw = ImageDraw.Draw(draw_img)
      histogram_value = [0] * HISTOGRAM_ELEMS
      out_data = OUT.astype(np.int32)

      for th in range(n_threads):
        for i in range(SIMD):
          histogram_value[i] += out_data[th, i]

      for i in range(15, 0, -1):
        histogram_value[i] -= histogram_value[i-1]

      for i in range(HISTOGRAM_ELEMS):
        draw.rectangle((graph_width * i, H - (histogram_value[i] / W), graph_width * (i+1), H), fill=(255, 255, 255))

      out_img = draw_img
      overlay_dstimg.OnOverlayUpdated(out_img, format='rgb', fullscreen=False, window=(WINDOW_W, 0, WINDOW_W, WINDOW_H))

    def info_out_thread():
      draw_img = Image.new('L', (W, H), 0)
      hdmi.addText(draw_img, *(10, 32 * 0), "Raspberry Pi VC4")
      hdmi.addText(draw_img, *(10, 32 * 2), f'Binarization')
      hdmi.addText(draw_img, *(10, 32 * 3), f'{H}x{W}')
      hdmi.addText(draw_img, *(10, 32 * 5), f'{fps.get():.3f} FPS')

      draw_img = draw_img.convert('RGB')
      overlay_dstimg2.OnOverlayUpdated(draw_img, format='rgb', fullscreen=False, window=(WINDOW_W*2, 0, WINDOW_W, WINDOW_H))


    try:
      fps = FPS()
      stream = io.BytesIO()
      with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
          concurrent.futures.wait([
            executor.submit(fps_thread),
            executor.submit(capture_thread),
            executor.submit(gpu_thread),
            executor.submit(image_out_thread),
            executor.submit(info_out_thread),
          ])

    except KeyboardInterrupt:
      # Ctrl-C を捕まえた！
      print('\nCtrl-C is pressed, end of execution!')
      cam.stop_preview()
      cam.close()
