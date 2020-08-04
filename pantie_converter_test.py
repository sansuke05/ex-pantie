import os
from PIL import Image, ImageOps
from skimage import io
import skimage.transform as skt
import numpy as np
import matplotlib.pyplot as plt

dreamdir = '.\\img\\'
converted_dir = '.\\converted\\'
mask = io.imread('./mask/mask_chronos.png')


def resize(img, mag):
    return skt.resize(
        img, (np.int(img.shape[0] * mag[0]), np.int(img.shape[1] * mag[1])),
        anti_aliasing=True,
        mode='reflect')


def show(image):
    io.imshow(image)
    io.show()


def affine_transform_by_arr(img,
                            arrx,
                            arry,
                            smoothx=False,
                            smoothy=False,
                            mvx=10,
                            mvy=10):
    # 変形前の座標点を生成
    [r, c, d] = img.shape
    src_cols = np.linspace(0, c, int(np.sqrt(len(arrx))))
    src_rows = np.linspace(0, r, int(np.sqrt(len(arry))))
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    # 必要ならば変形量配列に移動平均を適用する
    if smoothx:
        lx = len(arrx)
        arrx = np.convolve(arrx, np.ones(mvx) / mvx, mode='valid')
        arrx = skt.resize(arrx, (lx, 1), anti_aliasing=True, mode='reflect')[:,
                                                                             0]
    if smoothy:
        ly = len(arry)
        arry = np.convolve(arry, np.ones(mvy) / mvy, mode='valid')
        arry = skt.resize(arry, (ly, 1), anti_aliasing=True, mode='reflect')[:,
                                                                             0]
    # 座標点に変形量を加える
    dst_rows = src[:, 1] + arrx
    dst_cols = src[:, 0] + arry
    dst = np.vstack([dst_cols, dst_rows]).T
    # 区分的アフィン変換を行う
    affin = skt.PiecewiseAffineTransform()
    affin.estimate(src, dst)
    return skt.warp(img, affin)


def convert_chronos_pantie(image):
    pantie = np.array(image)
    patch = np.copy(pantie[-100:-5, 546:, :])
    pantie[-100:, 546:, :] = 0
    patch = skt.resize(patch[::-1, ::-1, :],
                       (patch.shape[0] + 30, patch.shape[1]),
                       anti_aliasing=True,
                       mode='reflect')
    [pr, pc, d] = patch.shape
    pantie[127 - 5:127 - 5 + pr, :pc, :] = np.uint8(patch * 255)

    front = pantie[:350, :300]
    show(front)
    arrx = (np.linspace(0, 1, 25)**2) * 43
    arrx[5:20] += np.sin(np.linspace(0, np.pi, 15)) * 4
    arrx[5:10] += np.sin(np.linspace(0, np.pi, 5)) * 13
    arry = np.zeros(25)
    arrx -= 30
    front = affine_transform_by_arr(front, arrx, arry)
    front = np.uint8(front[::-1, 7:][88:] * 255)
    show(front)

    back = pantie[:350, 300:-10][:, ::-1]
    show(back)
    arrx = (np.linspace(0, 1, 25)**2) * 115
    arrx[2:14] += np.sin(np.linspace(0, np.pi, 12)) * 7
    arry = np.zeros(25)
    arrx -= 70
    back = affine_transform_by_arr(back, arrx, arry)
    back = np.uint8(back[3:, 10:10 + front.shape[1]] * 255)

    pantie = np.concatenate((back, front), axis=0)
    pantie = np.uint8(resize(pantie, [1.55, 1.745]) * 255)
    #pantie = np.bitwise_and(pantie, mask)
    pantie = np.concatenate((pantie[:, ::-1], pantie), axis=1)
    return pantie
    #return Image.fromarray(pantie)


def convert_vroid_pantie(pantie):
    pantie = np.pad(pantie, [(100, 0), (10, 0), (0, 0)], mode='constant')
    arrx = np.zeros(100)
    arrx[10:50] = (np.sin(np.linspace(0, 1 * np.pi, 100))[20:60] * 30)
    arrx[50:] = -(np.sin(np.linspace(0, 1 * np.pi, 100))[50:] * 15)
    arrx[40:60] += (np.sin(np.linspace(0, 1 * np.pi, 100))[40:60] * 15)
    arrx[00:10] -= (np.sin(np.linspace(0, 1 * np.pi, 100))[50:60] * 35)

    arry = (np.sin(np.linspace(0, 0.5 * np.pi, 100)) * 70)
    arry[10:30] -= (np.sin(np.linspace(0, 1 * np.pi, 100)) * 20)[50:70]

    tf_pantie = affine_transform_by_arr(pantie,
                                        arrx,
                                        arry,
                                        smoothx=True,
                                        mvx=30)
    tf_pantie = tf_pantie[60:430, 16:-80]
    vroid_pantie = np.concatenate([tf_pantie[:, ::-1], tf_pantie], axis=1)
    return (vroid_pantie * 255).astype(np.uint8)


def convert_quiche_pantie(img):
    img = img[0:-1, 8:-1]
    nbody_pantie = np.concatenate([img[:, ::-1], img], axis=1)
    return nbody_pantie


#for num in range(10):
#    pantie_name = f'{dreamdir}{num+1:04d}.png'
#    img = Image.open(pantie_name)
#    pantie = convert_nbody(img)
#    pantie.save(f'{converted_dir}{num+1:04d}.png')

# ribbon.resize((ribbon.width*m, ribbon.height*m), resample=Image.BICUBIC)
available_models = ['quiche', 'vroid', 'chronos']
available_panties = sorted(os.listdir(dreamdir))


def load_pantie(num):
    while f'{num:04d}.png' not in available_panties:
        try:
            num = int(input('Please input favorite pantie number:'))
        except ValueError:
            print('Input number is not Integar value')
    return (num, io.imread(f'{dreamdir}{num:04d}.png'))


def select_model(model):
    while model not in available_models:
        try:
            model = str(input('Please input convert target model name:'))
        except:
            print('An error has occurred.')
            exit(1)
    return model


def save_pantie(num, model, pantie):
    io.imsave(f'{converted_dir}{model}_{num:04d}.png', pantie)


def patcher(num=0, model=None):
    converted_pantie = None
    # Input check
    num, pantie = load_pantie(num)
    if model is None:
        model = select_model(model)
    # Convert
    if model == available_models[0]:
        converted_pantie = convert_quiche_pantie(pantie)
    elif model == available_models[1]:
        converted_pantie = convert_vroid_pantie(pantie)
    else:
        converted_pantie = convert_chronos_pantie(pantie)
    # Output
    #save_pantie(num, model, converted_pantie)
    io.imshow(converted_pantie)
    io.show()


# 以下は想定される実行形式(適宜コメントアウトしながらテストしてください)
patcher()
#patcher(num=2)
#patcher(model='vroid')
#patcher(num=2, model='quiche')