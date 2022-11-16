# -*- coding:utf-8 -*-
from PIL import Image
from ssd_tools.ssd_utils import BBoxUtility
from ssd_tools.ssd import SSD300
import cv2
import argparse
import os
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import gc
import glob
import json
from keras import backend as K
K.clear_session()


os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
np.set_printoptions(suppress=True)

# パラメータ
batch_size = 10
NUM_CLASSES = 2
input_shape = (300, 300, 3)


model = SSD300(input_shape, num_classes=NUM_CLASSES)
bbox_util = BBoxUtility(NUM_CLASSES)


dpiinfo = {}


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def resize_pil(pil_img, short):
    w, h = pil_img.size
    if w < h:
        h = int(h*short/w+0.5)
        w = short
    else:
        w = int(w*short/h+0.5)
        h = short
    return (pil_img.resize((w, h)))


def divide_facing_page(input, input_path=None, output="NO_DUMP",
                    left='_01', right='_02', single='_00', ext='.jpg',
                    quality=100,  # output jpeg quality
                    short=None,
                    debug=False,
                    log='trim_pos.tsv',
                    conf_th=0.2,
                    with_cli=False):
    if not with_cli:
        model.load_weights(os.path.join('ssd_tools', 'weights.hdf5'), by_name=True)

    if log:
        if not os.path.exists(log):
            with open(log, mode='a') as f:
                line = 'image_name\ttrimming_x\n'
                f.write(line)

    imglist = []
    filenames = []
    if with_cli:
        if type(input) is np.ndarray:
            imglist = [input]
        elif type(input) is not list:
            raise ValueError(
                'input for divide_facing_page_with_cli must be np.array or list.')

        if type(input_path) is str:
            filenames = [input_path]
        elif type(input_path) is not list:
            raise ValueError(
                'input_path for divide_facing_page_with_cli must be str or list.')
        else:
            filenames = input_path

    else: # without_cli
        if os.path.isdir(input):
            imgpathlist = list(glob.glob(os.path.join(input, "*")))
        else:
            imgpathlist = [input]
        for imgpath in imgpathlist:
            imglist.append(cv2.imread(imgpath, cv2.IMREAD_COLOR))
            filenames.append(os.path.basename(imgpath))

    cnt = 0
    while cnt < len(imglist):
        inputs = []
        images = []
        for cv_img in imglist[cnt:min(cnt+batch_size, len(imglist))]:
            img = image.img_to_array(cv2pil(cv_img).resize((300, 300)))
            images.append(cv_img)     # original size images
            inputs.append(img.copy())  # resized to (300, 300)

        inputs = preprocess_input(np.array(inputs))
        preds = model.predict(inputs, batch_size=1, verbose=1)
        results = bbox_util.detection_out(preds)
        # results[i][b, p] ... i: image index; b: bbox index; p: [label, confidence, xmin, ymin, xmax, ymax]
        
        for i, cvimg in enumerate(images):
            if len(results[i]) == 0:
                top_conf = 0.0
            else:
                top_conf = results[i][0, 1]
                top_xmin = results[i][0, 2]
                top_xmax = results[i][0, 4]
            print('img {} top conf: {}'.format(i, top_conf))

            div_x = 0
            basename, ext_ori = os.path.splitext(
                os.path.basename(filenames[i + cnt]))
            if ext == "SAME":
                ext = ext_ori

            if top_conf <= conf_th:
                # save log
                if log:
                    with open(log, mode='a') as f:
                        line = '{}\t{}\n'.format(basename+single+ext, 0)
                        f.write(line)
                if with_cli:
                    return [cvimg]
                elif output != "NO_DUMP":
                    im = cv2pil(cvimg)
                    if short:
                        im = resize_pil(im, short)
                    im.save(os.path.join(output, basename+single+ext),
                            dpi=(dpiinfo["width_dpi"], dpiinfo["height_dpi"]), quality=100)

            else:
                xmin = int(round(top_xmin * cvimg.shape[1]))
                xmax = int(round(top_xmax * cvimg.shape[1]))
                div_x = (xmin+xmax)//2
                # save log
                if log:
                    with open(log, mode='a') as f:
                        line = '{}\t{}\n'.format(basename+left+ext, div_x-1)
                        f.write(line)
                        line = '{}\t{}\n'.format(basename+right+ext, div_x)
                        f.write(line)
                # save split images
                if with_cli:
                    return [cvimg[:, :div_x, :], cvimg[:,  div_x:, :]]
                else:
                    if output != "NO_DUMP":
                        im1 = cv2pil(cvimg[:, :div_x, :])
                        im2 = cv2pil(cvimg[:,  div_x:, :])

                        if short:
                            im1 = resize_pil(im1, short)
                            im2 = resize_pil(im2, short)
                        im1.save(os.path.join(output, basename+left+ext),
                                dpi=(dpiinfo["width_dpi"], dpiinfo["height_dpi"]),
                                quality=quality)
                        im2.save(os.path.join(output, basename+right+ext),
                                dpi=(dpiinfo["width_dpi"], dpiinfo["height_dpi"]),
                                quality=quality)
                    # (debug) add bounding box and gutter line to the image
                    if debug:
                        for k in range(len(results[i])):
                            xmin = int(round(results[i][k, 2] * cvimg.shape[1]))
                            ymin = int(round(results[i][k, 3] * cvimg.shape[0]))
                            xmax = int(round(results[i][k, 4] * cvimg.shape[1]))
                            ymax = int(round(results[i][k, 5] * cvimg.shape[0]))
                            print(results[i][k, :])
                            bgr = (0, 0, 255)
                            t = 2
                            if k == 0:
                                if top_conf > 0.2:
                                    t = 5
                                cv2.line(cvimg, ((xmin+xmax)//2, 0), ((xmin+xmax)//2, cvimg.shape[0]),
                                        color=(255, 0, 0), thickness=t)
                            cv2.rectangle(cvimg, (xmin, ymin),
                                        (xmax, ymax), bgr, thickness=t)
                        im = cv2pil(cvimg)
                        os.makedirs(output+'_rect', exist_ok=True)
                        im.save(os.path.join(output+'_rect', basename+ext),
                                dpi=(dpiinfo["width_dpi"], dpiinfo["height_dpi"]),
                                quality=quality)

        cnt += batch_size

        del inputs, images
        gc.collect()


def divide_facing_page_with_cli(input, input_path,
                                left='_01', right='_02', single='_00', ext='.jpg',
                                quality=100,  # output jpeg quality
                                short=None,
                                conf_th=0.2,
                                log='trim_pos.tsv'):

    return divide_facing_page(input=input,
                              input_path=input_path,
                              output="NO_DUMP",
                              left=left, right=right, single=single, ext=ext,
                              quality=quality,  # output jpeg quality
                              short=short,
                              debug=False,
                              log=log,
                              conf_th=conf_th,
                              with_cli=True)


def load_weightfile(model_path):
    model.load_weights(model_path, by_name=True)


def parse_args():
    usage = 'python3 {} [-i INPUT] [-o OUTPUT] [-l LEFT] [-r RIGHT] [-s SINGLE] \
             [-e EXT] [-q QUALITY]'.format(__file__)
    argparser = argparse.ArgumentParser(
        usage=usage,
        description='Divide facing images at the gutter',
        formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument(
        '-i',
        '--input',
        default='inference_input',
        help='input image file or directory path\n'
             '(default: inference_input)',
        type=str)
    argparser.add_argument(
        '-o',
        '--out',
        default='inference_output',
        help='directory path (default: inference_output)\n'
             'if OUT is "NO_DUMP", no images is output',
        type=str)
    argparser.add_argument(
        '-l',
        '--left',
        default='_01',
        help='file name footer of left side page image to be output\n'
             'e.g) input image:  input.jpg, LEFT: _01(default)\n'
             '     output image: input_01.jpg',
        type=str)
    argparser.add_argument(
        '-r',
        '--right',
        default='_02',
        help='file name footer of right side page image to be output\n'
             'e.g) input image:  input.jpg, RIGHT: _02(default)\n'
             '     output image: input_02.jpg',
        type=str)
    argparser.add_argument(
        '-s',
        '--single',
        default='_00',
        help='file name footer of the image with no detected gutters to be output\n'
             'e.g) input image:  input.jpg, SINGLE: _00(default)\n'
             '     output image: input_00.jpg',
        type=str)
    argparser.add_argument(
        '-e',
        '--ext',
        default='.jpg',
        help='output image extension. default: .jpg \n'
             'if EXT is \"SAME\", the same extension as the input image will be used.',
        type=str)
    argparser.add_argument(
        '-q', '--quality',
        default=100,
        dest='quality',
        help='output jpeg image quality.\n'
             '1 is worst quality and smallest file size,\n'
             'and 100 is best quality and largest file size.\n'
             '[1, 100], default: 100',
        type=int)
    argparser.add_argument(
        '--short',
        default=None,
        dest='short',
        help='the length of the short side of the output image.',
        type=int)
    argparser.add_argument(
        '--debug',
        action='store_true')
    argparser.add_argument(
        '-lg', '--log',
        default=None,
        help='path of the tsv file that records the split x position'
             'output format:'
             'file name <tab> trimming_x',
        type=str)

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(os.path.join('ssd_tools', 'dpiconfig.json'))as f:
        dpiinfo = json.load(f)

    if args.out != "NO_DUMP":
        os.makedirs(args.out, exist_ok=True)
    else:
        print('Not dump split images')

    if args.debug:
        print('Run in debug mode: dump images added bounding box and gutter lines')
    if args.log is not None:
        print('Export estimated gutter position to {}'.format(args.log))

    divide_facing_page(input=args.input,
                       output=args.out,
                       left=args.left,
                       right=args.right,
                       single=args.single,
                       ext=args.ext,
                       quality=args.quality,
                       short=args.short,
                       debug=args.debug,
                       log=args.log)
