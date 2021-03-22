import numpy as np
from bresenham import bresenham
import scipy.ndimage
from PIL import Image
from matplotlib import pyplot as plt
import torch
from utils import to_normal_strokes


def mydrawPNG(vector_image, Side = 256):

    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    stroke_bbox = []
    stroke_cord_buffer = []
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)
        stroke_cord_buffer.extend([list(i) for i in cordList])

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        if vector_image[i, 2] == 1:
            min_x = np.array(stroke_cord_buffer)[:, 0].min()
            min_y = np.array(stroke_cord_buffer)[:, 1].min()
            max_x = np.array(stroke_cord_buffer)[:, 0].max()
            max_y = np.array(stroke_cord_buffer)[:, 1].max()
            stroke_bbox.append([min_x, min_y, max_x, max_y])
            stroke_cord_buffer = []

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    #utils.image_boxes(Image.fromarray(raster_image).convert('RGB'), stroke_bbox).show()
    return raster_image, stroke_bbox


def preprocess(sketch_points, side = 256.0):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([256, 256])
    sketch_points[:,:2] = sketch_points[:,:2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def rasterize_Sketch(sketch_points):
    sketch_points = preprocess(sketch_points)
    raster_images, _ = mydrawPNG(sketch_points)
    return raster_images

def to_delXY(sketch):
    new_skech = sketch.copy()
    new_skech[:-1,:2]  = new_skech[1:,:2] - new_skech[:-1,:2]
    new_skech[:-1, 2] = new_skech[1:, 2]
    return new_skech[:-1,:]


def to_Absolute(sketch, start_point=(0,0)):
    new_skech = sketch.copy()
    origin = np.array([start_point[0], start_point[1], 0])
    new_skech = np.vstack((origin, new_skech))  # add the implicit origin
    new_skech[:, :2] = np.cumsum(new_skech[:, :2], axis=0)
    return new_skech



def toStrokeList(sketch):
    return np.split(sketch, np.where(sketch[:, 2])[0] + 1, axis=0)[:-1]


def to_FivePoint(sketch, max_seq_len=130):
    len_seq = len(sketch[:, 0])
    new_seq = np.zeros((max_seq_len, 5))
    new_seq[0:len_seq, :2] = sketch[:, :2]
    new_seq[0:len_seq, 3] = sketch[:, 2]
    new_seq[0:len_seq, 2] = 1 - new_seq[0:len_seq, 3]
    new_seq[(len_seq - 1):, 4] = 1
    new_seq[(len_seq - 1), 2:4] = 0
    return new_seq


def to_stroke_list(sketch):
    ## sketch: an `.npz` style sketch from QuickDraw
    sketch = np.vstack((np.array([0, 0, 0]), sketch))
    sketch[:,:2] = np.cumsum(sketch[:,:2], axis=0)

    # range normalization
    xmin, xmax = sketch[:,0].min(), sketch[:,0].max()
    ymin, ymax = sketch[:,1].min(), sketch[:,1].max()

    sketch[:,0] = ((sketch[:,0] - xmin) / float(xmax - xmin)) * (255.-60.) + 30.
    sketch[:,1] = ((sketch[:,1] - ymin) / float(ymax - ymin)) * (255.-60.) + 30.
    sketch = sketch.astype(np.int64)

    stroke_list = np.split(sketch[:,:2], np.where(sketch[:,2])[0] + 1, axis=0)

    if stroke_list[-1].size == 0:
        stroke_list = stroke_list[:-1]

    if len(stroke_list) == 0:
        stroke_list = [sketch[:, :2]]
        # print('error')

    return stroke_list


def rasterize_relative(stroke_list, fig, xlim=[0,255], ylim=[0,255]):
    # Usage:  image = rasterize_relative(to_stroke_list(to_normal_strokes(data['relative_fivePoint'][0])), canvas)
    # fig = plt.figure(frameon=False, figsize=(2.56, 2.56))
    for stroke in stroke_list:
        stroke = stroke[:,:2].astype(np.int64)
        plt.plot(stroke[:,0], stroke[:,1])
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.gca().invert_yaxis(); plt.axis('off')
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer._renderer)
    plt.gca().cla()
    X = X[...,:3] / 255.
    X = X.mean(2)
    X[X == 1.] = 0.; X[X > 0.] = 255.0
    sketch_img = Image.fromarray(X).convert('RGB')
    # plt.close(fig)
    return sketch_img

def mydrawPNG_from_list(vector_image, Side = 256):

    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)

    for stroke in vector_image:
        initX, initY = int(stroke[0, 0]), int(stroke[0, 1])

        for i_pos in range(1, len(stroke)):
            cordList = list(bresenham(initX, initY, int(stroke[i_pos, 0]), int(stroke[i_pos, 1])))
            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] <= Side and cord[1] <= Side):
                    raster_image[cord[1], cord[0]] = 255.0
                else:
                    print('error')
            initX, initY =  int(stroke[i_pos, 0]), int(stroke[i_pos, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0

    return Image.fromarray(raster_image).convert('RGB')


def batch_rasterize_relative(sketch):

    def to_stroke_list(sketch):
        ## sketch: an `.npz` style sketch from QuickDraw
        sketch = np.vstack((np.array([0, 0, 0]), sketch))
        sketch[:, :2] = np.cumsum(sketch[:, :2], axis=0)

        # range normalization
        xmin, xmax = sketch[:, 0].min(), sketch[:, 0].max()
        ymin, ymax = sketch[:, 1].min(), sketch[:, 1].max()

        sketch[:, 0] = ((sketch[:, 0] - xmin) / float(xmax - xmin)) * (255. - 60.) + 30.
        sketch[:, 1] = ((sketch[:, 1] - ymin) / float(ymax - ymin)) * (255. - 60.) + 30.
        sketch = sketch.astype(np.int64)

        stroke_list = np.split(sketch[:, :2], np.where(sketch[:, 2])[0] + 1, axis=0)

        if stroke_list[-1].size == 0:
            stroke_list = stroke_list[:-1]

        if len(stroke_list) == 0:
            stroke_list = [sketch[:, :2]]
            # print('error')
        return stroke_list

    batch_redraw = []
    if sketch.shape[-1] == 5:
        for data in sketch:
            # image = rasterize_relative(to_stroke_list(to_normal_strokes(data.cpu().numpy())), canvas)
            image = mydrawPNG_from_list(to_stroke_list(to_normal_strokes(data.cpu().numpy())))
            batch_redraw.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
    elif sketch.shape[-1] == 3:
        for data in sketch:
            # image = rasterize_relative(to_stroke_list(data.cpu().numpy()), canvas)
            image = mydrawPNG_from_list(to_stroke_list(data.cpu().numpy()))
            batch_redraw.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))

    return torch.stack(batch_redraw).float()



# def rasterize_relative_V(stroke_list, fig, xlim=[0,255], ylim=[0,255]):
#     # Usage:  image = rasterize_relative(to_stroke_list(to_normal_strokes(data['relative_fivePoint'][0])), canvas)
#
#     for stroke in stroke_list:
#         stroke = stroke[:,:2].astype(np.int64)
#         plt.plot(stroke[:,0], stroke[:,1])
#     plt.xlim(*xlim)
#     plt.ylim(*ylim)
#
#     plt.gca().invert_yaxis(); plt.axis('off')
#     fig.canvas.draw()
#
#     # w, h = fig.canvas.get_width_height()
#     X = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
#     X.shape = (256, 256, 4)
#
#     # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#     X = np.roll(X, 3, axis=2)
#     X = Image.frombytes("RGBA", (256, 256), X.tostring())
#     X = np.array(X.convert('RGB')).mean(2)/255.
#     X[X == 1.] = 0.; X[X > 0.] = 255.0
#     sketch_img = Image.fromarray(X).convert('RGB')
#     return sketch_img

    #
    #
    #
    #
    # X = np.array(fig.canvas.renderer._renderer)
    # plt.gca().cla()
    # X = X[...,:3] / 255.
    # X = X.mean(2)
    # X[X == 1.] = 0.; X[X > 0.] = 255.0
    # sketch_img = Image.fromarray(X).convert('RGB')
    # plt.close(fig)
    # return sketch_img

# def toAbsolute(sketch):
#     new_skech = sketch.copy()
#     origin = np.array([0, 0, 0])
#     new_skech = np.vstack((origin, new_skech))  # add the implicit origin
#     new_skech[:, :2] = np.cumsum(new_skech[:, :2], axis=0)
#     return new_skech
# def to_delXY(sketch):
#     new_skech = sketch.copy()
#     new_skech[:,:2] = new_skech[:,:2] - new_skech[0,:2]
#     new_skech[1:,:2] -= new_skech[:-1,:2]
#     new_skech = new_skech[1:,:]
#     return new_skech
# stroke_list = toStrokeList(sketch_abs) # convenient structure for drawing
# for stroke in stroke_list:
#     stroke = stroke[:,:-1]
#     plt.plot(stroke[:,0], stroke[:,1])
# plt.axis('off')
# plt.gca().invert_yaxis()
# plt.show()



