import pickle
import os
import numpy as np
from bresenham import bresenham
from rasterize import rasterize_Sketch
from PIL import Image
from rdp import rdp

if __name__ == '__main__':
    coordinate_path = os.path.join('/home/media/On_the_Fly/Code_ALL/Final_Dataset/ShoeV2/ShoeV2_Coordinate')
    # coordinate_path = r'E:\sketchPool\Sketch_Classification\TU_Berlin'
    with open(coordinate_path, 'rb') as fp:
        Coordinate = pickle.load(fp)

    # for key in Coordinate.keys():
    key_list = list(Coordinate.keys())

    for i_rdp in [3, 4]: #reversed(range(11)):
        rdp_simplified = {}
        max_points_old = []
        max_points_new = []

        if not os.path.exists(str(i_rdp)):
            os.makedirs(str(i_rdp))

        for num, key in enumerate(key_list):

            print(i_rdp, num, key)
            sketch_points = Coordinate[key]
            sketch_points_orig = sketch_points

            sketch_points = sketch_points.astype(np.float)
            # sketch_points[:, :2] = sketch_points[:, :2] / np.array([800, 800])
            # sketch_points[:, :2] = sketch_points[:, :2] * 256
            sketch_points = np.round(sketch_points)

            all_strokes = np.split(sketch_points, np.where(sketch_points[:, 2])[0] + 1, axis=0)[:-1]

            max_points_old.append(sketch_points_orig.shape)

            sketch_img_orig = rasterize_Sketch(sketch_points_orig)
            sketch_img_orig = Image.fromarray(sketch_img_orig).convert('RGB')
            # sketch_img_orig.show()
            sketch_img_orig.save(str(i_rdp) + '/' + str(num)+'.jpg')

            sketch_points_sampled_new = []
            for stroke in all_strokes:
                stroke_new = rdp(stroke[:, :2], epsilon=i_rdp, algo="iter")
                stroke_new = np.hstack((stroke_new, np.zeros((stroke_new.shape[0], 1))))
                stroke_new[-1, -1] = 1.
                # print(stroke_new.shape, stroke.shape)
                sketch_points_sampled_new.append(stroke_new)
            sketch_points_new = np.vstack(sketch_points_sampled_new)

            max_points_new.append(sketch_points_new.shape[0])
            # print(sketch_points_orig.shape, sketch_points_new.shape)


            sketch_img_orig = rasterize_Sketch(sketch_points_new)
            sketch_img_orig = Image.fromarray(sketch_img_orig).convert('RGB')
            # sketch_img_orig.show()
            sketch_img_orig.save(str(i_rdp) + '/' + str(num)+'Low_.jpg')

            # if sketch_points_new.shape[0] > 200:
            #     combined_image = np.concatenate(( sketch_img_orig, sketch_img_rdp), axis=1)
            #     combined_image = Image.fromarray(combined_image).convert('RGB')
            #     combined_image.save('./saved_folder2/image_' + str(num) + '_@'  +
            #                         str(sketch_points_orig.shape[0]) +
            #                         '_' +
            #                         str(sketch_points_new.shape[0]) + '.jpg')

            rdp_simplified[key] = sketch_points_new

            # print(sketch_img.shape)

        print('Max number of Points Old: {}'.format(max(max_points_old)))
        print('Max number of Points New: {}'.format(max(max_points_new)))

        with open('ShoeV2_RDP_' + str(i_rdp), 'wb') as fp:
            pickle.dump(rdp_simplified, fp)