import h5py
from pathlib import Path
import cv2
import scipy.io as sio
import argparse, json, re, shutil
from loguru import logger
import numpy as np
# from eyes_features import crop_two_eyes


#default dataset_path
dataset_path = '/home/data/wjc_data/xgaze_224'
outer_dataset_path = '/home/data/wjc_data/xgaze_224_prepare_two'

val_group = [3, 32, 33, 48, 52, 62, 80, 88, 101, 109 ]

#argparse
parser = argparse.ArgumentParser(description='iTracker-xgaze_dataset-prepare')
parser.add_argument('--dataset_path', help='Source dataset path which will to be prepared!')
parser.add_argument('--outer_dataset_path', help='where to write the output files')
args = parser.parse_args()


def main():
    #check the args
    if args.dataset_path is None:
        args.dataset_path = dataset_path
    if args.outer_dataset_path is None:
        args.outer_dataset_path = outer_dataset_path

    #cheak the path
    real_dataset = Path(args.dataset_path)
    real_outer_dataset = Path(args.outer_dataset_path)
    if not real_dataset.is_dir():
        raise RuntimeError('invalid path!')

    #prepare output data path
    real_outer_dataset = prepareOuterPath(real_outer_dataset, clear=1)
    landmark_dir = Path('/home/data/wjc_data/xgaze_224_prepare/xgaze_landmark')

    #define logger
    logger.add(real_outer_dataset.joinpath('train_prepare.log'), filter=lambda x: 'train' in x['message'] or 'validate' in x['message'])
    logger.add(real_outer_dataset.joinpath('test_prepare.log'), filter=lambda x: 'test' in x['message'])

    #define a file to store the train information
    meta_train = {
        'subject': [],
        'frameIndex': [],
        'face_gaze_direction': [],
        'is_equalization': []
    }

    meta_validate = {
        'subject': [],
        'frameIndex': [],
        'face_gaze_direction': [],
        'is_equalization': []
    }

    # define a file to store the test information
    meta_test = {
        'subject': [],
        'frameIndex': [],
        'is_equalization': []
    }

    train_dataset = real_dataset / 'train'
    test_dataset = real_dataset / 'test'
    train_outer_dataset = real_outer_dataset / 'train'
    test_outer_dataset = real_outer_dataset / 'test'
    val_outer_dataset = real_outer_dataset / 'val'
    train_landmarks = landmark_dir / 'train'
    test_landmarks = landmark_dir / 'test'

    for train_file in train_dataset.iterdir():
        with h5py.File(train_file) as train:
            frame = int(re.match('subject(\d{4})$', train_file.stem).group(1))
            if frame in val_group:
                subject_path = val_outer_dataset / train_file.stem
                mode = 'validate'
            else:
                subject_path = train_outer_dataset / train_file.stem
                mode = 'train'
            subject_path.mkdir(parents=True)
            subject_face_path = subject_path / 'face'
            subject_left_eye_path = subject_path / 'left_eye'
            subject_right_eye_path = subject_path / 'right_eye'
            subject_face_path.mkdir()
            subject_left_eye_path.mkdir()
            subject_right_eye_path.mkdir()
            count = 0
            train_ldmk = train_landmarks / train_file.name
            with h5py.File(train_ldmk) as ldmk:
                for image_num in range(0, train['face_patch'].shape[0]):
                # for image_num in range(0, 2):
                    print('[{}/{}] train picture is processing in {}'.format(image_num, train['face_patch'].shape[0], train_file.stem))
                    another_image = train['face_patch'][image_num, :]

                    lmk = ldmk['landmark'][image_num].copy()

                    left_eye_box = get_rect(lmk[42:47], scale=1)
                    right_eye_box = get_rect(lmk[36:41], scale=1)

                    left_eye_image_cv2 = another_image[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2], :]
                    right_eye_image_cv2 = another_image[right_eye_box[1]:right_eye_box[3], right_eye_box[0]:right_eye_box[2], :]

                    cv2.imwrite(str(subject_face_path.joinpath('{:0>6d}.jpg'.format(image_num))),
                                another_image)
                    cv2.imwrite(str(subject_left_eye_path.joinpath('{:0>6d}.jpg'.format(image_num))),
                                left_eye_image_cv2)
                    cv2.imwrite(str(subject_right_eye_path.joinpath('{:0>6d}.jpg'.format(image_num))),
                                right_eye_image_cv2)

                    if frame in val_group:
                        meta_validate['subject'].append(frame)
                        meta_validate['frameIndex'].append(image_num)
                        meta_validate['face_gaze_direction'].append(train['face_gaze'][image_num, :])
                        meta_validate['is_equalization'].append(train['frame_index'][image_num, 0])
                    else:
                        meta_train['subject'].append(frame)
                        meta_train['frameIndex'].append(image_num)
                        meta_train['face_gaze_direction'].append(train['face_gaze'][image_num, :])
                        meta_train['is_equalization'].append(train['frame_index'][image_num, 0])
                    count += 1
                logger.info('[{}/{}] {} picture is processing in {}', count, train['face_patch'].shape[0], mode, train_file.stem)

    sio.savemat(train_outer_dataset.joinpath('meta_train.mat'), meta_train)
    sio.savemat(val_outer_dataset.joinpath('meta_validate.mat'), meta_validate)

    test_split_path = real_dataset / 'train_test_split.json'
    test_outer_dataset = prepareOuterPath(test_outer_dataset, clear=1)

    with open(test_split_path, 'r') as f:
        test_file = json.load(f)
        test_file_name = test_file['test']

    for test_file in test_file_name:
        test_file = test_dataset / test_file
        with h5py.File(test_file) as test:
            subject_path = test_outer_dataset / test_file.stem
            subject_path.mkdir(parents=True)
            subject_face_path = subject_path / 'face'
            subject_left_eye_path = subject_path / 'left_eye'
            subject_right_eye_path = subject_path / 'right_eye'
            subject_face_path.mkdir()
            subject_left_eye_path.mkdir()
            subject_right_eye_path.mkdir()
            frame = int(re.match('subject(\d{4})$', test_file.stem).group(1))
            count_test = 0
            test_ldmk = test_landmarks / test_file.name
            with h5py.File(test_ldmk) as ldmk:
                for image_num in range(0, test['face_patch'].shape[0]):
                # for image_num in range(0, 40):
                    print('[{}/{}] test picture is processing'.format(image_num, test['face_patch'].shape[0]))
                    another_image = test['face_patch'][image_num, :]
                    
                    lmk = ldmk['landmark'][image_num].copy()

                    left_eye_box = get_rect(lmk[42:47], scale=1)
                    right_eye_box = get_rect(lmk[36:41], scale=1)

                    left_eye_image_cv2 = another_image[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2], :]
                    right_eye_image_cv2 = another_image[right_eye_box[1]:right_eye_box[3], right_eye_box[0]:right_eye_box[2], :]
                    
                    cv2.imwrite(str(subject_face_path.joinpath('{:0>6d}.jpg'.format(image_num))),
                                another_image)
                    cv2.imwrite(str(subject_left_eye_path.joinpath('{:0>6d}.jpg'.format(image_num))),
                                left_eye_image_cv2)
                    cv2.imwrite(str(subject_right_eye_path.joinpath('{:0>6d}.jpg'.format(image_num))),
                                right_eye_image_cv2)
                    # cv2.imwrite(str(subject_left_eye_rand_path.joinpath('{:0>6d}.jpg'.format(image_num))),
                    #             left_eye_image_cv2_rand)
                    # cv2.imwrite(str(subject_right_eye_rand_path.joinpath('{:0>6d}.jpg'.format(image_num))),
                    #             right_eye_image_cv2_rand)

                    meta_test['subject'].append(frame)
                    meta_test['frameIndex'].append(image_num)
                    meta_test['is_equalization'].append(test['frame_index'][image_num, 0])
                logger.info('\n-----------------------------------------------\n'
                            '[{}/{}] test picture is detected by cv2 in {}\n'
                            '-----------------------------------------------', count_test, test['face_patch'].shape[0], test_file.stem)

    sio.savemat(test_outer_dataset.joinpath('meta_test.mat'), meta_test)
    print('finished!')


# prepare output data path
def prepareOuterPath(path, clear = False):
    if not path.is_dir():
        path.mkdir()
    if clear:
        for path_dir in path.iterdir():
            if path_dir.is_dir():
                shutil.rmtree(path_dir, ignore_errors=True)
            if path_dir.is_file():
                path_dir.unlink()
    return path



def get_rect(points, ratio=1.0, scale=1):  # ratio = w:h
    x = points[:, 0]
    y = points[:, 1]

    x_expand = 0.1 * (max(x) - min(x))
    y_expand = 0.1 * (max(y) - min(y))

    x_max, x_min = max(x) + x_expand, min(x) - x_expand
    y_max, y_min = max(y) + y_expand, min(y) - y_expand

    # h:w=1:2
    if (y_max - y_min) * ratio < (x_max - x_min):
        h = (x_max - x_min) / ratio
        pad = (h - (y_max - y_min)) / 2
        y_max += pad
        y_min -= pad
    else:
        h = (y_max - y_min)
        pad = (h * ratio - (x_max - x_min)) / 2
        x_max += pad
        x_min -= pad

    int(x_min), int(x_max), int(y_min), int(y_max)
    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    bbox = np.array(bbox)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (224*scale, 224*scale))
    rect = np.concatenate([aSrc, bSrc])

    return rect


def flip_rect(rect, image_width=224):
    x1, y1, x2, y2 = rect
    y1_flip = y1
    y2_flip = y2
    x1_flip = image_width - x2
    x2_flip = image_width - x1
    rect_flip = np.array([x1_flip, y1_flip, x2_flip, y2_flip], dtype=np.int32)
    return rect_flip



if __name__ == '__main__':
    main()







