import numpy as np
import os
import pandas as pd
import random
import torch

from torchmed.datasets import MedFile, MedFolder
from torchmed.samplers import MaskableSampler
from torchmed.patterns import SquaredSlidingWindow
from torchmed.readers import SitkReader
from torchmed.utils.transforms import Pad
from torchmed.utils.augmentation import elastic_deformation_2d
import torchmed.utils.transforms as transforms



import math
import shutil
import time

import torchmed.utils.file as files
from torchmed.readers import SitkReader
from torchmed.utils.multiproc import parallelize_system_calls

#from mappings import MiccaiMapping

class MiccaiMapping(object):
    def __init__(self):
        self.all_labels = [0, 4, 11, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40,
                           41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55,
                           56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 69, 71, 72,
                           73, 75, 76, 100, 101, 102, 103, 104, 105, 106, 107,
                           108, 109, 112, 113, 114, 115, 116, 117, 118, 119,
                           120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                           132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
                           142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
                           152, 153, 154, 155, 156, 157, 160, 161, 162, 163,
                           164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
                           174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                           184, 185, 186, 187, 190, 191, 192, 193, 194, 195,
                           196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
                           206, 207]
        self.ignore_labels = [1, 2, 3] + \
                             [5, 6, 7, 8, 9, 10] + \
                             [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] + \
                             [24, 25, 26, 27, 28, 29] + \
                             [33, 34] + [42, 43] + [53, 54] + \
                             [63, 64, 65, 66, 67, 68] + [70, 74] + \
                             [80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                              90, 91, 92, 93, 94, 95, 96, 97, 98, 99] + \
                             [110, 111, 126, 127, 130, 131, 158, 159, 188, 189]
        self.overall_labels = set(self.all_labels).difference(
            set(self.ignore_labels))

        self.cortical_labels = [x for x in self.overall_labels if x >= 100]
        self.non_cortical_labels = \
            [x for x in self.overall_labels if x > 0 and x < 100]

        self.map = {v: k for k, v in enumerate(self.overall_labels)}
        self.reversed_map = {k: v for k, v in enumerate(self.overall_labels)}
        self.nb_classes = len(self.overall_labels)

    def __getitem__(self, index):
        return self.map[index]


def MICCAI_preprocessing(raw_dataset_dir, preprocessed_dataset_dir, nb_workers=1, data_split_ratio=0.7,per_patient_norm=False):
    print("got in MICCAI preprocessing!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    assert(nb_workers >= 1)

    print('\n' +
          '\n',
          ('##### Automatic Segmentation of brain MRI #####') + '\n',
          ('#####      By Pierre-Antoine Ganaye       #####') + '\n',
          ('#####         CREATIS Laboratory          #####'),
          '\n' * 2,
          ('The dataset can be downloaded at https://docs.google.com/forms/d/e/1FAIpQLSfwkdSt7hWo_tjHUDu2stDsxWTaWyLJIUiS_iapbtKaydEMIw/viewform') + '\n',
          ('This script will pre-process the MICCAI12 T1-w brain MRI multi-atlas segmentation dataset') + '\n',
          ('-------------------------------------------------------------') + '\n')

    root_dir = os.path.join(raw_dataset_dir, '')
    dest_dir = os.path.join(preprocessed_dataset_dir, '')

    # create destination directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    os.makedirs(os.path.join(dest_dir, 'train'))
    os.makedirs(os.path.join(dest_dir, 'test'))
    os.makedirs(os.path.join(dest_dir, 'validation'))

    # get IDs of train and test images from the original dataset
    print('1/ --> Reading train and validation directories')
    train_ids = set()
    for train_patient in os.listdir(os.path.join(root_dir, 'training-images')):
        filename = os.path.basename(train_patient)
        train_ids.add(filename[0:4])

    test_patients = set()
    for test_patient in os.listdir(os.path.join(root_dir, 'testing-images')):
        filename = os.path.basename(test_patient)
        test_patients.add(filename[0:4])

    # create destination directories of matching IDs
    print('2/ --> Creating corresponding tree hierarchy')
    # convert set to list so we can iterate over it
    # sorted so that we split in the same way each time this script is called
    train_ids = sorted(list(train_ids))
    test_patients = sorted(list(test_patients))

    for train_patient in train_ids:
        os.makedirs(os.path.join(dest_dir, 'train/' + train_patient))

    for test_patient in test_patients:
        os.makedirs(os.path.join(dest_dir, 'test/' + test_patient))

    print('3/ --> Copying files to destination folders')
    patient_paths = []
    # copy image for the train set
    for train_patient in train_ids:
        # image
        filename = os.path.join(
            root_dir,
            'training-images/' + train_patient + '_3.nii.gz'
        )
        dest_filename = os.path.join(
            dest_dir,
            'train/' + train_patient + '/' + 'image.nii.gz'
        )
        files.copy_file(filename, dest_filename)

        # segmentation map
        filename = os.path.join(
            root_dir,
            'training-labels/' + train_patient + '_3_glm.nii.gz'
        )
        dest_filename = os.path.join(
            dest_dir,
            'train/' + train_patient + '/' + 'segmentation.nii.gz'
        )
        files.copy_file(filename, dest_filename)

        # keep the path to the patient dir for later use
        patient_paths.append(
            os.path.join(dest_dir, 'train/' + train_patient + '/'))

    # copy image for the test set
    for test_patient in test_patients:
        # image
        filename = os.path.join(
            root_dir,
            'testing-images/' + test_patient + '_3.nii.gz'
        )
        dest_filename = os.path.join(
            dest_dir,
            'test/' + test_patient + '/' + 'image.nii.gz'
        )
        files.copy_file(filename, dest_filename)

        # segmentation map
        filename = os.path.join(
            root_dir,
            'testing-labels/' + test_patient + '_3_glm.nii.gz'
        )
        dest_filename = os.path.join(
            dest_dir,
            'test/' + test_patient + '/' + 'segmentation.nii.gz'
        )
        files.copy_file(filename, dest_filename)

        # keep the path to the patient dir for later use
        patient_paths.append(
            os.path.join(dest_dir, 'test/' + test_patient + '/'))

    print('4/ --> Affine registration to MNI space and bias field correction')
    print('Can take up to one hour depending on the chosen number of workers ({})'.format(nb_workers))

    command_list = []
    command_list2 = []
    for patient in patient_paths:
        input_scan = os.path.join(patient, 'image.nii.gz')
        registered_scan = os.path.join(patient, 'im_mni.nii.gz')
        registered_scan_bc = os.path.join(patient, 'im_mni_bc.nii.gz')
        transform = os.path.join(patient, 'mni_aff_transf.mat')

        # affine registration with flirt
        command = ('flirt -searchrx -180 180 -searchry -180 180'
                   ' -searchrz -180 180 -in {} -ref $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz '
                   '-interp trilinear -o {} -omat {}').format(
            input_scan, registered_scan, transform
        )
        command_list.append(command)

        # bias field correction
        command = ('N4BiasFieldCorrection -d 3 -i {}'
                   ' -o {} -s 4 -b 200').format(
            registered_scan, registered_scan_bc)
        command_list2.append(command)

    parallelize_system_calls(nb_workers, command_list)
    time.sleep(20)
    parallelize_system_calls(nb_workers // 2, command_list2)
    time.sleep(20)

    print('5/ --> Resampling of segmentation maps')
    print('Can take up to 30 minutes.')
    command_list = []
    command_list2 = []
    for patient in patient_paths:
        input_scan = os.path.join(patient, 'image.nii.gz')
        reg_scan = os.path.join(patient, 'im_mni.nii.gz')
        transform = os.path.join(patient, 'mni_aff_transf.mat')
        transform_c3d = os.path.join(patient, 'mni_aff_transf.c3dmat')
        seg = os.path.join(patient, 'segmentation.nii.gz')
        dest_seg = os.path.join(patient, 'seg_mni.nii.gz')

        # resampling of the image
        command = ('c3d_affine_tool -ref {} -src {} {} -fsl2ras -o {}').format(
            reg_scan, input_scan, transform, transform_c3d)
        command_list.append(command)

        # resampling of the segmentation map
        command = ('c3d {} -popas ref {} -split '
                   '-foreach -insert ref 1 -reslice-matrix {} '
                   '-endfor -merge -o {}').format(
            reg_scan, seg, transform_c3d, dest_seg)
        command_list2.append(command)

    # limit nb of workers because c3d uses a lot of memory
    parallelize_system_calls(2, command_list)
    time.sleep(20)
    parallelize_system_calls(2, command_list2)

    # delete all original images and segmentation maps
    time.sleep(20)
    for patient in patient_paths:
        input_scan = os.path.join(patient, 'image.nii.gz')
        seg = os.path.join(patient, 'segmentation.nii.gz')
        os.remove(input_scan)
        os.remove(seg)

    # pre-processing of segmentation maps
    print('6/ --> Remapping the labels')
    mapping = MiccaiMapping()
    invalid_classes = set()
    for i in mapping.all_labels:
        if i in mapping.ignore_labels:
            invalid_classes.add(i)

    for patient_path in patient_paths:
        label = SitkReader(patient_path + 'seg_mni.nii.gz')
        label_array = label.to_numpy()

        # filter out invalid class
        for inv_class in invalid_classes:
            label_array[label_array == inv_class] = -1

        # remap all valid class so that label numbers are contiguous
        for class_id in mapping.overall_labels:
            label_array[label_array == class_id] = mapping[class_id]

        # write back the changes to file
        label.to_image_file(patient_path + 'prepro_seg_mni.nii.gz')

    time.sleep(5)

    # split into train validation test
    train_patient_number = math.floor(data_split_ratio * len(train_ids))
    train_patients = train_ids[:train_patient_number]
    validation_patients = train_ids[train_patient_number:]

    for validation_patient in validation_patients:
        source = os.path.join(
            dest_dir,
            'train/' + validation_patient
        )
        destination = os.path.join(
            dest_dir,
            'validation/' + validation_patient
        )
        shutil.move(source, destination)

    # class statistics on the train dataset
    print('7/ --> Computing class statistics')
    sum_by_class = [0] * mapping.nb_classes
    class_log = open(os.path.join(dest_dir, 'train/class_log.csv'), 'a')
    class_log.write('class;volume\n')

    for train_id in train_patients:
        patient_dir = os.path.join(dest_dir, 'train/' + train_id)
        label = SitkReader(patient_dir + '/prepro_seg_mni.nii.gz')
        label_array = label.to_numpy()

        for class_id in range(0, mapping.nb_classes):
            remapped_labels = (label_array == class_id)
            nb_point = remapped_labels.sum()
            sum_by_class[class_id] += nb_point

    for class_id in range(0, mapping.nb_classes):
        class_log.write('{};{}\n'.format(class_id, sum_by_class[class_id]))
    class_log.flush()

    print('8/ --> Mean centering and reduction')
    mean, std = (0, 0)

    for train_id in train_patients:
        patient_dir = os.path.join(dest_dir, 'train/' + train_id)
        brain = SitkReader(patient_dir + '/im_mni_bc.nii.gz',
                           torch_type='torch.FloatTensor').to_torch()
        mean += brain.mean()
        std += brain.std()

    train_mean = mean / len(train_patients)
    train_std = std / len(train_patients)

    for name, dataset in [('train/', train_patients),
                          ('validation/', validation_patients),
                          ('test/', test_patients)]:
        for p_id in dataset:
            patient_dir = os.path.join(dest_dir, name + p_id)
            brain = SitkReader(patient_dir + '/im_mni_bc.nii.gz',
                               torch_type='torch.FloatTensor')
            brain_array = brain.to_numpy()
            if not per_patient_norm:
                brain_array[...] = (brain_array - train_mean) / train_std

            else:
                brain_array[...] = (brain_array - brain_array.mean()) / brain_array.std()

            brain.to_image_file(patient_dir + '/prepro_im_mni_bc.nii.gz')

    stats_log = open(os.path.join(dest_dir, 'train/stats_log.txt'), "w")
    stats_log.write('average mean: {:.10f}\n'
                    'average standard deviation: {:.10f}'.format(train_mean, train_std))
    stats_log.flush()

    # allowed data
    allowed_train = open(os.path.join(dest_dir, 'train/allowed_data.txt'), "w")
    for train_patient in train_patients:
        allowed_train.write(train_patient + '\n')
    allowed_train.flush()

    allowed_val = open(os.path.join(dest_dir, 'validation/allowed_data.txt'), "w")
    for val_patient in validation_patients:
        allowed_val.write(val_patient + '\n')
    allowed_val.flush()

    allowed_test = open(os.path.join(dest_dir, 'test/allowed_data.txt'), "w")
    for test_patient in test_patients:
        allowed_test.write(test_patient + '\n')
    allowed_test.flush()
    

class MICCAI2012Dataset(object):
    def __init__(self, dataset_dir, nb_workers, preprocess_raw):
        
        self.dataset_dir = dataset_dir
        
        self.train_dataset = MedFolder(
            generate_medfiles(os.path.join(dataset_dir, 'train'), nb_workers),
            transform=transform_train, target_transform=transform_target,
            paired_transform=elastic_transform)
        self.validation_dataset = MedFolder(
            generate_medfiles(os.path.join(dataset_dir, 'validation'), nb_workers),
            transform=transform_train, target_transform=transform_target,
            paired_transform=elastic_transform)

        # init all the images before multiprocessing
        for medfile in self.train_dataset._medfiles:
            medfile._sampler._coordinates.share_memory_()
            for k, v in medfile._sampler._data.items():
                v._torch_init()

        # init all the images before multiprocessing
        for medfile in self.validation_dataset._medfiles:
            medfile._sampler._coordinates.share_memory_()
            for k, v in medfile._sampler._data.items():
                v._torch_init()

        # read cumulated volume of each labels
        df = pd.read_csv(os.path.join(dataset_dir, 'train/class_log.csv'), sep=';', index_col=0)
        self.class_freq = torch.from_numpy(df['volume'].values).float()

        # read ground truth adjacency matrix
        adjacency_mat_path = os.path.join(dataset_dir, 'train/graph.csv')
        self.adjacency_mat = torch.from_numpy(np.loadtxt(adjacency_mat_path, delimiter=';'))


class SemiDataset(object):
    def __init__(self, dataset_dir, nb_workers):
        def transform_semi(tensor):
            return tensor.permute(1, 0, 2)

        self.train_dataset = MedFolder(
            generate_medfiles(dataset_dir, nb_workers, with_target=False),
            transform=transform_semi)

        # init all the images before multiprocessing
        for medfile in self.train_dataset._medfiles:
            medfile._sampler._coordinates.share_memory_()
            for k, v in medfile._sampler._data.items():
                v._torch_init()


def build_patient_data_map(dir, with_target):
    # pads each dimension of the image on both sides.
    pad_reflect = Pad(((1, 1), (3, 3), (1, 1)), 'reflect')
    file_map = {
        'image_ref': SitkReader(
            os.path.join(dir, 'prepro_im_mni_bc.nii.gz'),
            torch_type='torch.FloatTensor', transform=pad_reflect)
    }
    if with_target:
        file_map['target'] = SitkReader(
            os.path.join(dir, 'prepro_seg_mni.nii.gz'),
            torch_type='torch.LongTensor', transform=pad_reflect)

    return file_map


def build_sampler(nb_workers, with_target):
    # sliding window of size [184, 7, 184] without padding
    patch2d = SquaredSlidingWindow(patch_size=[184, 7, 184], use_padding=False)
    # pattern map links image id to a Sampler
    pattern_mapper = {'input': ('image_ref', patch2d)}
    if with_target:
        pattern_mapper['target'] = ('target', patch2d)

    # add a fixed offset to make patch sampling faster (doesn't look for all positions)
    return MaskableSampler(pattern_mapper, offset=[92, 1, 92],
                           nb_workers=nb_workers)


def elastic_transform(data, label):
    # elastic deformation
    if random.random() > 0.4:
        data_label = torch.cat([data, label.unsqueeze(0).float()], 0)
        data_label = elastic_deformation_2d(
            data_label,
            data_label.shape[1] * 1.05,  # intensity of the deformation
            data_label.shape[1] * 0.05,  # smoothing of the deformation
            0,  # order of bspline interp
            mode='nearest')  # border mode

        data = data_label[0:7]
        label = data_label[7].long()

    return data, label


def generate_medfiles(dir, nb_workers, data_map_fn=build_patient_data_map,
                      sampler_fn=build_sampler, with_target=True):
    # database composed of dirname contained in the allowed_data.txt
    database = open(os.path.join(dir, 'allowed_data.txt'), 'r')
    patient_list = [line.rstrip('\n') for line in database]
    medfiles = []

    # builds a list of MedFiles, one for each folder
    for patient in patient_list:
        if patient:
            patient_dir = os.path.join(dir, patient)
            patient_data = data_map_fn(patient_dir, with_target)
            patient_file = MedFile(patient_data, sampler_fn(nb_workers, with_target))
            medfiles.append(patient_file)

    return medfiles


def transform_train(tensor):
    return tensor.permute(1, 0, 2)


def transform_target(tensor):
    return tensor.permute(1, 0, 2)[3]