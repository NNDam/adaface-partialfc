import numbers
import os
import queue as Queue
import threading
import sys

# sys.path.append('/home/damnguyen/FaceRecognition/FaceMask/face_sdk/')
import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
# from torch.multiprocessing import set_start_method
from transforms import transform_JPEGcompression, transform_gaussian_noise, transform_resize, transform_eraser
from torchvision import transforms
import random
import yaml
# from face_masker import FaceMasker
import cv2
from PIL import Image


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank, is_train = True):
        super(MXFaceDataset, self).__init__()

        if is_train:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 # transforms.Lambda(lambda x: transform_gaussian_noise(x, mean = 0.0, var = 10.0)),
                 transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                 transforms.RandomHorizontalFlip(),
                 # transforms.Lambda(lambda x: transform_resize(x, resize_range = (32, 112), target_size = 112)),
                 # transforms.Lambda(lambda x: transform_JPEGcompression(x, compress_range = (30, 100))),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                 ])
        else:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                 ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        
        # sample = Image.fromarray(sample)
        # sample = transform_gaussian_noise(sample, mean = 0.0, var = 10.0)
        # sample = transform_resize(sample, resize_range = (32, 112), target_size = 112)
        # sample = transform_JPEGcompression(sample, compress_range = (30, 100))
        sample = np.array(sample, dtype = np.uint8)
        # cv2.imwrite('tmp_img/img_{}.jpg'.format(index), sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

def low_res_augmentation(img):
    # resize the image to a small size and enlarge it back
    img_shape = img.shape
    side_ratio = np.random.uniform(0.2, 1.0)
    small_side = int(side_ratio * img_shape[0])
    interpolation = np.random.choice(
        [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
    small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation)
    interpolation = np.random.choice(
        [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
    aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)

    return aug_img, side_ratio

class AdaFaceDataset(Dataset):
    def __init__(self,
                root_dir,
                local_rank,
                is_train = True,
                low_res_augmentation_prob = 0.2,
                crop_augmentation_prob = 0.2,
                photometric_augmentation_prob = 0.2):

        super(AdaFaceDataset, self).__init__()
        self.low_res_augmentation_prob = low_res_augmentation_prob
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.random_resized_crop = transforms.RandomResizedCrop(size=(112, 112),
                                                                scale=(0.2, 1.0),
                                                                ratio=(0.75, 1.3333333333333333))
        self.photometric = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)
        self.is_train = is_train
        self.transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def augment(self, sample):
        # crop with zero padding augmentation
        if np.random.random() < self.crop_augmentation_prob:
            # RandomResizedCrop augmentation
            new = np.zeros_like(np.array(sample))
            if hasattr(F, '_get_image_size'):
                orig_W, orig_H = F._get_image_size(sample)
            else:
                # torchvision 0.11.0 and above
                orig_W, orig_H = F.get_image_size(sample)
            i, j, h, w = self.random_resized_crop.get_params(sample,
                                                            self.random_resized_crop.scale,
                                                            self.random_resized_crop.ratio)
            cropped = F.crop(sample, i, j, h, w)
            new[i:i+h,j:j+w, :] = np.array(cropped)
            sample = Image.fromarray(new.astype(np.uint8))
            crop_ratio = min(h, w) / max(orig_H, orig_W)
        else:
            crop_ratio = 1.0

        # low resolution augmentation
        if np.random.random() < self.low_res_augmentation_prob:
            # low res augmentation
            img_np, resize_ratio = low_res_augmentation(np.array(sample))
            sample = Image.fromarray(img_np.astype(np.uint8))
        else:
            resize_ratio = 1

        # photometric augmentation
        if np.random.random() < self.photometric_augmentation_prob:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.photometric.get_params(self.photometric.brightness, self.photometric.contrast,
                                                  self.photometric.saturation, self.photometric.hue)
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    sample = F.adjust_brightness(sample, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    sample = F.adjust_contrast(sample, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    sample = F.adjust_saturation(sample, saturation_factor)

        information_score = resize_ratio * crop_ratio
        return sample, information_score

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        sample = Image.fromarray(sample.astype(np.uint8))
        if self.is_train:
            sample, _ = self.augment(sample)
        # print(sample.shape)
        # cv2.imwrite('tmp_img/img_{}.jpg'.format(index), cv2.cvtColor(np.array(sample, dtype = np.uint8), cv2.COLOR_RGB2BGR))
        # sample = transform_gaussian_noise(sample, mean = 0.0, var = 10.0)
        # sample = transform_resize(sample, resize_range = (32, 112), target_size = 112)
        # sample = transform_JPEGcompression(sample, compress_range = (30, 100))
        
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

class SyntheticDataset(Dataset):
    def __init__(self, local_rank):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000

def dali_data_iter(
    batch_size: int, root_dir: str, num_threads: int,
    initial_fill=32768, random_shuffle=False,
    prefetch_queue_depth=512, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5)):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    import torch.distributed as dist
    rec_file = os.path.join(root_dir, 'train.rec')
    idx_file = os.path.join(root_dir, 'train.idx')
    rank: int = dist.get_rank()
    world_size: int = dist.get_world_size()


    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def __len__(self):
        return 376166

    def reset(self):
        self.iter.reset()

if __name__ == '__main__':
    train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=0)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)