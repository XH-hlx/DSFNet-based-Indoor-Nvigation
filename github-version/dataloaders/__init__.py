
from .dataloader import NYUDataset
from .SUNCG_dataloader import SUNCGDataset
from .SUNCG_RGBD_dataloader import SUNCG_RGBDDataset
from config import Path

from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    if args.dataset == 'suncg':
        SUNCG_HHA_PATH_TRAIN = '/data/common_datasets/SUNCG/SATNet_datasets/shurans_selected_HHA'
        SUNCG_HHA_PATH_TEST = '/data/common_datasets/SUNCG/SATNet_datasets/shurans_selected_val_HHA'

        SUNCG_NPZ_PATH_TRAIN = '/data/common_datasets/SUNCG/SATNet_datasets/shurans_selected'
        SUNCG_NPZ_PATH_TEST = '/data/common_datasets/SUNCG/SATNet_datasets/shurans_selected_val'

        train_dataset = SUNCGDataset(SUNCG_HHA_PATH_TRAIN, SUNCG_NPZ_PATH_TRAIN, "train")
        val_dataset = SUNCGDataset(SUNCG_HHA_PATH_TEST, SUNCG_NPZ_PATH_TEST, "test")
        print('Training data')
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers)
        print('Validate data')
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.workers)

        return train_loader, val_loader

    elif args.dataset == 'suncg_rgbd':
        SUNCGRGBD_SAMPLE_TXT_TRAIN = '/data/common_datasets/SUNCG/SATNet_datasets/image_list_train.txt'
        SUNCGRGBD_SAMPLE_TXT_TEST = '/data/common_datasets/SUNCG/SATNet_datasets/image_list_val.txt'

        SUNCGRGBD_NPZ_PATH_TRAIN = '/data/common_datasets/SUNCG/SATNet_datasets/myselect_suncg'
        SUNCGRGBD_NPZ_PATH_TEST = '/data/common_datasets/SUNCG/SATNet_datasets/myselect_suncg_val'

        train_dataset = SUNCG_RGBDDataset(SUNCGRGBD_SAMPLE_TXT_TRAIN, SUNCGRGBD_NPZ_PATH_TRAIN, "train")
        val_dataset = SUNCG_RGBDDataset(SUNCGRGBD_SAMPLE_TXT_TEST, SUNCGRGBD_NPZ_PATH_TEST, "test")
        print('Training data')
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers)
        print('Validate data')
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.workers)

        return train_loader, val_loader
    elif args.dataset == 'nyu' or 'nyucad':
        base_dirs = Path.db_root_dir(args.dataset)

        print('Training data:{}'.format(base_dirs['train']))
        train_loader = DataLoader(
            dataset=NYUDataset(base_dirs['train'], istest=False),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers
        )

        print('Validate data:{}'.format(base_dirs['val']))
        val_loader = DataLoader(
            dataset=NYUDataset(base_dirs['val'], istest=True),
            batch_size=args.batch_size,  # 1 * torch.cuda.device_count(), 1 for each GPU
            shuffle=False,
            num_workers=args.workers  # 1 * torch.cuda.device_count()
        )

        return train_loader, val_loader
    else:
        print('Dataset {} not available.'.format(args.dataset))
        raise NotImplementedError
