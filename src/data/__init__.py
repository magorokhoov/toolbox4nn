"""Create dataset and dataloader"""
import logging

from torch.utils.data import Dataset, DataLoader


def create_dataset(option_ds: dict) -> Dataset:
    """
    Create Dataset.
    :param option_ds: Dataset configuration from option file
    """
    name = option_ds['name'].lower()

    if name == 'datasetcl2folder':
        from data.datasets.datasetCL2folder import DatasetCL2folder as D
    elif name == 'a2b':
        from data.datasets.datasetA2B import DatasetA2B as D
    elif name == 'img':
        from data.datasets.datasetImg import DatasetImg as D 
    elif name == 'folderfolders':
        from data.datasets.datasetFFs import DatasetFFs as D
    elif name == 'folderfolders_classae':
        from data.datasets.datasetFFs_classae import DatasetFFs as D 
    else:
        raise NotImplementedError(f'Dataset [{name:s}] is not recognized.')

    dataset = D(option_ds)
    logger = logging.getLogger('base')
    logger.info(f'Dataset [{(dataset.__class__.__name__):s} - {name}] is created.')

    return dataset


def create_dataloader(
        dataset: Dataset,
        option_ds: dict) -> DataLoader:
    """
    Create Dataloader.
    :param dataset: Dataset to use
    :param option_ds: Dataset configuration from opt file
    """

    dataloader_params = {
        "batch_size": option_ds['batch_size'],
        "shuffle": option_ds['shuffle'],
        "num_workers": option_ds['num_workers'],
        "drop_last": True
    }

    return DataLoader(dataset, pin_memory=True, **dataloader_params)
