import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_default_datasets(dataset, dataroot, imageSize, classes):
    if dataroot is None and str(dataset).lower() != 'fake':
        raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % dataset)

    if dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        nc=3
        return dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(imageSize),
                               transforms.CenterCrop(imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])), nc
    elif dataset == 'lsun':
        nc = 3
        classes = [ c + '_train' for c in classes.split(',')]
        return dset.LSUN(root=dataroot, classes=classes,
                    transform=transforms.Compose([
                        transforms.Resize(imageSize),
                        transforms.CenterCrop(imageSize),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])), nc
    elif dataset == 'cifar10':
        nc = 3
        return dset.CIFAR10(root=dataroot, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])), nc

    elif dataset == 'mnist':
        nc = 1
        return dset.MNIST(root=dataroot, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(imageSize),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,)),
                   ])), nc

    elif dataset == 'fake':
        nc = 3
        return dset.FakeData(image_size=(3, imageSize, imageSize),
                        transform=transforms.ToTensor()), nc
    else:
        raise ValueError(f'Invalid dataset: {dataset} specified')
