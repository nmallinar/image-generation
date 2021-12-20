def get_default_datasets(dataset, dataroot, imageSize, classes):
    if dataroot is None and str(dataset).lower() != 'fake':
        raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % dataset)

    dataset = None
    if dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(imageSize),
                                       transforms.CenterCrop(imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        nc=3
    elif dataset == 'lsun':
        classes = [ c + '_train' for c in classes.split(',')]
        dataset = dset.LSUN(root=dataroot, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(imageSize),
                                transforms.CenterCrop(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3
    elif dataset == 'cifar10':
        dataset = dset.CIFAR10(root=dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc=3

    elif dataset == 'mnist':
            dataset = dset.MNIST(root=dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                               ]))
            nc=1

    elif dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, imageSize, imageSize),
                                transform=transforms.ToTensor())
        nc=3
    else:
        raise ValueError(f'Invalid dataset: {dataset} specified')

    return dataset
