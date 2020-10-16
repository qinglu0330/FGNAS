from torch_geometric.datasets import Planetoid


cora = Planetoid(root='data/Cora', name='Cora', split='public')


def Dataset(name, path=None):
    if path is None:
        path = "data/" + name
    dataset = Planetoid(root=path, name=name, split='public')
    return dataset
