class Base:
    def __init__(self):
        self.phase = Phase.train
        # add weights in all layers in order to distinguish
        # trainable layers(FullyConnected & Conv) [self.weights will be not None]
        # and non-trainable layers(Flatten & Pooling & ReLU ...) [self.weights will be None]
        self.weights = None


class Phase:
    train = 'train'
    test = 'test'
