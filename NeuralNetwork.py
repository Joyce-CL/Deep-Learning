import copy

class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.num=0

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.forward()
        # get output of forward function of each layer
        # In Test, 4 Layers: FullyConnected -> ReLu -> FullyConnected -> SoftMax
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)
        # Here input_tensor is the output of SofaMax.forward()
        # loss_layer is Loss.CrossEntropyLoss(), use Loss.CrossEntropyLoss.forward() to get loss
        loss = self.loss_layer.forward(input_tensor, self.label_tensor)
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for i in range(len(self.layers)):
            layer_num = len(self.layers) - 1 - i
            error_tensor = self.layers[layer_num].backward(error_tensor)

    def append_trainable_layer(self, layer):
        # set the optimizer of each trainable_layer(In FullyConnected)
        layer.optimizer = copy.deepcopy(self.optimizer)
        # initializing the layer with the stored initializers.
        layer.initialize(self.weights_initializer, self.bias_initializer)
        # add layer into layer_list
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)
        return input_tensor



