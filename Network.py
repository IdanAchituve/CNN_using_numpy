import numpy as np
np.random.seed(222)

# parameters for initialization
INIT_MEAN = 0.0
INIT_STD = 0.01
MIN_LR = 0.0001
LR_SCALE = 2
MAX_MOMENTUM = 0.9
MOMENTUM_SCALE = 0.2
EPSILON = 10 ** -8
FILTER_SIZE = 3
POL_WINDOW = 2


class CNN:

    def __init__(self, nn_params):
        self.model = nn_params["model"]
        self.optimizer = nn_params["optimizer"]
        self.initial_lr = nn_params["lr"]  # initial learning rate
        self.lr = nn_params["lr"]  # learning rates
        self.momentum = nn_params["momentum"]
        self.second_moment = nn_params["second_moment"]
        self.update_counter = 0
        self.epoch = 0
        self.reg = nn_params["reg_lambda"]  # lambda
        self.initilal_reg = nn_params["reg_lambda"]  # initial reg param
        self.reg_type = nn_params["reg_type"]
        self.dropout = nn_params["dropout"]  # a list of dropout probability per layer
        self.layers = nn_params["layers"]  # list of layers size
        self.activation_functions = nn_params["activations"]  # list of activation functions
        self.operation = nn_params["operation"]  # operation on feature map
        self.num_filters = [3] + nn_params["conv_num_filters"]  # num of filters after each convolution

        self.is_train = True
        self.activations = []  # activations of fully connected
        self.activation_maps = []  # activations maps for saving conv/pol output
        self.mask = []  # dropout mask

        # data structures for saving weights and gradients of filters
        self.filters = [np.random.normal(INIT_MEAN, INIT_STD, (next_layer, prev_layer, FILTER_SIZE, FILTER_SIZE)) for prev_layer, next_layer in zip(self.num_filters, self.num_filters[1:])]
        self.biases = [np.random.normal(INIT_MEAN, INIT_STD, next_layer) for prev_layer, next_layer in zip(self.num_filters, self.num_filters[1:])]
        self.filters_grads = [np.zeros((next_layer, prev_layer, FILTER_SIZE, FILTER_SIZE)) for prev_layer, next_layer in zip(self.num_filters, self.num_filters[1:])]
        self.biases_grads = [np.zeros(next_layer) for prev_layer, next_layer in zip(self.num_filters, self.num_filters[1:])]

        # data structures for saving weights and gradients of each FC layer
        self.weights = [np.random.normal(INIT_MEAN, INIT_STD, (prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]
        self.grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]
        self.accum_grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]
        self.sec_accum_grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]
        self.logits = 0  # diff between each value an max value on final layer

    def forward(self, x):
        # x - tensor of examples. Each example is a tensor of size 3x32x32
        batch_size = np.size(x, 0)

        out = x.copy()  # copy input
        # conv/pol layers:
        for layer_num in range(len(self.operation)):
            self.activation_maps.append(out.copy())
            if self.operation[layer_num] == "conv":
                out = self.forward_conv_layer(out, layer_num)
            if self.operation[layer_num] == "pol":
                out = self.forward_pol_layer(out)

        # flatten last conv/pool layer and transpose so each column is an image in the batch
        out = out.reshape(batch_size, -1).transpose().copy()
        for layer_num in range(len(self.layers) - 1):
            # dropout in training time only
            success_prob = 1 - self.dropout[layer_num]  # 0.2 dropout is 0.2 success = ~0.8 should of neurons should not be zeroed out
            if self.is_train:
                dmask = np.random.binomial(n=1, p=success_prob, size=out.shape) / success_prob
                out = out * dmask   # element wise multiplication by the mask and scaling output
                self.mask.append(dmask)  # save mask for backprop phase

            # add bias to layer and save activations
            out = np.concatenate((np.ones(batch_size).reshape(1, -1), out), axis=0)
            self.activations.append(out.copy())

            # linear transformation
            out = np.dot(self.weights[layer_num].transpose(), out)  # z = W.T*x

            # non linearity
            if self.activation_functions[layer_num] == "relu":
                out = np.maximum(out, 0)  # a = relu(z, 0)
            elif self.activation_functions[layer_num] == "tanh":
                out = np.tanh(out)  # a = tanh(z)
            elif self.activation_functions[layer_num] == "softmax":
                max_val = np.max(out, axis=0)  # find the max valued class in each column (example)
                self.logits = out - max_val
                e_x = np.exp(self.logits)  # subtract max_val from all values of each example to prevent overflow
                out = e_x/np.sum(e_x, axis=0)

        return out.copy()

    #  forward part for convolution layer
    def forward_conv_layer(self, x, layer_num):
        # x shape: [#examples, #maps, height, width]
        side_size = np.size(x, 2)  # width or height size
        orig_side_size = side_size
        # pad with 0's all around
        while side_size % FILTER_SIZE != 0:
            # pad the height and width dimensions only
            x = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
            side_size = np.size(x, 2)  # width or height size

        # get current parameters
        filters = self.filters[layer_num]  # parameters of current layer
        biases = self.biases[layer_num]  # bias term
        num_filters = np.size(filters, 0)
        for filter_num in range(num_filters):
            filter = filters[filter_num]
            bias = biases[filter_num]
            for row in range(orig_side_size):
                for col in range(orig_side_size):
                    curr_patch = x[:, :, row:row+FILTER_SIZE, col:col+FILTER_SIZE]
                    # apply convolution operation
                    out_act = np.apply_over_axes(np.sum, np.multiply(curr_patch, np.expand_dims(filter, axis=0)), [1, 2, 3]) + bias
                    # activation and reshape to fill all values in a row
                    out_act = np.maximum(out_act, 0).reshape(-1, 1)
                    new_map = out_act.copy() if row == 0 and col == 0 else np.concatenate((new_map, out_act.copy()), axis=1)
            # reshape to (num_examples, 1, 32, 32) for example
            activation_map = np.reshape(new_map, (-1, 1, orig_side_size, orig_side_size))
            # save all activations in activation_maps and concatenate along the filter axis
            activation_maps = activation_map.copy() if filter_num == 0 else np.concatenate((activation_maps, activation_map.copy()), axis=1)

        return activation_maps

    #  forward part for convolution layer
    def forward_pol_layer(self, x):
        # x shape: [#examples, #maps, height, width]
        side_size = np.size(x, 2)  # width or height size
        num_filters = np.size(x, 1)  # width or height size

        # POL_WINDOW
        # get current parameters
        for row in range(0, side_size, POL_WINDOW):
            for col in range(0, side_size, POL_WINDOW):
                curr_patch = x[:, :, row:row + POL_WINDOW, col:col + POL_WINDOW]
                # get infinity norm in each window - since I will do it after relu all numbers are not negative
                out_act = np.apply_over_axes(np.amax, curr_patch, [2, 3]).reshape(-1, num_filters, 1)
                new_map = out_act.copy() if row == 0 and col == 0 else np.concatenate((new_map, out_act.copy()), axis=2)
        # reshape to (num_examples, num_filters, 16, 16) for example after 1st pooling
        activation_maps = np.reshape(new_map, (-1, num_filters, int(side_size/POL_WINDOW), int(side_size/POL_WINDOW)))
        return activation_maps

    def backward(self, net_out, labels):

        # activations point derivative
        def dactivation_dz(layer, activation_val):
            if self.activation_functions[layer] == "tanh":
                return 1 - np.tanh(activation_val) ** 2
            elif self.activation_functions[layer] == "relu":
                dactivation = activation_val.copy()
                dactivation[dactivation <= 0] = 0
                dactivation[dactivation > 0] = 1
                return dactivation
            else:
                return np.ones(activation_val.shape)

        batch_size = np.size(labels, 1)
        # for each example in the batch sum gradients on all layers
        dL_da = [0] * (len(self.layers) - 1)
        for layer in range(len(self.layers) - 2, -1, -1):
            # delta = dL/da * da/dz
            if layer == len(self.layers) - 2:
                delta = (net_out - labels).transpose()
            else:
                delta = dL_da[layer + 1] * (dactivation_dz(layer, self.activations[layer + 1][1:, :]).transpose())
            prev_act = self.activations[layer]  # get activation of the prev layer
            self.grads[layer] = np.dot(prev_act, delta)  # dL/dw = (a_m - T)*a_m-1^T
            dL_da[layer] = np.dot(delta, self.weights[layer][1:, :].transpose())  # dL/d(a_m-1) = w_m^T*(a_m - T)
            dL_da[layer] *= (self.mask[layer]).transpose()

        # add regularization to gradient and average loss on batch
        for layer in range(len(self.layers) - 2, -1, -1):
            # average gradients
            self.grads[layer] = self.grads[layer] / batch_size

            # in SGD the loss has L2 regularization
            if self.optimizer == "SGD":
                if self.reg_type == "L2":
                    dreg = self.weights[layer]
                elif self.reg_type == "L1":
                    dreg = self.weights[layer].copy()
                    dreg[dreg < 0] = -1.0
                    dreg[dreg > 0] = 1.0

                # add regularization
                self.grads[layer] += self.reg*dreg

    # return the sum of losses per batch
    def loss_function(self, batched_data, net_out, labels):
        sum_weights = 0.0
        if self.optimizer == "SGD":
            for l in range(len(self.layers) - 1):
                # L2 regularization proportional to the loss value
                reg_term = (1/2) * np.sum(self.weights[l] ** 2) if self.reg_type == "L2" else np.sum(np.abs(self.weights[l]))
                sum_weights += reg_term

        # numerically stable log likelihood calculation
        label_exit = np.sum(self.logits * labels, axis=0)  # get the value at the true exit
        e_x = np.exp(self.logits)
        loss = -(label_exit - np.log(np.sum(e_x, axis=0)))

        sum_loss = np.sum(loss) + self.reg*sum_weights
        return sum_loss

    def test_time(self):
        self.is_train = False

    def train_time(self):
        self.is_train = True

    def init_vals(self, init_grads=False):
        self.activations = []
        self.mask = []
        self.logits = 0
        if init_grads:
            self.grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]

    def step(self):
        self.update_counter += 1  # count time steps
        for layer_num in range(len(self.layers) - 1):
            # Nesterov gradient calculation
            if self.optimizer == "SGD":
                prev_accum_grads = self.accum_grads[layer_num].copy()
                self.accum_grads[layer_num] = self.momentum * self.accum_grads[layer_num] - self.lr * self.grads[layer_num]
                self.weights[layer_num] = self.weights[layer_num] - self.momentum * prev_accum_grads + (1 + self.momentum) * self.accum_grads[layer_num]

            # ADAM optimizer with weight decay
            else:
                self.accum_grads[layer_num] = self.momentum * self.accum_grads[layer_num] + (1 - self.momentum) * self.grads[layer_num]
                self.sec_accum_grads[layer_num] = self.second_moment * self.sec_accum_grads[layer_num] + (1 - self.second_moment) * (self.grads[layer_num] ** 2)
                m_hat = self.accum_grads[layer_num] / (1 - (self.momentum ** self.update_counter))  # bias corrected first moment
                v_hat = self.sec_accum_grads[layer_num] / (1 - (self.second_moment ** self.update_counter))  # bias corrected second moment
                self.weights[layer_num] = self.weights[layer_num] - self.lr * m_hat / (np.sqrt(v_hat) + EPSILON) - self.reg * self.weights[layer_num]

    def get_grads(self):
        return self.grads.copy()

    def get_params(self):
        return self.weights.copy()

    def set_param(self, layer, src_neuron, dst_neuron, val):
        self.weights[layer][src_neuron, dst_neuron] = val

    def decay_lr(self):
        self.epoch = self.epoch + 1
        if self.optimizer == "SGD":
            self.lr = max(MIN_LR, self.lr / LR_SCALE)  # cut by halve each time
        else:  # In ADAM decay lr  and regularization at each epoch
            self.lr = self.initial_lr / np.sqrt(self.epoch)
            self.reg = self.initilal_reg / np.sqrt(self.epoch)

    def momentum_change(self):
        self.momentum = min(MAX_MOMENTUM, self.momentum + MOMENTUM_SCALE)  # change momentum

    def weights_norm(self):
        # calc norm of each matrix and max eigenvalue
        for layer_num in range(len(self.layers) - 1):
            dot_product = np.dot(self.weights[layer_num], self.weights[layer_num].transpose())
            norm = np.linalg.norm(dot_product)
            max_eigenval = np.max(np.linalg.eig(dot_product)[0])
            norm_eig = np.asarray([int(layer_num + 1), float(norm), max_eigenval.real]).reshape(1, -1)
            net_norm = norm_eig.copy() if layer_num == 0 else np.concatenate((net_norm, norm_eig.copy()), axis=0)
        net_norm = np.concatenate((net_norm, np.zeros(3).reshape(1, -1)))  # add delimiter between epochs
        return net_norm

    def init_weights(self, weights, accum_grads, sec_accum_grads):
        # copy weights learned by AE aside from the last layer
        for layer_num in range(len(self.layers) - 1):
            self.weights[layer_num] = weights[layer_num].copy()
            self.accum_grads[layer_num] = accum_grads[layer_num].copy()
            self.sec_accum_grads[layer_num] = sec_accum_grads[layer_num].copy()
