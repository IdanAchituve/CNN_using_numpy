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

        # convolution parameters
        self.operation = nn_params["operation"]  # operation on feature map
        self.conv_layers = nn_params["conv_num_layers"]
        self.stride = nn_params["stride"]
        self.filter_size = nn_params["filter_size"]

        self.is_train = True
        self.activations = []  # activations of fully connected
        self.activation_maps = []  # activations maps for saving conv/pol output
        self.input_as_matrix = []  # activations maps for saving conv/pol output in the compact form
        self.mask = []  # dropout mask

        # data structures for saving parameters of convolution weights
        self.filters = []
        self.filters_grads = []
        self.filters_accum_grads = []

        # data structures for saving parameters of convolution biases
        self.biases = []
        self.biases_grads = []
        self.biases_accum_grads = []

        # data structures for saving parameters of FC layers
        self.weights = []
        self.grads = []
        self.accum_grads = []

        # initialise data structures for saving weights and gradients of filters - no option for 2 subsequent pooling layers
        idx = 0
        for prev_layer, next_layer in zip(self.conv_layers, self.conv_layers[1:]):
            filter_size = self.filter_size[idx]
            self.filters += [np.random.normal(INIT_MEAN, INIT_STD, (next_layer, prev_layer, filter_size, filter_size))]
            self.filters_grads += [np.zeros((next_layer, prev_layer, filter_size, filter_size))]
            self.filters_accum_grads += [np.zeros((next_layer, prev_layer, filter_size, filter_size))]

            self.biases += [np.random.normal(INIT_MEAN, INIT_STD, next_layer)]
            self.biases_grads += [np.zeros(next_layer)]
            self.biases_accum_grads += [np.zeros(next_layer)]
            idx += 1

        # initialise data structures for saving weights and gradients of each FC layer
        for prev_layer, next_layer in zip(self.layers, self.layers[1:]):
            self.weights += [np.random.normal(INIT_MEAN, INIT_STD, (prev_layer + 1, next_layer))]
            self.grads += [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]
            self.accum_grads += [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]

        self.sec_accum_grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]
        self.logits = 0  # diff between each value an max value on final layer

    def forward(self, x):
        # x - tensor of examples. Each example is a tensor of size 3x32x32
        batch_size = np.size(x, 0)

        out = x.copy()  # copy input
        # conv/pol layers:
        for layer_num in range(len(self.conv_layers)):
            self.activation_maps.append(out.copy())
            if self.operation[layer_num] == "conv":
                out, in_matrix = self.forward_conv_layer_fast(out, layer_num)
            if self.operation[layer_num] == "pol":
                out, in_matrix = self.forward_pol_layer_fast(out, layer_num)
            self.input_as_matrix.append(in_matrix)  # for convolution save the matrix form and for pooling save special reshaped form

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
    def forward_conv_layer(self, ex, layer_num):
        # x shape: [#examples, #maps, height, width]
        x = ex.copy()
        w = self.filters[layer_num]  # parameters of current layer
        b = self.biases[layer_num]  # bias term

        S = self.stride[layer_num]
        F = np.size(w, 3)  # all filters are square so it's enough to take one dimension
        C = np.size(w, 0)

        W_orig = H_orig = np.size(x, 2)  # width and height size
        P = 0
        # pad with 0's all around
        while ((W_orig + 2*P)/S) % F != 0:
            # pad the height and width dimensions only
            x = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
            P += 1  # width or height size

        H_new = int((H_orig + 2*P - F)/S + 1)  # activation output height
        W_new = int((W_orig + 2*P - F)/S + 1)  # activation output width

        for filter_num in range(C):
            filter = w[filter_num]
            bias = b[filter_num]
            for row in range(0, H_new, S):
                for col in range(0, W_new, S):
                    curr_patch = x[:, :, row:row+F, col:col+F]
                    # apply convolution operation
                    out_act = np.apply_over_axes(np.sum, np.multiply(curr_patch, np.expand_dims(filter, axis=0)), [1, 2, 3]) + bias
                    # activation and reshape to fill all values in a row
                    out_act = np.maximum(out_act, 0).reshape(-1, 1)
                    new_map = out_act.copy() if row == 0 and col == 0 else np.concatenate((new_map, out_act.copy()), axis=1)
            # reshape to (num_examples, 1, 32, 32) for example
            activation_map = np.reshape(new_map, (-1, 1, H_new, W_new))
            # save all activations in activation_maps and concatenate along the filter axis
            out_act = activation_map.copy() if filter_num == 0 else np.concatenate((out_act, activation_map.copy()), axis=1)

        return out_act

    #  forward part for convolution layer
    def forward_pol_layer(self, ex, layer_num):
        # x shape: [#examples, #maps, height, width]
        x = ex.copy()

        W = H = np.size(x, 2)  # width and height size
        A = np.size(x, 1)  # number of activation maps
        S = self.stride[layer_num]  # window size

        # get current parameters
        for row in range(0, H, S):
            for col in range(0, W, S):
                curr_patch = x[:, :, row:row + S, col:col + S]
                # get infinity norm in each window - since I will do it after relu all numbers are not negative
                out_act = np.apply_over_axes(np.amax, curr_patch, [2, 3]).reshape(-1, A, 1)
                new_map = out_act.copy() if row == 0 and col == 0 else np.concatenate((new_map, out_act.copy()), axis=2)
        # reshape to (num_examples, num_filters, 16, 16) for example after 1st pooling
        out = np.reshape(new_map, (-1, A, int(H/S), int(W/S)))
        return out

    #  forward part for convolution layer
    def forward_conv_layer_fast(self, ex, layer_num):
        # x shape: [#examples, #maps, height, width]
        x_padded = ex.copy()

        w = self.filters[layer_num]  # parameters of current layer
        b = self.biases[layer_num]  # bias term

        S = self.stride[layer_num]
        F = w.shape[3]  # all filters are square so it's enough to take one dimension

        W_orig = H_orig = np.size(x_padded, 2)  # width and height size
        P = 0
        # pad with 0's all around
        while ((W_orig + 2*P)/S) % F != 0:
            # pad the height and width dimensions only
            x_padded = np.pad(x_padded, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
            P += 1  # width or height size

        H_new = int((H_orig + 2*P - F)/S + 1)  # activation output height
        W_new = int((W_orig + 2*P - F)/S + 1)  # activation output width

        N, C, H, W = x_padded.shape  # [#examples, #maps, height, width]

        # Get combinations of indices to pick to get N matrices, with C*3*3 rows (each column is a respective field)
        i0 = np.repeat(np.arange(F), F)
        i0 = np.tile(i0, C)
        i1 = S * np.repeat(np.arange(H_new), W_new)
        j0 = np.tile(np.arange(F), F * C)
        j1 = S * np.tile(np.arange(W_new), H_new)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)  # indices of the rows to take from x_padded
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)  # indices of the columns to take fro x_padded

        # indices of the filter to take from x_padded it will broadcast to the same shape as i and j
        k = np.repeat(np.arange(C), F * F).reshape(-1, 1)

        # for each image, take all tuples that correspond in the location (i.e., has the same location) in matrices i, j, k.
        # basically pick 1 item at a time according to the exact SAME location in each of the matrices i,j,k
        x_cols = x_padded[:, k, i, j]

        # affine transformation
        out = np.dot(w.reshape(-1, F * F * C), x_cols) + b.reshape(-1, 1, 1)

        # rearrange to the agreed order: (examples, filters, height, width)
        out = out.transpose(1, 0, 2).reshape(N, w.shape[0], H_new, W_new)
        out_act = np.maximum(out, 0)  # not the best practice but apply relu here

        #
        # reshape to (# of respective fields, # of cols in each matrix in cols, # of examples) -
        # put all corresponding respective fields of the examples in the batch in the same internal matrix (each example is a column)
        # and then concatenate the respective fields of the examples
        #x_cols = cols.transpose(1, 2, 0).reshape(FILTER_SIZE * FILTER_SIZE * C, -1)

        # concatenate the weights
        #res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

        #out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
        #out = out.transpose(3, 0, 1, 2)

        return out_act, x_cols

    #  forward part for convolution layer
    def forward_pol_layer_fast(self, ex, layer_num):

        x = ex.copy()
        S = self.stride[layer_num]  # window size

        N, C, H, W = x.shape
        x_reshaped = x.reshape(N, C, int(H / S), S, int(W / S), S)
        out = x_reshaped.max(axis=3).max(axis=4)

        return out, x_reshaped

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

        # conv/pol layers:
        C = self.conv_layers[-1]
        height = width = int(np.sqrt(dL_da[layer].shape[1] / C))
        # get delta per neuron in the final pooling layer
        dout = dL_da[layer].reshape(batch_size, C, height, width)
        # get the activation value in the final pooling layer
        out = np.transpose(self.activations[0][1:]).reshape(batch_size, C, height, width)
        for layer_num in range(len(self.conv_layers) - 1, -1, -1):
            x = self.activation_maps[layer_num]
            x_reshaped = self.input_as_matrix[layer_num]
            if self.operation[layer_num] == "conv":
                dout = self.backward_conv(x, out, dout, layer_num)
            if self.operation[layer_num] == "pol":
                dout = self.backward_pool_fast(x, x_reshaped, dout, out)
            out = x.copy()  # save activation of the input layer for next iteration

        # add regularization to gradient and average loss on batch
        for layer in range(len(self.layers) - 2, -1, -1):
            # average gradients
            self.grads[layer] = self.grads[layer] / batch_size

            # in SGD the loss function has L2 regularization component which is equivalent to weight decay
            if self.optimizer == "SGD":
                if self.reg_type == "L2":
                    dreg = self.weights[layer]
                elif self.reg_type == "L1":
                    dreg = self.weights[layer].copy()
                    dreg[dreg < 0] = -1.0
                    dreg[dreg > 0] = 1.0

                # add regularization
                self.grads[layer] += self.reg*dreg

    def backward_conv(self, x, out, dout, layer_num):

        x_padded = x.copy()
        dx_padded = np.zeros_like(x).astype(float)

        N, _, H_orig, W_orig = x.shape
        _, C, _, _ = out.shape
        weights = self.filters[layer_num]  # parameters of current layer

        S = self.stride[layer_num]
        F = np.size(weights, 3)  # filter_size. All filters are square so it's enough to take one dimension

        P = 0
        # pad with 0's all around
        while ((W_orig + 2*P)/S) % F != 0:
            # pad the height and width dimensions only
            x_padded = np.pad(x_padded, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
            dx_padded = np.pad(dx_padded, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
            P += 1  # width or height size

        H_out = int((H_orig + 2*P - F)/S + 1)  # activation output height
        W_out = int((W_orig + 2*P - F)/S + 1)  # activation output width

        for n in range(N):
            for c in range(C):
                out_row = 0
                for h in range(0, H_out, S):
                    out_col = 0
                    for w in range(0, W_out, S):
                        in_vals = x_padded[n, :, h:h + F, w:w + F]
                        out_val = out[n, c, out_row, out_col]
                        dout_val = 1 if out_val > 0 else 0  # point derivative or ReLU activation
                        delta = dout[n, c, out_row, out_col]  # error on output neuron
                        self.filters_grads[layer_num][c, :, :, :] += in_vals * dout_val * delta
                        self.biases_grads[layer_num][c] += dout_val * delta
                        dx_padded[n, :, h:h + F, w:w + F] += weights[c, :, :, :] * delta
                        out_col += 1
                    out_row += 1
        return dx_padded[:, :, P:-P, P:-P]

    def backward_pool(self, x, dout, layer_num):

        N, C, H, W = x.shape
        S = self.stride[layer_num]  # window size

        # create a matrix with the same shape as the convolution that preceded it for saving the gradients
        dx = np.zeros((N, C, H, W))

        for n in range(N):
            for c in range(C):
                pol_row = 0
                for row in range(0, H, S):
                    pol_col = 0
                    for col in range(0, W, S):
                        # previous activation map
                        curr_patch = x[n, c, row:row + S, col:col + S]
                        # max location in each respective field
                        max_loc_idx = np.unravel_index(np.argmax(curr_patch, axis=None), curr_patch.shape)
                        # update value according to location
                        delta = dout[n, c, pol_row, pol_col]
                        dx[n, c, max_loc_idx[0] + row, max_loc_idx[1] + col] = delta
                        pol_col += 1
                    pol_row += 1

        return dx

    def backward_pool_fast(self, x, x_reshaped, dout, out):

        dx_reshaped = np.zeros_like(x_reshaped)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]  # add dimensions e.g., instead of 2,5,2,2 -> 2,5,2,1,2,1
        mask = (x_reshaped == out_newaxis)  # mask indicating locations of the max value in the input image
        dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
        # make the dimensions in both arrays the same by duplicating values in each array
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]  # select locations of the max value
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)  # in case there are several max values split the error evenly
        dx = dx_reshaped.reshape(x.shape)  # reshape to the input shape

        return dx

    # return the sum of losses per batch
    def loss_function(self, labels):
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
            #else:
            #    self.accum_grads[layer_num] = self.momentum * self.accum_grads[layer_num] + (1 - self.momentum) * self.grads[layer_num]
            #    self.sec_accum_grads[layer_num] = self.second_moment * self.sec_accum_grads[layer_num] + (1 - self.second_moment) * (self.grads[layer_num] ** 2)
            #    m_hat = self.accum_grads[layer_num] / (1 - (self.momentum ** self.update_counter))  # bias corrected first moment
            #    v_hat = self.sec_accum_grads[layer_num] / (1 - (self.second_moment ** self.update_counter))  # bias corrected second moment
            #    self.weights[layer_num] = self.weights[layer_num] - self.lr * m_hat / (np.sqrt(v_hat) + EPSILON) - self.reg * self.weights[layer_num]

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
