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

class CNN:

    def __init__(self, nn_params):
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
        self.conv_activation = nn_params["conv_activation"]
        self.padding = nn_params["padding"]
        self.pooling_stride = nn_params["pooling_stride"]

        self.is_train = True
        self.logits = 0  # diff between each value an max value on final layer
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

        # initialise data structures for saving weights and gradients of filters
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
            self.grads += [np.zeros((prev_layer + 1, next_layer))]
            self.accum_grads += [np.zeros((prev_layer + 1, next_layer))]

    def init_vals(self, init_grads=False):
        self.activations = []  # activations of fully connected
        self.activation_maps = []  # activations maps for saving conv/pol output
        self.input_as_matrix = []  # activations maps for saving conv/pol output in the compact form
        self.mask = []  # dropout mask
        self.logits = 0

        if init_grads:
            self.grads = []
            self.filters_grads = []
            self.biases_grads = []

            # initialise data structures for saving weights and gradients of filters
            idx = 0
            for prev_layer, next_layer in zip(self.conv_layers, self.conv_layers[1:]):
                filter_size = self.filter_size[idx]
                self.filters_grads += [np.zeros((next_layer, prev_layer, filter_size, filter_size))]
                self.biases_grads += [np.zeros(next_layer)]
                idx += 1

            # initialise data structures for saving weights and gradients of each FC layer
            for prev_layer, next_layer in zip(self.layers, self.layers[1:]):
                self.grads += [np.zeros((prev_layer + 1, next_layer))]

    def forward(self, x):
        # x - tensor of examples. Each example is a tensor of size 3x32x32
        batch_size = np.size(x, 0)

        out = x.copy()  # copy input
        conv_idx = 0
        pool_idx = 0
        # conv/pol layers:
        for layer_num in range(len(self.operation)):
            self.activation_maps.append(out.copy())
            if self.operation[layer_num] == "conv":
                padding = self.padding[conv_idx]
                stride = self.stride[conv_idx]
                w = self.filters[conv_idx]  # parameters of current layer
                b = self.biases[conv_idx]  # bias term
                out, in_matrix = self.forward_conv_layer_fast(out, padding, stride, w, b)  # convolution operation
                # activation
                if self.conv_activation[conv_idx] == "relu":
                    out = np.maximum(0, out)
                conv_idx += 1
            if self.operation[layer_num] == "pol":
                stride = self.pooling_stride[pool_idx]
                out, in_matrix = self.forward_pol_layer_fast(out, stride)
                pool_idx += 1
            self.input_as_matrix.append(in_matrix)  # for convolution save the matrix form and for pooling save special reshaped form

        self.activation_maps.append(out.copy())
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

    def get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        assert (H + 2 * padding - field_height) % stride == 0
        assert (W + 2 * padding - field_height) % stride == 0
        out_height = int((H + 2 * padding - field_height) / stride + 1)
        out_width = int((W + 2 * padding - field_width) / stride + 1)

        # Get combinations of indices to pick to get N matrices, with C*3*3 rows (each column is a respective field)
        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)  # indices of the rows to take from x_padded
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)  # indices of the columns to take fro x_padded

        # indices of the filter to take from x_padded it will broadcast to the same shape as i and j
        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return (k, i, j)

    #  forward part for convolution layer
    def forward_conv_layer(self, ex, padding, stride, w, b):
        # x shape: [#examples, #maps, height, width]
        x = ex.copy()
        num_filters, _, field_height, field_width = w.shape

        W_orig = H_orig = np.size(x, 2)  # width and height size

        # Check dimensions
        assert (W_orig + 2 * padding - field_height) % stride == 0, 'width does not work'
        assert (H_orig + 2 * padding - field_width) % stride == 0, 'height does not work'

        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
        H_new = int((H_orig + 2*padding - field_height)/stride + 1)  # activation output height
        W_new = int((W_orig + 2*padding - field_width)/stride + 1)  # activation output width

        for filter_num in range(num_filters):
            filter = w[filter_num]
            bias = b[filter_num]
            for row in range(0, H_new, stride):
                for col in range(0, W_new, stride):
                    curr_patch = x[:, :, row:row+field_height, col:col+field_width]
                    # apply convolution operation
                    out_patch = np.apply_over_axes(np.sum, np.multiply(curr_patch, np.expand_dims(filter, axis=0)), [1, 2, 3]) + bias
                    out_patch = out_patch.reshape(-1, 1)
                    new_map = out_patch.copy() if row == 0 and col == 0 else np.concatenate((new_map, out_patch.copy()), axis=1)
            # reshape to (num_examples, 1, 32, 32) for example
            activation_map = np.reshape(new_map, (-1, 1, H_new, W_new))
            # save all activations in activation_maps and concatenate along the filter axis
            out = activation_map.copy() if filter_num == 0 else np.concatenate((out, activation_map.copy()), axis=1)

        return out

    #  forward part for convolution layer
    def forward_pol_layer(self, ex, stride):
        # x shape: [#examples, #maps, height, width]
        x = ex.copy()

        W = H = np.size(x, 2)  # width and height size
        A = np.size(x, 1)  # number of activation maps

        # get current parameters
        for row in range(0, H, stride):
            for col in range(0, W, stride):
                curr_patch = x[:, :, row:row + stride, col:col + stride]
                # get infinity norm in each window - since I will do it after relu all numbers are not negative
                out_act = np.apply_over_axes(np.amax, curr_patch, [2, 3]).reshape(-1, A, 1)
                new_map = out_act.copy() if row == 0 and col == 0 else np.concatenate((new_map, out_act.copy()), axis=2)
        # reshape to (num_examples, num_filters, 16, 16) for example after 1st pooling
        out = np.reshape(new_map, (-1, A, int(H/stride), int(W/stride)))
        return out

    #  forward part for convolution layer
    def forward_conv_layer_fast(self, ex, padding, stride, w, b):
        # x shape: [#examples, #maps, height, width]

        N, C, H, W = ex.shape  # [#examples, #maps, height, width]
        num_filters, _, field_height, field_width = w.shape

        # Check dimensions
        assert (W + 2 * padding - field_height) % stride == 0, 'width does not work'
        assert (H + 2 * padding - field_width) % stride == 0, 'height does not work'

        out_height = int((H + 2*padding - field_height)/stride + 1)  # activation output height
        out_width = int((W + 2*padding - field_width)/stride + 1)  # activation output width
        out = np.zeros((N, num_filters, out_height, out_width), dtype=ex.dtype)

        x_padded = np.pad(ex.copy(), ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
        k, i, j = self.get_im2col_indices(ex.shape, field_height, field_width, padding, stride)

        # for each image, take all tuples that correspond in the location (i.e., has the same location) in matrices i, j, k.
        # basically pick 1 item at a time according to the exact SAME location in each of the matrices i,j,k
        x_cols = x_padded[:, k, i, j]

        # reshape to (# of respective fields, # of cols in each matrix in cols, # of examples) -
        # put all corresponding respective fields of the examples in the batch in the same internal matrix (each example is a column)
        # and then concatenate the respective fields of the examples
        x_cols = x_cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)

        # concatenate the weights and make affine transformation
        res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

        out = res.reshape(w.shape[0], out.shape[2], out.shape[3], ex.shape[0])
        out = out.transpose(3, 0, 1, 2)

        return out, x_cols

    #  forward part for convolution layer
    def forward_pol_layer_fast(self, ex, stride):

        x = ex.copy()

        N, C, H, W = x.shape
        x_reshaped = x.reshape(N, C, int(H / stride), stride, int(W / stride), stride)
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
        out_as_mat = self.activation_maps[-1]
        C, height, width = out_as_mat.shape[1], out_as_mat.shape[2], out_as_mat.shape[3]  # get number of filters in the last layer
        # get delta per neuron in the final pooling layer
        dout = dL_da[layer].reshape(out_as_mat.shape)
        # get the activation value in the final pooling layer
        out = np.transpose(self.activations[0][1:]).reshape(batch_size, C, height, width)
        conv_idx = len(self.conv_activation) - 1
        pool_idx = len(self.pooling_stride) - 1
        for layer_num in range(len(self.operation) - 1, -1, -1):
            x = self.activation_maps[layer_num]
            x_reshaped = self.input_as_matrix[layer_num]
            if self.operation[layer_num] == "conv":
                # update neuron error by applying activation derivative
                if self.conv_activation[conv_idx] == "relu":
                    dout[out <= 0] = 0
                padding = self.padding[conv_idx]
                stride = self.stride[conv_idx]
                w = self.filters[conv_idx]  # parameters of current layer
                b = self.biases[conv_idx]  # bias term
                dout = self.backward_conv_fast(x, x_reshaped, dout, conv_idx, padding, stride, w, b)  # get the error on the input layer
                conv_idx -= 1
            if self.operation[layer_num] == "pol":
                stride = self.pooling_stride[pool_idx]
                dout = self.backward_pool_fast(x, x_reshaped, dout, out)  # get the error on the input layer
                pool_idx -= 1
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
                self.grads[layer] += self.reg * dreg

        # add regularization to gradient and average loss on batch in convolution layers. filter size is the filter of each
        # convolution layer so it's a good indication for the number of conv layers
        for layer in range(len(self.filter_size) - 1, -1, -1):
            # average gradients
            self.filters_grads[layer] = self.filters_grads[layer] / batch_size
            self.biases_grads[layer] = self.biases_grads[layer] / batch_size

            # in SGD in the loss function there is L2 regularization component which is equivalent to weight decay
            if self.optimizer == "SGD":
                if self.reg_type == "L2":
                    dreg = self.filters[layer]  # apply regularization only on weights
                elif self.reg_type == "L1":
                    dreg = self.filters[layer].copy()
                    dreg[dreg < 0] = -1.0
                    dreg[dreg > 0] = 1.0

                # add regularization
                self.filters_grads[layer] += self.reg * dreg

    def backward_conv(self, x, out, dout, conv_idx, padding, stride, w, b):

        x_padded = x.copy()
        dx_padded = np.zeros_like(x).astype(float)

        N, C, H, W = x.shape
        num_filters, _, field_height, field_width = w.shape

        x_padded = np.pad(x_padded, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
        dx_padded = np.pad(dx_padded, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)

        H_out = int((H + 2*padding - field_height)/stride + 1)  # activation output height
        W_out = int((W + 2*padding - field_width)/stride + 1)  # activation output width

        for n in range(N):
            for c in range(num_filters):
                out_row = 0
                for h in range(0, H_out, stride):
                    out_col = 0
                    for w in range(0, W_out, stride):
                        in_vals = x_padded[n, :, h:h + field_height, w:w + field_width]
                        delta = dout[n, c, out_row, out_col]  # error on output neuron
                        self.filters_grads[conv_idx][c, :, :, :] += in_vals * delta
                        self.biases_grads[conv_idx][c] += delta
                        dx_padded[n, :, h:h + field_height, w:w + field_width] += w[c, :, :, :] * delta
                        out_col += 1
                    out_row += 1
        dx = dx_padded[:, :, padding:-padding, padding:-padding] if padding > 0 else dx_padded.copy()
        return dx

    def backward_conv_fast(self, x, x_cols, dout, conv_idx, padding, stride, w, b):

        N, C, H, W = x.shape

        # gradient of bias units
        self.biases_grads[conv_idx] = np.sum(dout, axis=(0, 2, 3))  # gradient on bias
        # gradient of weights
        num_filters, _, field_height, field_width = w.shape
        # move the batch to the last axis and then concatenate all values per filter
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        self.filters_grads[conv_idx] = dout_reshaped.dot(x_cols.T).reshape(w.shape)

        H_padded, W_padded = H + 2 * padding, W + 2 * padding

        # gradient over input
        dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=dx_cols.dtype)
        k, i, j = self.get_im2col_indices(x.shape, field_height, field_width, padding, stride)
        cols_reshaped = dx_cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        dx = x_padded[:, :, padding:-padding, padding:-padding] if padding > 0 else x_padded

        return dx

    def backward_pool(self, x, dout, stride):

        N, C, H, W = x.shape

        # create a matrix with the same shape as the convolution that preceded it for saving the gradients
        dx = np.zeros((N, C, H, W))

        for n in range(N):
            for c in range(C):
                pol_row = 0
                for row in range(0, H, stride):
                    pol_col = 0
                    for col in range(0, W, stride):
                        # previous activation map
                        curr_patch = x[n, c, row:row + stride, col:col + stride]
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

            for l in range(len(self.filter_size)):
                # L2 regularization proportional to the loss value
                reg_term = (1/2) * np.sum(self.filters[l] ** 2) if self.reg_type == "L2" else np.sum(np.abs(self.filters[l]))
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

    def step(self):
        self.update_counter += 1  # count time steps
        for layer_num in range(len(self.layers) - 1):
            # Nesterov gradient calculation
            if self.optimizer == "SGD":
                prev_accum_grads = self.accum_grads[layer_num].copy()
                self.accum_grads[layer_num] = self.momentum * self.accum_grads[layer_num] - self.lr * self.grads[layer_num]
                self.weights[layer_num] = self.weights[layer_num] - self.momentum * prev_accum_grads + (1 + self.momentum) * self.accum_grads[layer_num]

        # update step for convolution layers
        for layer_num in range(len(self.filter_size)):
            # Nesterov gradient calculation
            if self.optimizer == "SGD":
                # update weights
                prev_accum_grads = self.filters_accum_grads[layer_num].copy()
                self.filters_accum_grads[layer_num] = self.momentum * self.filters_accum_grads[layer_num] - self.lr * self.filters_grads[layer_num]
                self.filters[layer_num] = self.filters[layer_num] - self.momentum * prev_accum_grads + (1 + self.momentum) * self.filters_accum_grads[layer_num]

                # update bias
                prev_accum_grads = self.biases_accum_grads[layer_num].copy()
                self.biases_accum_grads[layer_num] = self.momentum * self.biases_accum_grads[layer_num] - self.lr * self.biases_grads[layer_num]
                self.biases[layer_num] = self.biases[layer_num] - self.momentum * prev_accum_grads + (1 + self.momentum) * self.biases_accum_grads[layer_num]

                # ADAM optimizer with weight decay
            #else:
            #    self.accum_grads[layer_num] = self.momentum * self.accum_grads[layer_num] + (1 - self.momentum) * self.grads[layer_num]
            #    self.sec_accum_grads[layer_num] = self.second_moment * self.sec_accum_grads[layer_num] + (1 - self.second_moment) * (self.grads[layer_num] ** 2)
            #    m_hat = self.accum_grads[layer_num] / (1 - (self.momentum ** self.update_counter))  # bias corrected first moment
            #    v_hat = self.sec_accum_grads[layer_num] / (1 - (self.second_moment ** self.update_counter))  # bias corrected second moment
            #    self.weights[layer_num] = self.weights[layer_num] - self.lr * m_hat / (np.sqrt(v_hat) + EPSILON) - self.reg * self.weights[layer_num]

    def get_grads(self):
        return self.grads.copy(), self.filters_grads.copy(), self.biases_grads.copy()

    def get_params(self):
        return self.weights.copy(), self.filters.copy(), self.biases.copy()

    def set_param(self, layer, src_neuron, dst_neuron, val):
        self.weights[layer][src_neuron, dst_neuron] = val

    def set_param_conv(self, layer, filter_out, filter_in, row, col, val):
        self.filters[layer][filter_out, filter_in, row, col] = val

    def set_param_conv_bias(self, layer, filter_out, val):
        self.filters[layer][filter_out] = val

    def decay_lr(self):
        self.epoch = self.epoch + 1
        if self.optimizer == "SGD":
            self.lr = max(MIN_LR, self.lr / LR_SCALE)  # cut by halve each time
        else:  # In ADAM decay lr  and regularization at each epoch
            self.lr = self.initial_lr / np.sqrt(self.epoch)
            self.reg = self.initilal_reg / np.sqrt(self.epoch)

    def momentum_change(self):
        self.momentum = min(MAX_MOMENTUM, self.momentum + MOMENTUM_SCALE)  # change momentum

    def init_weights(self, model2):
        for layer_num in range(len(self.layers) - 1):
            self.weights[layer_num] = model2.weights[layer_num].copy()
            self.accum_grads[layer_num] = model2.accum_grads[layer_num].copy()
        for layer_num in range(len(self.filter_size)):
            self.filters[layer_num] = model2.filters[layer_num].copy()
            self.filters_accum_grads[layer_num] = model2.filters_accum_grads[layer_num].copy()
            self.biases[layer_num] = model2.biases[layer_num].copy()
            self.biases_accum_grads[layer_num] = model2.biases_accum_grads[layer_num].copy()
