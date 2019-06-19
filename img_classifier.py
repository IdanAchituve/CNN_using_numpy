import numpy as np
import Network
import csv
import copy
import pickle
import matplotlib.pyplot as plt


np.random.seed(222)
NUM_CLASSES = 10
img_channels = 3
img_size = 32


def gradient_check(model, x, y):

    grads, grads_filters, grads_biases = model.get_grads()
    weights, filters, biases = model.get_params()

    eps = 0.00001

    for layer in range(len(weights)):
        for src_neuron in range(np.size(weights[layer], 0)):
            for dst_neuron in range(np.size(weights[layer], 1)):

                param_val = weights[layer][src_neuron, dst_neuron].copy()
                grad_val = grads[layer][src_neuron, dst_neuron].copy()

                # compute (loss) function value of epsilon addition to one of the parameters
                model.init_vals()
                model.set_param(layer, src_neuron, dst_neuron, param_val + eps)
                out = model.forward(x)
                upper_val = model.loss_function(y)

                # compute (loss) function value of epsilon reduction from one of the parameters
                model.init_vals()
                model.set_param(layer, src_neuron, dst_neuron, param_val - eps)  # multiply by 2 because the current value is w + epslion
                out = model.forward(x)
                lower_val = model.loss_function(y)

                # return to original state
                model.init_vals()
                model.set_param(layer, src_neuron, dst_neuron, param_val)

                numeric_grad = (upper_val - lower_val)/(2*eps)

                # Compare gradients
                reldiff = abs(numeric_grad - grad_val) / max(1, abs(numeric_grad), abs(grad_val))
                if reldiff > 1e-5:
                    print("Gradient check failed")
                    exit()

    for layer in range(len(filters)):
        for filter_out in range(np.size(filters[layer], 0)):
            for filter_in in range(np.size(filters[layer], 1)):
                for row in range(np.size(filters[layer], 2)):
                    for col in range(np.size(filters[layer], 3)):

                        param_val = filters[layer][filter_out, filter_in, row, col].copy()
                        grad_val = grads_filters[layer][filter_out, filter_in, row, col].copy()

                        # compute (loss) function value of epsilon addition to one of the parameters
                        model.init_vals()
                        model.set_param_conv(layer, filter_out, filter_in, row, col, param_val + eps)
                        out = model.forward(x)
                        upper_val = model.loss_function(y)

                        # compute (loss) function value of epsilon reduction from one of the parameters
                        model.init_vals()
                        model.set_param_conv(layer, filter_out, filter_in, row, col, param_val - eps)  # multiply by 2 because the current value is w + epslion
                        out = model.forward(x)
                        lower_val = model.loss_function(y)

                        # return to original state
                        model.init_vals()
                        model.set_param_conv(layer, filter_out, filter_in, row, col, param_val)

                        numeric_grad = (upper_val - lower_val)/(2*eps)

                        # Compare gradients
                        reldiff = abs(numeric_grad - grad_val) / max(1, abs(numeric_grad), abs(grad_val))
                        if reldiff > 1e-5:
                            print("Gradient check failed")
                            exit()

    print("Gradient check passed")


def print_img(X):

    # move the channel dimension to the last
    X = np.rollaxis(X, 1, 4)

    # plot a randomly chosen image
    img = 64
    plt.figure(figsize=(4, 2))
    plt.imshow(X[img, :, :, 0], cmap=plt.get_cmap('gray'), interpolation='none')
    plt.show()


def read_data(file, nn_params, dataset="train"):
    labels = []
    images = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if dataset == "train" or dataset == "validation":
                labels.append(int(row[0]))
            row_img = np.asarray([float(i) for i in row[1:]])
            images.append(row_img)
            if dataset == "train" and nn_params["augmentation"]:
                img = row_img.reshape([img_channels, img_size, img_size])
                # move the channel dimension to the last
                img = np.rollaxis(img, 0, 3)
                permut = np.random.randint(3)

                # flip left to right
                flippedlr_img = np.fliplr(img)
                flippedlr_img = np.rollaxis(flippedlr_img, 2, 0)
                flippedlr_img = flippedlr_img.reshape(img_channels * img_size * img_size)
                labels.append(int(row[0]))
                images.append(flippedlr_img)

                # flip up to down
                flippedud_img = np.flipud(img)
                flippedud_img = np.rollaxis(flippedud_img, 2, 0)
                flippedud_img = flippedud_img.reshape(img_channels * img_size * img_size)
                labels.append(int(row[0]))
                images.append(flippedud_img)

                # rotate by 90 degrees
                rot_img = np.transpose(img, (1, 0, 2))
                rot_img = np.rollaxis(rot_img, 2, 0)
                rot_img = rot_img.reshape(img_channels * img_size * img_size)
                labels.append(int(row[0]))
                images.append(rot_img)

    labels = np.asarray(labels)
    images = np.asarray(images).reshape(len(images), img_channels, img_size, img_size)
    return images, labels


def z_scaling(X, avg=None, std=None):

    # if X is the train set
    if avg is None:
        avg = np.mean(X, axis=0)  # mean for each feature
        std = np.std(X, axis=0)

    scaled_X = (X - avg)/std  # z score scaling
    return scaled_X, avg, std


def train_model(model, nn_params, log, exp, train_path, val_path, save_logs):

    epochs = nn_params["epochs"]
    batch_size = nn_params["train_batch_size"]

    # Initialize printing templates
    per_log_template = '    '.join('{:05d},{:09.5f},{:09.5f},{:06.5f},{:06.5f}'.split(','))
    header_template = ''.join('{:<9},{:<13},{:<13},{:<10},{:<10}'.split(','))

    # read data
    log.log("Read Train Data")
    X_train, Y_train = read_data(train_path, nn_params, "train")
    log.log("Read Validation Data")
    X_val, Y_val = read_data(val_path, nn_params, "validation")

    # apply z score scaling
    mean, std = 0, 0
    if nn_params["z_scale"]:
        X_train, mean, std = z_scaling(X_train.copy())
        X_val, _, __ = z_scaling(X_val.copy(), mean, std)

    # initialize experiment params
    best_accu = 0
    best_loss = 10000
    model_to_save = copy.deepcopy(model)
    log.log(header_template.format('Epoch', 'Trn_Loss', 'Val_Loss', 'Trn_Acc', ' Val_Acc'))

    for epoch in range(1, epochs + 1):

        # shuffle examples
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        # initialize epoch params
        cum_loss = 0.0
        correct = 0

        for ind in range(0, X_train.shape[0], batch_size):

            # set the model to train mode, zero gradients and zero activations
            model.train_time()
            model.init_vals(True)

            # run the forward pass
            batched_data = X_train[ind:ind + batch_size].copy()
            labels = Y_train[ind:ind + batch_size].copy() - 1

            # from labels to 1-hot
            labels_vec = np.eye(NUM_CLASSES)[labels].transpose()

            # forward
            #batched_data = np.arange(96).reshape(2, 3, 4, 4)
            out = model.forward(batched_data)
            loss = model.loss_function(labels_vec)

            # compute gradients and make the optimizer step
            model.backward(out, labels_vec)
            #gradient_check(model, batched_data, labels_vec)
            model.step()

            cum_loss += loss  # sum losses on all examples

            pred = np.argmax(out, axis=0)
            correct += np.sum(labels == pred)

        # decay learning rate
        if epoch % nn_params["lr_decay_epoch"] == 0:
            model.decay_lr()

        if epoch % nn_params["momentum_change_epoch"] == 0:
            model.momentum_change()

        # average train loss
        train_loss = cum_loss / X_train.shape[0]
        train_acc = correct / X_train.shape[0]

        # apply model on validation set
        val_loss, val_acc = test_model(model, nn_params, exp, X_val, Y_val, save_logs, "val", best_accu)

        # print progress
        metrics_to_print = str(per_log_template.format(epoch, train_loss, val_loss, train_acc, val_acc))
        log.log(metrics_to_print)

        # early stopping
        was_best = False
        if val_acc > best_accu or (val_acc == best_accu and val_loss < best_loss):
            model_to_save = copy.deepcopy(model)
            best_accu = val_acc
            was_best = True

        # save best model
        if save_logs and was_best:
            file_name = "/best_model" if nn_params["model"] == "CNN" else "/best_model_AE"
            with open("./logs/" + exp + file_name, 'wb') as best_model:
                pickle.dump(model_to_save, best_model)

    return model, mean, std


# make predictions on dev set
def test_model(model, nn_params, exp, X, Y, save_logs, dataset="val", best_accu=0):

    batch_size = nn_params["test_batch_size"]

    # path for saving test predictions
    test_pred_path = "./logs/" + exp + "/predictions_" + dataset + ".txt" if save_logs else None

    # initialize epoch params
    cum_loss = 0.0
    correct = 0
    for ind in range(0, X.shape[0], batch_size):
        # set the model to train mode, zero gradients and zero activations
        model.test_time()
        model.init_vals(True)

        # run the forward pass
        batched_data = X[ind:ind + batch_size].copy()

        # forward
        out = model.forward(batched_data)

        # save predictions
        pred = np.argmax(out, axis=0)
        all_preds = (pred + 1).copy() if ind == 0 else np.concatenate((all_preds, (pred + 1).copy()))

        if dataset == "val":
            # from labels to 1-hot
            labels = Y[ind:ind + batch_size].copy() - 1
            labels_vec = np.eye(NUM_CLASSES)[labels].transpose()
            loss = model.loss_function(labels_vec)

            # calc loss and accuracy
            cum_loss += loss
            correct += np.sum(labels == pred)

    set_loss = cum_loss / X.shape[0]
    accuracy = correct / X.shape[0]

    # write predictions to file
    if save_logs and (dataset == "test" or (dataset == "val" and accuracy > best_accu)):
        np.savetxt(test_pred_path, all_preds.transpose(), fmt='%d')

    return set_loss, accuracy


def classifier(nn_params, log, exp, train_path, val_path, test_path, save_logs):

    # create model and train it
    model = Network.CNN(nn_params)
    if nn_params["load_model"] is not None:
        with open(nn_params["load_model"], 'rb') as pickle_file:
            model2 = pickle.load(pickle_file)
        model.init_weights(model2)

    model, mean, std = train_model(model, nn_params, log, exp, train_path, val_path, save_logs)

    # test model
    X_test, Y_test = read_data(test_path, nn_params, "test")

    # apply z score scaling
    if nn_params["z_scale"]:
        X_test, _, __ = z_scaling(X_test.copy(), mean, std)

    test_model(model, nn_params, exp, X_test, Y_test, save_logs, "test")