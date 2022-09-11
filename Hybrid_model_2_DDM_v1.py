"""A code to try to use PyTorch get x,y-coordinates on a circle, given the angle and radius of the circle
Took this applications so it gives data close to movement in a orbit, and gives the opportunity to use two input/output
    arguments in the NN.
Data from NN is compared to the real solution.

PS: Tensorboard is initialized by downloading through pip and running "tensorboard --logdir=runs".
The SummaryWriter needs to save to the same directory, so the f-string in this script must be changed"""

# TODO
# Set up Google Colab
# Test residual terms for two planets
# Test 5 steps in training phase
# Test three points as input in NN
# Expand to all the planets
# Save model
# Run for same time interval as the DDM and PBM
# Optimize DDM
# Optimize Hybrid
# Try RNN


import math
import random
import sys
import time
from datetime import datetime

import Hybrid_model_1_PBM_v2 as pbm
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits import mplot3d
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def main():
    nasa_data = np.load('nasa_255y_1h.npy').tolist()[:21000]
    pbm_data = np.load('pbm_255y_1d_1h.npy').tolist()[:21000]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    M = 24
    N = 6  # Number of inputs
    L = 3
    dt = 1

    # The following procedure consists of 4 steps:
    # 1. Construct data: data and dataset-class
    # 2. Construct data driven solver: hyperparameters, neural network architecture, per-epoch training structure,
    #                                  training-loop, validation dataset pass, running/average loss & accuracy reporting
    # 3. Create, solve and report on test problems using the data-driven model

    #######################################################################################
    # Step 1 - Construct data
    #######################################################################################

    # Create data
    class DataConstructor:
        def __init__(self, nasa_data, pbm_data):
            # Import data
            # Normalize data
            pbm_data_pre_norm = [x[0:0 + N] for x in pbm_data]
            nasa_data_pre_norm = [x[0:0 + N] for x in nasa_data]
            output_dataset = [[x[i] - y[i] for i in range(N)]
                              for x, y in zip(nasa_data_pre_norm[1 * M:], pbm_data_pre_norm[:-1 * M])]

            # Normalize data
            if True:
                self.avg = [np.average([x[i] for x in nasa_data_pre_norm]) for i in range(N)]
                self.std = [np.std([x[i] for x in nasa_data_pre_norm]) for i in range(N)]
                self.avg_res = [np.average([x[i] for x in output_dataset]) for i in range(N)]
                self.std_res = [np.std([x[i] for x in output_dataset]) for i in range(N)]

            # Create dedicated input and output data
            # Create dedicated input and output data
            input_dataset = [[a, b, c, d, e, f, g, h] for a, b, c, d, e, f, g, h in zip(nasa_data[:-7*M],
                                                                            nasa_data[1*M:-6*M],
                                                                            nasa_data[2*M:-5*M], nasa_data[3*M:-4*M],
                                                                            nasa_data[4*M:-3*M], nasa_data[5*M:-2*M],
                                                                            nasa_data[6*M:-1*M], nasa_data[7*M:])]

            # Reduce input and output dataset for training and validation
            input_dataset = input_dataset[:3 * math.ceil(len(input_dataset) / 4)]

            # Store for later reference
            self.input = np.array(input_dataset)
            self.data_size = len(self.input)

            # For splitting data into training and validation; number indicates percent used for training
            self.split_index = math.ceil(0.80 * self.data_size)

        def input_data(self):
            return self.input

        def output_data(self):
            return self.output

        def __len__(self):
            return self.data_size

        def plot(self):

            # Plot data set
            plot = plt.axes(projection='3d')

            # Data for a three-dimensional line - prepare for 9 planets
            x = [[] for i in range(int(9))]
            y = [[] for i in range(int(9))]
            z = [[] for i in range(int(9))]
            # Plot only 1 planet orbit
            for j in range(int(1)):
                for i in range(len(self.nasa_data)):
                    x[j].append(self.nasa_data[i][j * 6])
                    y[j].append(self.nasa_data[i][j * 6 + 1])
                    z[j].append(self.nasa_data[i][j * 6 + 2])
                plot.plot(x[j], y[j], z[j])
            x_pbm = [[] for i in range(int(9))]
            y_pbm = [[] for i in range(int(9))]
            z_pbm = [[] for i in range(int(9))]
            # Plot only 1 planet orbit
            for j in range(int(1)):
                for i in range(len(self.nasa_data)):
                    x_pbm[j].append(self.pbm_data[i][j * 6])
                    y_pbm[j].append(self.pbm_data[i][j * 6 + 1])
                    z_pbm[j].append(self.pbm_data[i][j * 6 + 2])
                plot.plot(x_pbm[j], y_pbm[j], z_pbm[j])
            plt.show()

    # Create Dataset class
    class PositionDataset(Dataset):
        """ Dataset for circle data."""

        def __init__(self, input):
            # data loading
            self.input = torch.from_numpy(input)
            self.n_samples = input.shape[0]

        def __getitem__(self, index):
            return self.input[index]

        def __len__(self):
            return self.n_samples

    # Create Dataset object from generated data
    data = DataConstructor(nasa_data, pbm_data)
    split_index = data.split_index
    input = data.input_data()

    # Create training set and validation set
    training_dataset = PositionDataset(input[:split_index, :])
    validation_dataset = PositionDataset(input[split_index:, :])

    # Plot data
    # data.plot()

    #######################################################################################
    # Step 2 - Construct data driven solver
    #######################################################################################

    # Hyperparameters controlling the model optimization
    # Can add values to lists to run through different hyperparameter values for optimization of HPs
    # Ex: lr = [0.1, 0.01, 0.001], bs = [2, 64, 100, 1000]
    learning_rates = [0.0005]  # [0.005, 0.01, 0.0001, 0.0005]
    batch_sizes = [64]  # [32, 64, 128, 256]
    epochs = 100
    l2_lambda = 0.0
    weight_decay = 0.0
    dropouts = [0.0]  # [0.0, 0.01, 0.05, 0.1, 0.5]
    patience = 20

    # Define NN architecture - Current activation function: ReLU
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.pooling = nn.MaxPool1d(7)

            self.linear_relu_stack_1_pos = nn.Sequential(
                nn.Linear(9, 50),  # Take (x, y, z, vx, vy, vz) @ t_n as input
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(50, 10),  # Return (x, y, z, vx, vy, vz) @ t_n+1 as output
            )

            self.linear_relu_stack_1_vel = nn.Sequential(
                nn.Linear(9, 50),  # Take (x, y, z, vx, vy, vz) @ t_n as input
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(50, 10),  # Return (x, y, z, vx, vy, vz) @ t_n+1 as output
            )

            self.linear_relu_stack_2_pos = nn.Sequential(
                nn.Linear(9, 50),  # Take (x, y, z, vx, vy, vz) @ t_n as input
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(50, 10),  # Return (x, y, z, vx, vy, vz) @ t_n+1 as output
            )

            self.linear_relu_stack_2_vel = nn.Sequential(
                nn.Linear(9, 50),  # Take (x, y, z, vx, vy, vz) @ t_n as input
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(50, 10),  # Return (x, y, z, vx, vy, vz) @ t_n+1 as output
            )

            self.linear_relu_stack_11_pos = nn.Sequential(
                nn.Linear(12, 10),  # Take (x, y, z, vx, vy, vz) @ t_n as input
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(10, 3),  # Return (x, y, z, vx, vy, vz) @ t_n+1 as output
            )

            self.linear_relu_stack_11_vel = nn.Sequential(
                nn.Linear(12, 10),  # Take (x, y, z, vx, vy, vz) @ t_n as input
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(10, 3),  # Return (x, y, z, vx, vy, vz) @ t_n+1 as output
            )

            self.linear_relu_stack_12_pos = nn.Sequential(
                nn.Linear(12, 10),  # Take (x, y, z, vx, vy, vz) @ t_n as input
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(10, 3),  # Return (x, y, z, vx, vy, vz) @ t_n+1 as output
            )

            self.linear_relu_stack_12_vel = nn.Sequential(
                nn.Linear(12, 10),  # Take (x, y, z, vx, vy, vz) @ t_n as input
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(10, 3),  # Return (x, y, z, vx, vy, vz) @ t_n+1 as output
            )

        def forward(self, x1_pos, x1_vel, x2_pos, x2_vel):
            logits1_pos = self.linear_relu_stack_1_pos(x1_pos)
            logits1_vel = self.linear_relu_stack_1_vel(x1_vel)
            logits2_pos = self.linear_relu_stack_2_pos(x2_pos)
            logits2_vel = self.linear_relu_stack_2_vel(x2_vel)

            logits = torch.cat((logits1_pos, logits1_vel, logits2_pos, logits2_vel), -1)
            logits = torch.unsqueeze(logits, dim=0)
            pool = self.pooling(logits)
            pool = torch.squeeze(pool, dim=0)
            logits1_pos = self.linear_relu_stack_11_pos(torch.cat((logits1_pos, pool), -1))
            logits1_vel = self.linear_relu_stack_11_vel(torch.cat((logits1_vel, pool), -1))
            logits2_pos = self.linear_relu_stack_12_pos(torch.cat((logits2_pos, pool), -1))
            logits2_vel = self.linear_relu_stack_12_vel(torch.cat((logits2_vel, pool), -1))

            logits = torch.cat((logits1_pos, logits1_vel, logits2_pos, logits2_vel), -1)
            return logits

    # Define one epoch of the training w/ running results reported in TensorBoard
    def train_one_epoch(epoch_index, l2_lambda):
        running_loss = 0.  # Running loss for reporting during training
        running_correct = 0.  # Amount of correct output pairs within a given range
        losses = 0.  # Total loss
        num_correct = 0.  # Total amount of correct output pairs
        num_total = 0

        # Here, we use enumerate(training_loader) instead of iter(training_loader)
        # so that we can track the batch index and do some intra-epoch reporting
        for i, data in enumerate(training_dataloader):

            # Every data instance is an input + label pair
            input_torch = data
            input_torch = input_torch.float()

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            preds, output_torch = pbm_interaction(input_torch)

            # Compute the loss
            loss = loss_fn(preds, output_torch)

            # Add regularization
            l2_norm = sum(p.pow(2.).sum() for p in dd_model.parameters())
            loss += l2_lambda * l2_norm

            # Compute gradients for each parameter
            loss.backward()

            # Adjust learning weights and biases
            optimizer.step()

            # Gather data and report
            # Check if output is within 10% of real value; checking whole vectors for each planet, not single values
            check_preds = torch.isclose(preds, output_torch, atol=0, rtol=0.1)
            for k in range(len(check_preds)):
                if check_preds[k].sum() == 3:
                    running_correct += 1
                    num_correct += 1

            running_loss += loss.item()
            losses += loss.item()

            if (i + 1) % 10 == 0:
                # Report data for every 100 batch
                step = epoch_index * len(training_dataloader) + i + 1
                writer.add_scalar('Running Loss', running_loss / 100, step + 1)
                writer.add_scalar('Running Accuracy', running_correct / 100 / len(output_torch), step + 1)
                running_loss = 0.
                running_correct = 0.
            num_total += len(output_torch)
        # Returns average loss and accuracy for one epoch
        return losses / (i + 1), num_correct / num_total

    def validate_one_epoch():
        # Do one pass through model with validation dataset
        v_losses = 0.  # Total loss in validation set
        v_num_correct = 0.  # Total amount of correct output pairs in validation set
        v_num_total = 0  # Total amount of data pairs (need due to uneven data sizes)

        # Here, we use enumerate(training_loader) instead of iter(training_loader)
        # so that we can track the batch index and do some intra-epoch reporting

        for i, v_data in enumerate(validation_dataloader):
            # Every data instance is an input + label pair
            v_input_torch = v_data
            v_input_torch = v_input_torch.float()

            # Make predictions for this batch
            v_preds, v_output_torch = pbm_interaction(v_input_torch)

            # Compute the loss
            v_losses += loss_fn(v_preds, v_output_torch).item()

            # Gather data and report
            # Check if output is within 10% of real value; checking whole vectors for each planet, not single values
            v_check_preds = torch.isclose(v_preds, v_output_torch, atol=0, rtol=0.1)
            for k in range(len(v_check_preds)):
                if v_check_preds[k].sum() == 3:
                    v_num_correct += 1

            v_num_total += len(v_output_torch)
        # Returns average loss and accuracy for one epoch

        return v_losses / (i + 1), v_num_correct / v_num_total

    def pbm_interaction(inputs):
        preds = []
        sols = []
        for i, nasa in enumerate(inputs):
            pbm_input = nasa[0].detach().numpy()  # Initial value

            for j in range(len(nasa)-1):
                # Create input for PBM with right format and passes it through PBM
                pbm_input_packed = [pbm_input[i:i + 6] for i in range(0, len(pbm_input), 6)]
                pbm_output, H = pbm.main(pbm_input_packed, dt)

                # Normalize
                if True:
                    # Skips normalization of sun, insert sun values at start of list
                    pbm_input_1 = pbm_input[0:0 + N]
                    pbm_output_1 = pbm_output[0:0 + N]
                    nasa_data_1 = nasa[j+1][0:0 + N]
                    sol = torch.from_numpy(np.array(nasa_data_1) - np.array(pbm_output_1)).float()
                    pbm_input_norm = [(pbm_input_1[j] - data.avg[j]) / data.std[j] for j in range(N)]
                    pbm_output_norm = [(pbm_output_1[j] - data.avg[j]) / data.std[j] for j in range(N)]
                    sol = torch.stack([(sol[j] - data.avg_res[j]) / data.std_res[j] for j in range(N)])
                sols.append(sol)

                # Convert to torch format; input = (x_0, x_1), output = (approx x_1 - exact x_1)
                # (For one planet)
                input_torch = torch.from_numpy(np.array([[pbm_input_norm[i], pbm_output_norm[i]]
                                                         for i in range(len(pbm_input_norm))]).flatten()).float()

                # Compute prediction of residual for each entry of state vectors for each planet
                t_x1 = input_torch[0:6]
                t_x2 = input_torch[6:12]
                pred_torch = dd_model(t_x1, t_x2)
                # Compute prediction of residual for each entry of state vectors for each planet)
                preds.append(pred_torch)

                # Converts back to plottable format
                residual = list(pred_torch.detach().numpy())

                # Re-normalize
                if True:
                    for i in range(len(residual)):
                        residual[i] = (residual[i] * data.std_res[i]) + data.avg_res[i]
                residuals = residual + [0] * 42

                # Create new input; adds pred residual to PBM approximated value
                pbm_input = [a + b for a, b in zip(pbm_output, residuals)]

        return torch.stack(preds), torch.stack(sols)

    start_time = time.time()  # Initiates timer
    # Training loop w/ a pass trough a validation dataset and averaged results per epoch reported in TensorBoard
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for dropout in dropouts:
                # Build a corresponding data loader to dataset
                training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=0)
                validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False,
                                                   num_workers=0)
                # Initiates the NN
                dd_model = NeuralNetwork().to("cpu")  # here the use of GPU instead of CPU can be controlled

                # Definition of the minimization problem
                loss_fn = nn.MSELoss()

                # Optimizer algorithm (current:ADAM)
                optimizer = torch.optim.Adam(dd_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                # Initiate SummaryWriter for reporting results in TensorBoard
                writer = SummaryWriter(f'runs/Hybrid/BatchSize {batch_size} LR {learning_rate} DO {dropout}')

                best_avg_loss = 1_000_000
                best_avg_vloss = 1_000_000
                best_avg_acc = 0
                best_avg_vacc = 0
                triggertimes = 0
                for epoch in range(epochs):
                    print(f'EPOCH: {epoch + 1}')

                    # Turn on gradient tracking and train one epoch
                    dd_model.train(True)
                    avg_loss, avg_acc = train_one_epoch(epoch, l2_lambda)

                    # Turn of gradient tracking for reporting on validation dataset
                    dd_model.train(False)
                    avg_vloss, avg_vacc = validate_one_epoch()

                    # Log the loss and accuracy averaged per epoch for both training and validation
                    # Log hyperparameters (if one is doing hyperparameter optimization)
                    writer.add_scalars('Training vs. Validation Loss',
                                       {'Training': avg_loss, 'Validation': avg_vloss},
                                       epoch + 1)
                    writer.add_scalars('Training vs. Validation Accuracy',
                                       {'Training': avg_acc, 'Validation': avg_vacc},
                                       epoch + 1)
                    if avg_loss < best_avg_loss:
                        best_avg_loss = avg_loss
                    if avg_vloss < best_avg_vloss:
                        best_avg_vloss = avg_vloss
                        triggertimes = 0
                    else:
                        triggertimes += 1
                        if triggertimes >= patience:
                            print('Early stop! Training complete')
                            break
                    if avg_acc > best_avg_acc:
                        best_avg_acc = avg_acc
                    if avg_vacc > best_avg_vacc:
                        best_avg_vacc = avg_vacc

                    writer.flush()

                writer.add_hparams(hparam_dict={"lr": learning_rate, "bsize": batch_size, "dropout": dropout},
                                   metric_dict={"accuracy": best_avg_acc, "loss": best_avg_loss,
                                                "vaccuracy": best_avg_vacc, "vloss": best_avg_vloss})
                writer.flush()

        print(f"BatchSize {batch_size} LR {learning_rate} DO {dropout}:"
              f"\nAccuracy: {best_avg_acc}"
              f"\nLoss: {best_avg_loss}"
              f"\nValidation Accuracy: {best_avg_vacc} "
              f"\nValidation Loss: {best_avg_vloss}")

        print("Training complete")

    #######################################################################################
    # Step 3 - Create, solve and report on test problems using the data-driven model
    #######################################################################################

    def test(nasa_data, dt):

        # Set up test problems
        dd_model.eval()  # Sets NN to evaluation mode

        preds = []
        res = []
        num_correct = 0
        total_loss = 0

        pbm_input = nasa_data[0]  # Initial value
        nasa_data = nasa_data[24::24]  # Exact solution

        for i in range(len(nasa_data)):
            # Create input for PBM with right format and passes it through PBM
            pbm_input_packed = [pbm_input[i:i + 6] for i in range(0, len(pbm_input), 6)]
            pbm_output, H = pbm.main(pbm_input_packed, dt)

            # Normalize
            if True:
                # Skips normalization of sun, insert sun values at start of list
                pbm_input_1 = pbm_input[0:0 + N]
                pbm_output_1 = pbm_output[0:0 + N]
                nasa_data_1 = nasa_data[i][0:0 + N]
                pbm_input_norm = [(pbm_input_1[j] - data.avg[j]) / data.std[j] for j in range(N)]
                pbm_output_norm = [(pbm_output_1[j] - data.avg[j]) / data.std[j] for j in range(N)]
                nasa_input_norm = [(nasa_data_1[j] - data.avg[j]) / data.std[j] for j in range(N)]

            # Convert to torch format; input = (x_0, x_1), output = (approx x_1 - exact x_1)
            input_torch = torch.from_numpy(np.array([[pbm_input_norm[i], pbm_output_norm[i]]
                                                     for i in range(len(pbm_input_norm))]).flatten()).float()
            output_torch = torch.from_numpy(np.array(pbm_output_norm) - np.array(nasa_input_norm)).float()

            print(input_torch)
            # Compute prediction of residual for each entry of state vectors for each planet
            t_x1 = input_torch[0:6]
            print(t_x1)
            t_x2 = input_torch[6:12]
            pred_torch = dd_model(t_x1, t_x2)
            # Compute prediction of residual for each entry of state vectors for each planet

            # Converts back to plottable format
            residual = list(pred_torch.detach().numpy())

            # Re-normalize
            if True:
                for i in range(len(residual)):
                    residual[i] = (residual[i] * data.std_res[i]) + data.avg_res[i]
            residuals = residual + [0] * 45
            res.append(np.array(residual))

            # Create new input; adds pred residual to PBM approximated value
            pbm_input = [a + b for a, b in zip(pbm_output, residuals)]
            preds.append(pbm_input)
            # Calculate loss
            loss = loss_fn(pred_torch, output_torch)
            total_loss += loss.item()

        # Plot data set
        plot = plt.axes(projection='3d')

        # Data for a three-dimensional line - prepare for 9 planets
        x = [[] for i in range(int(9))]
        y = [[] for i in range(int(9))]
        z = [[] for i in range(int(9))]
        # Plot 9 planet orbits
        for j in range(int(8)):
            for i in range(len(preds)):
                x[j].append(preds[i][j * 6])
                y[j].append(preds[i][j * 6 + 1])
                z[j].append(preds[i][j * 6 + 2])
            plot.plot(x[j], y[j], z[j])
        plt.show()

        print('Average test loss:', total_loss / (i + 1),
              'Average test accuracy', num_correct / (2 * (i + 1)))

        return preds, nasa_data, res

    # Test 1
    preds, output, res = test(nasa_data, dt)
    np.save('Hybrid_app2.npy', [preds, output, res])

    print("time elapsed: {:.2f}s".format(time.time() - start_time))  # Prints running time for the code


if __name__ == "__main__":
    main()
