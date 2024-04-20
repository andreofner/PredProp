import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.io import loadmat
import matplotlib.pyplot as plt
from itertools import cycle
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import urllib.request
import copy
import time
import pickle

plt.rcParams["figure.dpi"] = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 4
lamb = 0.05


class MNIST:
    def __init__(self, batch_size, logit_transform=False):
        """[-1, 1, 28, 28]"""
        self.logit_transform = logit_transform
        directory = "./datasets/MNIST"
        if not os.path.exists(directory):
            os.makedirs(directory)

        kwargs = (
            {"num_workers": num_workers, "pin_memory": True}
            if torch.cuda.is_available()
            else {}
        )
        self.train_loader = DataLoader(
            datasets.MNIST(
                "./datasets/MNIST",
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs,
        )

        self.test_loader = DataLoader(
            datasets.MNIST(
                "./datasets/MNIST", train=False, transform=transforms.ToTensor()
            ),
            batch_size=batch_size,
            shuffle=False,
            **kwargs,
        )

        self.dim = [1, 28, 28]

        train = torch.stack(
            [data for data, _ in list(self.train_loader.dataset)], 0
        ).cuda()
        train = train.view(train.shape[0], -1)
        if self.logit_transform:
            train = train * 255.0
            train = (train + torch.rand_like(train)) / 256.0
            train = lamb + (1 - 2.0 * lamb) * train
            train = torch.log(train) - torch.log(1.0 - train)

        self.mean = train.mean(0)
        self.logvar = torch.log(torch.mean((train - self.mean) ** 2)).unsqueeze(0)

    def preprocess(self, x):
        if self.logit_transform:
            # apply uniform noise and renormalize
            x = x.view([-1, np.prod(self.dim)]) * 255.0
            x = (x + torch.rand_like(x)) / 256.0
            x = lamb + (1 - 2.0 * lamb) * x
            x = torch.log(x) - torch.log(1.0 - x)
            return x - self.mean
        else:
            return x.view([-1, np.prod(self.dim)]) - self.mean

    def unpreprocess(self, x):
        if self.logit_transform:
            x = x + self.mean
            x = torch.sigmoid(x)
            x = (x - lamb) / (1.0 - 2.0 * lamb)
            return x.view([-1] + self.dim)
        else:
            return (x + self.mean).view([-1] + self.dim)


class BaseDecoder(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim):
        """PC layer with multi-layer dense NN"""
        super().__init__()

        # decoder weights
        self.linear_hidden0 = nn.Linear(z_dim, h_dim, bias=False)
        self.linear_hidden1 = nn.Linear(h_dim, h_dim, bias=False)
        self.linear_mu = nn.Linear(h_dim, x_dim, bias=False)

        # initialise weights
        torch.nn.init.xavier_normal_(self.linear_hidden0.weight)
        torch.nn.init.xavier_normal_(self.linear_hidden1.weight)
        torch.nn.init.xavier_normal_(self.linear_mu.weight)

        self.weights = [self.linear_hidden0, self.linear_hidden1, self.linear_mu]

    def forward(self, x):
        """
        Compute prediction, store covariance of intermediate inputs and intermediate predictions
        """
        self.input_covars = []  # input activity covariance
        self.outputs = []  # store intermediates to compute their covariance later
        self.input_covars.append(
            torch.matmul(x.unsqueeze(-1), torch.transpose(x.unsqueeze(-1), 1, 2))
            .mean(0)
            .data
        )

        # input layer
        x = F.relu(self.linear_hidden0(x))
        x.retain_grad()  # make sure intermediates also get gradients
        self.outputs.append(x)
        self.input_covars.append(
            torch.matmul(x.unsqueeze(-1), torch.transpose(x.unsqueeze(-1), 1, 2))
            .mean(0)
            .data
        )
        # hidden layer
        x = F.relu(self.linear_hidden1(x))
        x.retain_grad()
        self.outputs.append(x)
        self.input_covars.append(
            torch.matmul(x.unsqueeze(-1), torch.transpose(x.unsqueeze(-1), 1, 2))
            .mean(0)
            .data
        )
        # output layer
        x = self.linear_mu(x)
        x.retain_grad()
        self.outputs.append(x)
        self.input_covars.append(
            torch.matmul(x.unsqueeze(-1), torch.transpose(x.unsqueeze(-1), 1, 2))
            .mean(0)
            .data
        )
        return x

    def step(self, lr=0.9, damp_in=0.1, damp_out=0.1):
        """
        Weight update with covariance of inputs and gradients
        """
        for i in range(len(self.weights)):
            # input covariance
            input_covar = self.input_covars[i]
            eye_in = torch.eye(input_covar.shape[-1]).cuda()
            right = torch.inverse(input_covar + damp_in * eye_in)  # input precision

            # error covariance
            grad = self.weights[i].weight.grad.data
            self.error_covars = []
            for x in self.outputs:
                self.error_covars.append(
                    torch.matmul(
                        x.grad.data.unsqueeze(-1),
                        torch.transpose(x.grad.data.unsqueeze(-1), 1, 2),
                    )
                    .mean(0)
                    .data
                )
            error_covar = self.error_covars[i]
            eye_out = torch.eye(error_covar.shape[-1]).cuda()
            left = torch.inverse(error_covar + damp_out * eye_out)  # error precision

            # update weights
            self.weights[i].weight.data -= lr * torch.matmul(
                torch.matmul(left, grad), right
            )


class PC(nn.Module):
    """Predictive coding network"""

    def __init__(self, obs_size=784, prior_size=64, activation=F.relu):
        super().__init__()

        # generative networks
        self.dec3 = BaseDecoder(z_dim=prior_size, x_dim=latent_dim, h_dim=hidden_dim)
        self.dec2 = BaseDecoder(z_dim=latent_dim, x_dim=latent_dim, h_dim=hidden_dim)
        self.dec1 = BaseDecoder(z_dim=latent_dim, x_dim=obs_size, h_dim=hidden_dim)
        self.decoders = [self.dec1, self.dec2, self.dec3]

        # prediction error precision
        self.error_precision = [None for _ in range(len(self.decoders) + 1)]

        # logging
        self.log_errors, self.log_errors_test = [], []
        self.log_error_posterior, self.log_error_posterior_test = [], []

    def predictive_dist(self):
        """Predict from inferred state in each PC layer"""
        mus = self.mus
        pred_post = [
            self.decoders[l].forward(mus[l + 1].detach())
            for l in range(len(self.decoders))
        ]
        return pred_post

    def backward_pass(self, prior, x):
        """Top-down predicted prior (pass through entire network)"""
        pred_global = [prior]
        for l in reversed(range(len(self.decoders))):
            tmp = self.decoders[l].forward(pred_global[-1].detach())
            pred_global.append(tmp)
        pred_global = list(reversed(pred_global))

        # initialise states
        mus_TD = pred_global  # top-down predicted prior
        mus = [x] + pred_global[1:]  # prior inferred state

        return mus_TD, mus

    def forward(self, x, prior):
        """Iterative inference on observation"""
        prior = prior * 0 + 0.00001
        mus_TD, mus = self.backward_pass(prior, x)

        # iterative inference
        for step in range(inference_steps):
            prior = mus[-1]

            for l, (target, mu, mu_TD, decoder) in enumerate(
                zip(mus[:-1], mus[1:], mus_TD[1:], self.decoders)
            ):
                mu = mu.clone().detach().requires_grad_()  # inferred state
                mu_TD = mu_TD.clone().detach()  # top-down predicted state
                mu.grad = torch.zeros_like(mu)  # initialise state gradient

                # predict
                pred = decoder.forward(mu)  # prediction

                # error
                error = pred - target.view(pred.shape).detach()  # bottom-up error
                error_TD = 0.1 * (mu - mu_TD.detach())  # top-down error
                error_ = error * error
                error_TD_ = error_TD * error_TD

                # gradients
                error_.mean(1).backward(
                    gradient=torch.ones_like(error_.mean(1)), retain_graph=True
                )
                error_TD_.mean(1).backward(
                    gradient=torch.ones_like(error_TD_.mean(1)), retain_graph=True
                )

                # precision weighting
                if inference_NGD:

                    def precision(error, damp):
                        error_cov = torch.matmul(
                            error.unsqueeze(-1), error.unsqueeze(-1).transpose(-2, -1)
                        ).mean(0, keepdims=True)
                        error_prec = torch.inverse(
                            error_cov + torch.eye(error_cov.shape[1]).cuda() * damp
                        )
                        return error_prec

                    error_prec = precision(mu.grad, damp_err_inf)
                    state_prec = precision(mu.data, damp_act_inf)

                    mu.grad.data = torch.matmul(
                        state_prec, torch.matmul(error_prec, (mu.grad).unsqueeze(-1))
                    ).squeeze(-1)

                # update state
                mus[l + 1] -= inference_lr * mu.grad
                mu.grad = torch.zeros_like(mu)

        # top-down predicted posterior
        pred_post = [
            self.decoders[l].forward(mus[l + 1].detach())
            for l in range(len(self.decoders))
        ]

        # logging: top-down predicted posterior (pass through entire network)
        pred_global = mus[-1]
        for l in reversed(range(len(self.decoders))):
            pred_global = self.decoders[l].forward(pred_global.detach())

        # logging: errors and inferred states
        self.error_global_pred = (
            torch.mean(((pred_global - mus[0].detach()) ** 2), dim=1)
        ).mean()
        self.errors = [
            (torch.mean(((pred_post[l] - mus[l].detach()) ** 2), dim=1)).mean()
            for l in range(len(self.decoders))
        ]
        self.error = torch.stack(self.errors).sum()
        self.mus = mus

        # logging
        if self.test:
            self.log_errors_test.append([e.cpu().detach().numpy() for e in self.errors])
            self.log_error_posterior_test.append(
                (torch.mean(((pred_global - x) ** 2), dim=1))
                .mean()
                .cpu()
                .detach()
                .numpy()
            )
        else:
            self.log_errors.append([e.cpu().detach().numpy() for e in self.errors])
            self.log_error_posterior.append(
                (torch.mean(((pred_global - x) ** 2), dim=1))
                .mean()
                .cpu()
                .detach()
                .numpy()
            )

        return pred_post, pred_post[0], pred_global, mus


def train(model, data, test_data, dataset, max_updates):
    updates = 0

    if WEIGHTS_NGD:
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
    else:
        opt = torch.optim.Adam(
            model.parameters(), lr=LR_weights, betas=(beta_1, beta_2)
        )
        print("beta_1", beta_1, "beta_2", beta_2)

    while True:
        for batch, (x, y) in enumerate(data):

            # log test batch
            if updates % test_interval == 0:
                for batch, (x_test, y_test) in enumerate(test_data):
                    model.test = True
                    x_test = dataset.preprocess(x_test.to(device)).view([-1, obs_size])
                    prior_test = torch.zeros([x_test.shape[0], latent_dim]).cuda()
                    pred_post, pred_post[0], pred_global, mus = model.forward(
                        x_test, prior_test
                    )  # infer state
                    break

            # train batch
            model.test = False
            x = dataset.preprocess(x.to(device)).view([-1, obs_size])
            prior = torch.zeros([x.shape[0], latent_dim]).cuda()

            # iterative update of state
            opt.zero_grad()
            pred_post, pred_post[0], pred_global, mus = model.forward(x, prior)

            # predict
            opt.zero_grad()
            p_post = vae.predictive_dist()

            # prediction errors
            errors = [
                p - m.detach() for p, m in zip(p_post, vae.mus)
            ]  # prediction - inferred mean
            [(e * e).mean().backward() for e in errors]

            # update weights
            if WEIGHTS_NGD:
                [
                    d.step(lr=LR_weights, damp_in=damp_in, damp_out=damp_out)
                    for d in vae.decoders
                ]  # PredProp 0.1 0.1
            else:
                opt.step()  # Adam

            if updates + 1 >= max_updates:
                return model

            updates += 1


def visualize(vae, data, dataset, examples=1, plot_target=True):

    for batch, (x, y) in enumerate(data):
        x = x.to(device)
        x = dataset.preprocess(x).view([-1, obs_size])
        prior = torch.zeros([x.shape[0], latent_dim]).cuda()

        if plot_target:
            # plot ground truth
            plt.figure(figsize=(2, 2))
            plt.imshow(
                (dataset.unpreprocess(x).view([-1, obs_size])[0])
                .detach()
                .cpu()
                .reshape([int(np.sqrt(obs_size)), int(np.sqrt(obs_size))])
            )
            plt.colorbar()
            plt.title("Target")
            plt.show()

        pred_post, p_post, pred_global, mus = vae.forward(x, prior)

        # plot reconstruction
        plt.figure(figsize=(2, 2))
        plt.imshow(
            (dataset.unpreprocess(pred_global[0]).view([-1, obs_size]))
            .detach()
            .cpu()
            .reshape([int(np.sqrt(obs_size)), int(np.sqrt(obs_size))])
        )
        plt.colorbar()
        plt.title("Global posterior")
        plt.show()
        if batch == examples - 1:
            break


def plot_training(models, run_names, DS_name):
    linecycler = ["-", "--", "-."]
    fig = plt.figure()
    ax = plt.subplot(111)
    for vae, run_name in zip(models, run_names):  # train errors
        ls = "-." if "True" in run_name else "-"
        ax.plot(vae.log_error_posterior, label=run_name, linestyle=ls)
    for vae, run_name in zip(models, run_names):  # test errors
        ax.scatter(
            np.asarray(range(len(vae.log_error_posterior_test))) * test_interval,
            vae.log_error_posterior_test,
            color="gray",
        )
    ax.grid()
    ax.legend(loc="upper right")
    ax.set_ylabel("Mean squared error")
    ax.set_xlabel("Updates")
    plt.title(DS_name)
    plt.savefig(f"{DS}_training.pdf")
    plt.show()


""" PCNs with multi-layer dense NN in each PC layer """

# Experiment
DS, DS_name = MNIST, "MNIST"
updates = 1000
test_interval = 50
hidden_dim = 256
latent_dim = 64
inference_lr = 0.9
inference_steps = 20
obs_size = 28 * 28
RUNS = 1
BATCH_SIZE = 32

# PredProp optimizer parameters
damp_in = 0.005
damp_out = 0.1
damp_err_inf = 0.9
damp_act_inf = 0.9

# Baseline optimizer parameters
beta_1 = 0.9
beta_2 = 0.999

# load dataset
dataset = DS(batch_size=BATCH_SIZE, logit_transform=False)
train_data = dataset.train_loader
test_data = dataset.test_loader

# train models
models_all, run_names_all = [], []
models, run_names = [], []
for WEIGHTS_NGD, OPT_NAME in zip([True, False], ["PC-PredProp", "PC-Adam"]):
    LR_weights_list = [0.9] if WEIGHTS_NGD else [0.001]
    for LRW, LR_weights in enumerate(LR_weights_list):
        for inference_NGD in [True, False]:
            for run in range(RUNS):
                run_names.append(f"{OPT_NAME} ({LR_weights}, {inference_NGD})")
                print(DS_name, run_names[-1])
                vae = PC(obs_size=obs_size, prior_size=latent_dim).to(device)
                vae = train(vae, train_data, test_data, dataset, updates)
                models.append(vae)

models_all.append(models)
run_names_all.append(run_names)

# plot training progress
plot_training(models, run_names, DS_name)

# visualize reconstructions
# for model in models:
#    visualize(model, test_data, dataset)

"""
Experiments with a single layer decoder network in each PCN layer
"""


class BaseDecoder_single(BaseDecoder):
    def __init__(self, z_dim, x_dim, h_dim):
        """Each PC layer has a dense decoder DNN with one layer"""
        super().__init__(z_dim, x_dim, 32)

        # decoder weights
        self.linear_hidden0 = nn.Linear(z_dim, x_dim, bias=False)
        torch.nn.init.xavier_normal_(self.linear_hidden0.weight)
        self.weights = [self.linear_hidden0]

    def forward(self, x):
        """Compute prediction and input covariance"""
        self.input_covars = []  # input activity covariance
        self.outputs = []  # store intermediates to compute their covariance later

        self.input_covars.append(
            torch.matmul(x.unsqueeze(-1), torch.transpose(x.unsqueeze(-1), 1, 2))
            .mean(0)
            .data
        )

        x = F.tanh(self.linear_hidden0(x))
        x.retain_grad()
        self.outputs.append(x)
        self.input_covars.append(
            torch.matmul(x.unsqueeze(-1), torch.transpose(x.unsqueeze(-1), 1, 2))
            .mean(0)
            .data
        )
        return x


class PC_single(PC):
    """Predictive coding network with single layer decoder networks"""

    def __init__(self, obs_size=784, prior_size=64, activation=F.relu):
        super().__init__(obs_size=784, prior_size=64, activation=F.relu)

        # generative networks
        self.dec3 = BaseDecoder_single(z_dim=prior_size, x_dim=64, h_dim=0)
        self.dec2 = BaseDecoder_single(z_dim=64, x_dim=128, h_dim=0)
        self.dec1 = BaseDecoder_single(z_dim=128, x_dim=obs_size, h_dim=0)

        self.decoders = [self.dec1, self.dec2, self.dec3]

        # prediction error precision
        self.error_precision = [None for _ in range(len(self.decoders) + 1)]

        # logging
        self.log_errors, self.log_errors_test = [], []
        self.log_error_posterior, self.log_error_posterior_test = [], []


# Experiment
DS, DS_name = MNIST, "MNIST"
BATCH_SIZE = 128
updates = 1000
test_interval = 50
hidden_dim = 256
latent_dim = 64
inference_lr = 0.9
inference_steps = 20
obs_size = 28 * 28
RUNS = 1

# PredProp optimizer parameters
damp_out = 0.1
damp_in = 0.0001
damp_err_inf = 0.9
damp_act_inf = 0.9

# load dataset
dataset = DS(batch_size=BATCH_SIZE, logit_transform=False)
train_data = dataset.train_loader
test_data = dataset.test_loader

# train models
models, run_names = [], []
for WEIGHTS_NGD, OPT_NAME in zip([True, False], ["PC-PredProp", "PC-Adam"]):
    betas_1 = [0.0, 0.1, 0.9] if not WEIGHTS_NGD else [""]

    for beta_1 in betas_1:
        if not WEIGHTS_NGD:
            betas_2 = [0.0] if beta_1 == 0.0 else [0.999]
        else:
            betas_2 = [""]

        for beta_2 in betas_2:
            if not WEIGHTS_NGD:
                LR_weights_list = [0.001] if beta_2 > 0.0 else [0.1, 0.01]
            else:
                LR_weights_list = [0.5]

            for LRW, LR_weights in enumerate(LR_weights_list):
                inferences_NGD = [True, False] if WEIGHTS_NGD else [False]
                for inference_NGD in inferences_NGD:

                    for run in range(RUNS):
                        if OPT_NAME == "PC-PredProp":
                            run_names.append(
                                f"{OPT_NAME} ({LR_weights}, {damp_in}, {inference_NGD})"
                            )
                        elif beta_2 == 0.0:
                            run_names.append(
                                f"PC-SGD ({LR_weights}, {inference_NGD}, {beta_1})"
                            )
                        else:
                            run_names.append(
                                f"{OPT_NAME} ({LR_weights}, {inference_NGD}, {beta_1}, {beta_2})"
                            )
                        print(DS_name, run_names[-1])

                        vae = PC_single(obs_size=obs_size, prior_size=latent_dim).to(
                            device
                        )
                        vae = train(vae, train_data, test_data, dataset, updates)

                        models.append(vae)

# plot train and test errors
plot_training(models, run_names, DS_name)
