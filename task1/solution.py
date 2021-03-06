import os
import typing

from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib import cm

import tqdm
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy



# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0

class ApproximateGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(ApproximateGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        kernel = gpytorch.kernels.RBFKernel(length_scale=0.01)
        # kernel = gpytorch.kernels.RBFKernel(length_scale=1) + gpytorch.kernels.WhiteKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGP():
    def __init__(self):
        self.model = None
        self.likelihood = None

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = torch.from_numpy(x).float()
        self.model.eval()
        self.likelihood.eval()
        means = torch.tensor([0.])
        with torch.no_grad():
            preds = self.model(x)
        gp_mean, gp_std = preds.mean, preds.stddev
        gp_mean = gp_mean.cpu().detach().numpy()
        gp_std = gp_std.cpu().detach().numpy()
        predictions = gp_mean
        return predictions, gp_mean, gp_std

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        train_x = torch.from_numpy(train_x).contiguous().float()
        train_y = torch.from_numpy(train_y).contiguous().float()
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        inducing_points = train_x[:2000, :]
        self.model = ApproximateGPModel(inducing_points=inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # train the model
        num_epochs = 10

        self.model.train()
        self.likelihood.train()

        # two optimizers
        variational_ngd_optimizer = gpytorch.optim.NGD(self.model.variational_parameters(),
                                                       num_data=train_y.size(0), lr=1)
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': self.model.hyperparameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.1)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))

        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        losses = []
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()
                output = self.model(x_batch)
                # print("test MAE: ", torch.mean(torch.abs(output.mean - y_batch)))
                loss = -mll(output, y_batch)
                losses.append(loss.item())
                test_mae = torch.mean(torch.abs(output.mean - y_batch))
                minibatch_iter.set_postfix(loss=loss.item(),
                                           bar = "test_mae: {}".format(test_mae))
                # minibatch_iter.set_postfix()
                loss.backward()
                variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()

class NaiveGP_gpy():
    def __init__(self):
        rbf = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
        self.gpr = GPy.models.GPRegression(rbf)
        # # Fix the noise variance to known value
        # gpr.Gaussian_noise.variance = noise ** 2
        # gpr.Gaussian_noise.variance.fix()

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Get into evaluation (predictive posterior) mode
        x = torch.from_numpy(x).float()
        self.model.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.model(x)
        gp_mean, gp_std = preds.mean, preds.stddev
        gp_mean = gp_mean.cpu().detach().numpy()
        gp_std = gp_std.cpu().detach().numpy()
        predictions = gp_mean
        return predictions, gp_mean, gp_std
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #     observed_pred = self.likelihood(model(x))

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        # Run optimization
        gpr.optimize(train_x, train_y);


class NaiveGP():
    def __init__(self):
        kernel = RBF() + WhiteKernel() + Matern()
        self.gpr = GaussianProcessRegressor(kernel=kernel, random_state=0,
                                            normalize_y=True)

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gp_mean, gp_std = self.gpr.predict(x, return_std=True)
        predictions = gp_mean
        return predictions, gp_mean, gp_std

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        perm = np.random.permutation(train_x.shape[0])
        train_x = train_x[perm]
        train_y = train_y[perm]

        N = 5000
        if train_x.shape[0] > N:
            train_x = train_x[:N, :]
            train_y = train_y[:N]
        self.gpr.fit(train_x, train_y)
        print("test MAE: ", np.mean(np.abs(self.gpr.predict(train_x) - train_y)))
        print("score of gpr (RBF): ", self.gpr.score(train_x, train_y))
        print("LLD of gpr: ", self.gpr.log_marginal_likelihood())
        print("parameters of gpr: ", self.gpr.get_params())


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        # TODO: Add custom initialization for your model here if necessary
        # self.model = VariationalGP()
        self.model = NaiveGP()
        # self.model = NaiveGP_gpy()
        print("self.model: ", self.model)


    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        return  self.model.predict(x)
        # # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        # # TODO: Use the GP posterior to form your predictions here

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        self.model.fit_model(train_x, train_y)
        # TODO: Fit your model here


def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    print(train_x.shape)
    print(train_y.shape)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fit_model(train_x, train_y)

    # Predict on the test features
    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
