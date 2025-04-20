import os
from typing import Optional

import torch
from sklearn.decomposition import PCA
from scipy import stats
import pickle as pk
import numpy as np

from ..utils.data import IVDataset, FullBatchDataLoader
from ..utils import _cat
from .core import MREstimatorWithUncertainty
import glob
from tqdm import tqdm

import json

from ml_mr.estimation.quantile_iv import QuantileIVEstimator


class QuantileIVLinearEstimator(MREstimatorWithUncertainty):
    """
    A class that represents a QuantileIV linear estimator.

    A QuantileIV linear estimator is a QuantileIV estimator that uses the
    linear inference strategy, which consists of performing an IV linear
    regression manually in the new representation of the exposure learned by
    the outcome neural network.

    To initialize a QuantileIVLinearEstimator, the user first needs to
    run the fit_and_save class method using a path to a folder cotanining a
    QuantileIV estimator or multiple QuantileIV estimators.
    """
    def __init__(
        self,
        estimator_path: str
    ):
        """
        Initializes a QuantileIVLinearEstimator object.

        Parameters
        ----------
        estimator_path: str
            A path to a QuantileIVLinearEstimator created with the fit_and_save
            class method.

        Attributes
        ----------
        self.estimator: QuantileIVEstimator
            The parent QuantileIVEstimator of this QuantileIVLinearEstimator
            instance.
        self.pca: sklearn.decomposition.PCA
            PCA fitted to this instance of QuantileIVLinearEstimtor.
        self.betas: torch.Tensor
            Computed betas of this instance of QuantileIVLinearEstimator.
        self.betas_variance: torch.Tensor
            Computed variances of the betas of this instance of
            QuantileIVLinearEstimator
        self.covars: torch.Tensor
            covariables of the parent QuantileIVEstimator.
        self.h: Callable
            Method to compute predicted outcomes from the estimated h function.
        """
        # We load the parent estimator
        parent_estimator_path = os.path.join(estimator_path, "parent")
        parent_estimator = os.readlink(parent_estimator_path)
        self.estimator = QuantileIVEstimator.from_results(parent_estimator)

        # We load the pca
        pca = os.path.join(estimator_path, "pca")
        self.pca = pk.load(open(pca, "rb"))

        # We load the betas
        betas = os.path.join(estimator_path, "betas")
        self.betas = torch.load(betas)

        # We load the betas variance
        betas_variance = os.path.join(estimator_path, "betas_variance")
        self.betas_variance = torch.load(betas_variance)

        # We save the covariables of the parent estimator
        self.covars = self.estimator.covars

        # We save the h function
        self.h = self.__iv_reg

    @classmethod
    def create_instances(self, estimator_path: str):
        """
        Creates one or multiple instances of QuantileIVLinearEstimator class.

        Parameters
        ----------
        estimator_path: str
            A path to a or multiple QuantileIVEstimator estimators.

        Returns
        -------
        List
            QuantileIVLinearEstimator objects in a list format.

        Raises
        ------
        ValueError
            When the path provided is not a directory.
        """
        if not os.path.isdir(estimator_path):
            raise ValueError("Path provided is not a directory")

        QIVLinList = []
        for path in glob.glob(os.path.join(estimator_path, "*")):
            QIVLinList.append(QuantileIVLinearEstimator(path))
        return QIVLinList

    @classmethod
    def fit_and_save(
        cls,
        dataset: IVDataset,
        qiv_estimator_path: str,
        output_dir: str
    ):
        """
        Performs the linear inference and saves the following information:
            -PCA object information
            -Betas calculated from linear inference strategy
            -Variances of the betas
            -Parent estimator (QuantileIVEstimator) used
        The method then calls create_instances(self, estimator_path: str) to
        create the fitted QuantileIVLinearEstimator estimators.

        Parameters
        ----------
        dataset: IVDataset
            The dataset used for training the parent QuantileIVEstimator.
        qiv_estimator_path: str
            Path to the parent QuantileIVEstimator.
        output_dir: str
            Path to the output directory where the files will be saved.

        Returns
        -------
        List
            A list containing all QuantileIVLinearEstimator created. The list
            is of size 1 or of size equal to the number of subdirectories
            containing a meta.json file.

        Raises
        ------
        ValueError
            When the path qiv_estimator_path provided contains no
            QuantileIVEstimator.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        meta_path = os.path.join(qiv_estimator_path, "meta.json")
        is_meta = os.path.isfile(meta_path)

        estimators = {}
        skipped_log = {}
        skipped_fits_list = []
        fit_skipped = 0
        if is_meta:
            estimators[os.path.abspath(qiv_estimator_path)] = (
                QuantileIVEstimator.from_results(qiv_estimator_path)
            )
        else:
            for e in glob.glob(os.path.join(qiv_estimator_path, "*")):
                estimator = QuantileIVEstimator.from_results(e)
                if estimator is None:
                    print(f"SKIPPING {os.path.abspath(e)}. meta.json was not \
                        found.")
                    skipped_fits_list.append(os.path.abspath(e))
                    fit_skipped += 1
                    continue
                estimators[os.path.abspath(e)] = estimator
                

        if len(estimators) == 0:
            raise ValueError("No QuantileIVEstimator were found in the\
                             provided path")
            return None

        QIVLs = []
        for abspath in tqdm(estimators):
            last_part = os.path.basename(abspath)
            output_dir_estimator = os.path.join(output_dir,
                                                f"linear-{last_part}")
            os.makedirs(output_dir_estimator)

            estimator = estimators[abspath]

            pca = PCA(n_components=0.9, random_state=42)
            dl = FullBatchDataLoader(dataset)
            X, Y, ivs, covars = next(iter(dl))

            eta_bar = cls.__construct_eta_bar(estimator, covars, ivs)
            H = cls.__construct_H(estimator, X, covars)

            pca = cls.__fit_PCA(pca, eta_bar)
            eta_bar_pca = cls.__apply_PCA(eta_bar, pca)
            H_pca = cls.__apply_PCA(H, pca)

            eta_bar_pca = cls.__add_intercept_col(eta_bar_pca)
            H_pca = cls.__add_intercept_col(H_pca)

            betas = cls.__compute_IV_betas(
                H_pca,
                eta_bar_pca,
                Y
            )

            betas_variance = cls.__compute_IV_betas_variance(
                H_pca,
                eta_bar_pca,
                Y,
                betas
            )

            pca_file_path = os.path.join(output_dir_estimator, "pca")
            betas_file_path = os.path.join(output_dir_estimator, "betas")
            betas_variance_file_path = os.path.join(output_dir_estimator,
                                                    "betas_variance")

            symlink_file = os.path.join(output_dir_estimator, "parent")
            os.symlink(abspath, symlink_file)

            with open(pca_file_path, "wb") as f:
                pk.dump(pca, f)

            torch.save(betas, betas_file_path)
            torch.save(betas_variance, betas_variance_file_path)

            QIVLs.append(cls(output_dir_estimator))
        
        skipped_log["Skipped fits"] = {
            "Skipped": skipped_fits_list,
            "Reason": "Missing meta.json file."
        }
        skipped_log["Fit skipped"] = fit_skipped
            
        with open(os.path.join(output_dir, "skipped_log.json"), 'w') as f:
            json.dump(skipped_log, f)

        return QIVLs

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.05
    ):
        """
        Applies IV regression and calculates the confidence intervals of the
        predicted outcomes.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor, default=None, of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.
        alpha: float, default=0.05
            The significance level of the confidence interval, representing
            the probability of rejecting the null hypothesis when it is true.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1, 3)
            Lower bound of the predicted outcome in the first column,
            predicted outcome in the second column and upper bound of the
            predicted outcome in the third column.
        """
        ys, se_ys = self.h(x, covars)
        lower_ci = ys - 1.96 * se_ys
        upper_ci = ys + 1.96 * se_ys

        return torch.hstack([lower_ci, ys, upper_ci]).view(
            -1, 1, 3
        )

    def avg_iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        low_memory: bool = True,
        alpha: float = 0.05
    ) -> torch.Tensor:
        """
        Applies IV regression in the case when we want to average over the
        covariables and calculates the confidence intervals of the
        predicted outcomes.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor, default=None, of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.
        low_memory: bool, default=True
            A boolean value indicating the use or not of a low memory approach
            to averaging over the covariables.
        alpha: float, default=0.05
            The significance level of the confidence interval, representing
            the probability of rejecting the null hypothesis when it is true.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1, 3)
            Lower bound of the predicted outcome in the first column,
            predicted outcome in the second column and upper bound of the
            predicted outcome in the third column.
        """
        if covars is None:
            if self.estimator.covars is None:
                return self.iv_reg_function(x, None)
            else:
                covars = self.estimator.covars
        y_hats = super().avg_iv_reg_function(
            x=x, covars=covars, low_memory=low_memory)
        y_hat = y_hats[:, :, 1]
        avg_h_var = self.compute_avg_h_variance(
            x,
            covars
        )
        avg_h_var = torch.clip(
            torch.diag(avg_h_var), 1e-200
        ).squeeze().reshape(-1, 1)

        avg_h_sd = torch.sqrt(avg_h_var)
        y_hats[:, :, 0] = y_hat - 1.96 * avg_h_sd
        y_hats[:, :, 2] = y_hat + 1.96 * avg_h_sd

        return y_hats

    @staticmethod
    def __low_mem_avg_repr(
        estimator: QuantileIVEstimator,
        x: torch.Tensor,
        covars: torch.Tensor
    ) -> torch.Tensor:
        """
        @staticmethod
        Low memory approach to averaging over the covariables.

        Parameters
        ----------
        estimator: QuantileIVEstimator
            A QuantileIVEstimator object.
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1)
            Predicted outcome after averaging over the covariables.
        """
        avgs = []
        if covars is not None:
            num_covars = covars.shape[0]
        else:
            num_covars = 1
        for cur_x in x:
            cur_x_rep = cur_x.repeat(num_covars).reshape(num_covars, -1)
            cur_cf = torch.mean(estimator.outcome_network.get_repr(
                _cat(cur_x_rep, covars)
            ), dim=0, keepdim=True)
            avgs.append(cur_cf)
        return torch.vstack(avgs)

    def compute_avg_h_variance(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ):
        """
        Computes the variance of the estimated h function when averaging over
        the covariables. If covars is none, then computes the variance
        of the estimated h function.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor, default=None, of shape (n_samples, d_features)
            covariables, where d_features is the number of
            covariables.

        Returns
        -------
        torch.Tensor of shape (n_samples, n_samples)
            Variance of the estimated h function averaged over the observed
            confunders. The variances are found on the diagonal of the returned
            tensor.
        """
        avg_eta = self.__low_mem_avg_repr(self.estimator, x, covars)
        avg_eta_pca = self.__apply_PCA(avg_eta, self.pca)
        avg_eta_pca = self.__add_intercept_col(avg_eta_pca)

        result = avg_eta_pca @ self.betas_variance @ avg_eta_pca.T

        return result

    def compute_cate_variance(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ):
        """
        Computes the variance of the CATE.

        Parameters
        ----------
        x0: torch.Tensor of shape (n_samples, 1)
            Baseline exposure data.
        x1: torch.Tensor of shape (n_samples,1 )
            Post-treatment exposure data.
        covars: torch.Tensor of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.

        Returns
        -------
        torch.Tensorof shape (n_samples, n_samples)
            Variance of the CATE for each estimated outcome.
            The variances are found on the diagonal of the returned tensor.
        """
        eta0 = self.estimator.outcome_network.get_repr(
            _cat(x0, covars)
        )
        eta1 = self.estimator.outcome_network.get_repr(
            _cat(x1, covars)
        )

        eta_pca_0 = self.__apply_PCA(eta0, self.pca)
        eta_pca_1 = self.__apply_PCA(eta1, self.pca)

        eta_pca_0 = self.__add_intercept_col(eta_pca_0)
        eta_pca_1 = self.__add_intercept_col(eta_pca_1)

        var = (
            (eta_pca_1 - eta_pca_0)
            @ self.betas_variance
            @ (eta_pca_1 - eta_pca_0).T
        )
        
        # var = (
        #     eta_pca_1 @ self.betas_variance @ eta_pca_1.T
        #     + eta_pca_0 @ self.betas_variance @ eta_pca_0.T
        #     - eta_pca_1 @ self.betas_variance @ eta_pca_0.T
        #     - eta_pca_0 @ self.betas_variance @ eta_pca_1.T
        # )

        return var
    

    def compute_avg_cate_variance(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        covars: torch.Tensor
    ):
        """
        Computes the variance of the CATE when averaging over the observed
        confounders.

        Parameters
        ----------
        x0: torch.Tensor of shape (n_samples, 1)
            Baseline exposure data.
        x1: torch.Tensor of shape (n_samples,1 )
            Post-treatment exposure data.
        covars: torch.Tensor of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.

        Returns
        -------
        torch.Tensorof shape (n_samples, n_samples)
            Variance of the CATE for each estimated outcome when averaging over
            the covariables. The variances are found on the diagonal
            of the returned tensor.
        """
        avg_eta_0 = self.__low_mem_avg_repr(self.estimator, x0, covars)
        avg_eta_1 = self.__low_mem_avg_repr(self.estimator, x1, covars)

        avg_eta_pca_0 = self.__apply_PCA(avg_eta_0, self.pca)
        avg_eta_pca_1 = self.__apply_PCA(avg_eta_1, self.pca)

        avg_eta_pca_0 = self.__add_intercept_col(avg_eta_pca_0)
        avg_eta_pca_1 = self.__add_intercept_col(avg_eta_pca_1)

        var = (
            (avg_eta_pca_1 - avg_eta_pca_0)
            @ self.betas_variance
            @ (avg_eta_pca_1-avg_eta_pca_0).T
        )

        return var

    @staticmethod
    def __construct_eta_bar(
        estimator: QuantileIVEstimator,
        covars: torch.Tensor,
        ivs: torch.Tensor
    ):
        """
        Computes and constructs the eta_bar variable specified in the report.
        Eta_bar represents the averaged representation over the conditional
        quantiles of the exposure learned by the first stage neural network.
        For more details, see report.

        Parameters
        ----------
        estimator: QuantileIVEstimator
            QuantileIVEstimator object.
        covars: torch.Tensor of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.
        ivs: torch.Tensor of shape (n_samples, m_features)
            Instrumental variables data, where m_features is the number of
            instrumental variables.

        Returns
        -------
        torch.Tensor of shape (n_samples, outcome_network_last_dimension)
            The computed eta_bar, where outcome_network_last_dimension is the
            last dimension of the outcome network (default=32).
        """
        x_hats = estimator.exposure_network(_cat(ivs, covars))
        n_quantiles = x_hats.size(1)

        eta_bar = 0
        # Loop through all quantiles
        # Equivalent to the 1/K sum step in equation (6)
        for j in range(n_quantiles):
            current_representation = estimator.outcome_network.get_repr(
                _cat(x_hats[:, [j]], covars)
            ) / (n_quantiles)

            eta_bar += current_representation

        return eta_bar

    @staticmethod
    def __construct_H(
        estimator: QuantileIVEstimator,
        x: torch.Tensor,
        covars: torch.Tensor
    ):
        """
        Computes and construct the H variable. The H variable is a
        basically the eta variable applied to the exposure and observed
        confounders. For more information, see report..

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.

        Returns
        -------
        torch.Tensor of shape (n_samples, outcome_network_last_dimension)
            The computed H, where outcome_network_last_dimension is the
            dimension of the outcome network (default=32).
        """
        # Get the representation of the exposure x
        H = estimator.outcome_network.get_repr(
            _cat(x, covars)
        )

        return H

    @staticmethod
    def __compute_IV_betas(
        X: torch.Tensor,
        Z: torch.Tensor,
        Y: torch.Tensor
    ):
        """
        Computes the estimated betas of the IV regression using theoretical
        formula for IV regression.

        Parameters
        ----------
        X: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        Z: torch.Tensor of shape (n_samples, d_features)
            Instrumental variables data, where d_features is the number of
            instrumental variables.
        Y: torch.Tensor of shape (n_samples, 1)
            Outcome data.

        Returns
        -------
        torch.Tensor of shape (d_features, 1)
            The computed betas.
        """
        beta_IV_hat = torch.linalg.lstsq(
            Z.T @ X, torch.eye(Z.size(1))
            ).solution @ Z.T @ Y

        return beta_IV_hat

    @staticmethod
    def __compute_IV_betas_variance(
        X: torch.Tensor,
        Z: torch.Tensor,
        Y: torch.Tensor,
        betas: torch.Tensor
    ):
        """
        Computes the variances of the betas from the IV regression.

        Parameters
        ----------
        X: torch.Tensor of shape (n_samples, 1)
            Exposure data, where n_samples is the number of samples.
        Z: torch.Tensor of shape (n_samples, d_features)
            Instrumental variables data, where n_samples is the number of
            samples and d_features is the number of instrumental variables.
        Y: torch.Tensor of shape (n_samples, 1)
            Outcome data, where n_samples is the number of samples.

        Returns
        -------
        torch.Tensor of shape (d_features, d_features)
            The computed variances of the betas.
        """
        # We compute the variance of the coefficients beta_IV_hat
        ZTX_inv = torch.linalg.lstsq(Z.T @ X, torch.eye(Z.size(1))).solution
        xby = ((X @ betas - Y) ** 2).numpy().flatten()
        var_beta_IV_hat = (
            ZTX_inv
            @ Z.T
            * xby
            @ Z
            @ ZTX_inv
        )

        return var_beta_IV_hat

    @staticmethod
    def __fit_PCA(pca: PCA, to_fit: torch.Tensor):
        """
        Fits a PCA using the pca object provided to the tensor provided.

        Parameters
        ----------
        pca: sklearn.decomposition.PCA
            A sklearn.decomposition.PCA object.
        to_fit: torch.Tensor of shape (n_samples, m_features)
            The tensor on which the pca will be fitted.

        Returns
        -------
        sklearn.decomposition.PCA
            Instance of pca object itself.
        """
        if to_fit.requires_grad:
            to_fit = to_fit.detach()
        return pca.fit(to_fit.numpy())

    @staticmethod
    def __apply_PCA(x: torch.Tensor, pca: PCA):
        """
        Applies the sklearn.decomposition.PCA transform method from the pca
        object provided to the x tensor provided.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, d_features)
            Data where n_samples is the number of sampels and d_features is the
            number of features.
        pca: sklearn.decomposition.PCA
            sklearn.decomposition.PCA object already fitted.

        Returns
        -------
        torch.Tensor of shape (n_samples, p_features)
            Reduced x data number of features from d_features to p_features
            where p_features < d_features.

        Raises
        ------
        ValueError
            When pca object provided has not yet been fitted.
        """
        if not hasattr(pca, "components_"):
            raise ValueError("PCA needs to be fitted before being applied.")
        x_pca = torch.from_numpy(
            pca.transform(x.detach().numpy())
        )
        return x_pca

    @staticmethod
    def __add_intercept_col(x: torch.Tensor):
        """
        Adds a column of 1 to tensor x as the first column of x.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, d_features)
            Tensor object with n_samples lines and d_features columns.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1 + d_features)
            the tensor x with an added column of 1 as its first column.
        """
        return torch.hstack((torch.ones((x.size(0), 1)), x))

    def __iv_reg(self, x: torch.Tensor, covars: torch.Tensor):
        """
        Applies the h function formula as seen in the report. The h function
        formula is eta(x, covars)@betas. For more details, see report.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor of shape (n_samples, d_features)
            covariables data where d_features is the number of
            features.

        Returns
        -------
        Tuple(torch.Tensor, torch.Tensor)
        of shapes ((n_samples, 1), (n_samples, 1))
            Predicted outcomes and standard deviations of the predicted
            outcomes.
        """
        betas = self.betas
        betas_var = self.betas_variance

        pca = self.pca

        eta = self.estimator.outcome_network.get_repr(
            _cat(x, covars)
        )

        eta_pca = self.__apply_PCA(eta, pca)
        eta_pca = self.__add_intercept_col(eta_pca)

        # IV regression using the estimated beta_IV_hat
        h_hat = betas.T @ eta_pca.T
        se_h_hat = torch.sqrt(
            torch.clip(
                torch.diag(eta_pca @ betas_var @ eta_pca.T), 1e-200
            )
        )
        h_hat = h_hat.squeeze().reshape(-1, 1)
        se_h_hat = se_h_hat.squeeze().reshape(-1, 1)

        return h_hat, se_h_hat

    def ate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the ATE.

        NOTE: Variance of the ATE might not be calculated properly.

        Parameters
        ----------
        x0: torch.Tensor of shape (n_samples, 1)
            Baseline exposure data.
        x1: torch.Tensor of shape (n_samples,1 )
            Post-treatment exposure data.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1, 3)
            Lower bound of the predicted ATE in the first column,
            predicted ATE in the second column and upper bound of the
            predicted ATE in the third column.
        """
        y1_ci = self.avg_iv_reg_function(x1)
        y0_ci = self.avg_iv_reg_function(x0)

        var0 = self.compute_avg_h_variance(x0, self.estimator.covars)
        var1 = self.compute_avg_h_variance(x1, self.estimator.covars)

        y1 = y1_ci[:, :, 0]
        y0 = y0_ci[:, :, 0]

        ate = y1 - y0

        lower_ci = ate + stats.norm.ppf(0.05/2) * torch.sqrt(var0 + var1)
        upper_ci = ate + stats.norm.ppf(1 - 0.05/2) * torch.sqrt(var0 + var1)

        return torch.hstack([lower_ci, ate, upper_ci]).view(
            -1, 1, 3
        )

    def cate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        covars: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the CATE.

        NOTE: Variance of the CATE is not calculated properly.

        Parameters
        ----------
        x0: torch.Tensor of shape (n_samples, 1)
            Baseline exposure data.
        x1: torch.Tensor of shape (n_samples,1 )
            Post-treatment exposure data.
        covars: torch.Tensor of shape (n_samples, d_features)
            covariables, where d_features is the number of
            confounders.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1, 3)
            Lower bound of the predicted CATE in the first column,
            predicted CATE in the second column and upper bound of the
            predicted CATE in the third column.
        """
        y1, _ = self.h(x1, covars)
        y0, _ = self.h(x0, covars)

        cate = y1 - y0

        var = torch.diag(
            self.compute_cate_variance(x0, x1, covars)
        ).reshape(-1, 1)

        lower_ci = cate - 1.96 * torch.sqrt(var)
        upper_ci = cate + 1.96 * torch.sqrt(var)

        return torch.hstack([lower_ci, cate, upper_ci]).view(
            -1, 1, 3
        )


class GaussianMixtureEstimator(MREstimatorWithUncertainty):
    """
    Estimator used to calculate the gaussian mixture variance.
    """
    def __init__(
        self,
        estimators_path: str
    ):
        """
        Initializes a GaussianMixtureVarianceEstimator. This estimator
        ensembles multiple QuantileIVLinearEstimator estimators. To predict
        the outcomes, we compute the average predicted outcome over the
        QuantileIVLinearEstiamtor estimators. To get the variance of the
        predicted outcomes, we apply the gaussian mixture variance method
        presented in the report. See report for more details.

        Attributes
        ----------
        self.estimators: List[QuantileIVLinearEstimator]
            List of QuantileIVLinearEstimator objects used construct the
            GaussianMixtureVarianceEstimator
        self.m: int
            Number of QuantileIVLinearEstimator composing this instance.
        self.h: Callable
            Method to compute predicted outcomes from the estimated h function
            in the case of ensembling multiple QuantileIVLinearEstimator
            estimators.
        self.covars: torch.Tensor
            covariables of the parent QuantileIVEstimator.
        """
        self.estimators = QuantileIVLinearEstimator.create_instances(
            estimators_path
        )
        self.m = len(self.estimators)
        self.h = self.__iv_reg
        self.covars = self.estimators[0].covars

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.05
    ):
        """
        Applies IV regression and calculates the confidence intervals of the
        predicted outcomes.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor, default=None, of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.
        alpha: float, default=0.05
            The significance level of the confidence interval, representing
            the probability of rejecting the null hypothesis when it is true.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1, 3)
            Lower bound of the predicted outcome in the first column,
            predicted outcome in the second column and upper bound of the
            predicted outcome in the third column.
        """
        ys, sd_vt = self.h(x, covars)

        lower_ci = ys - 1.96 * sd_vt
        upper_ci = ys + 1.96 * sd_vt

        return torch.hstack([lower_ci, ys, upper_ci]).view(-1, 1, 3)

    def __iv_reg(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ):
        """
        Applies the h function formula in the case of ensembling multiple
        QuantileIVLinearEstimator estimators using the gaussian mixture
        variance strategy. See report for more details.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor of shape (n_samples, d_features)
            covariables data where d_features is the number of
            features.

        Returns
        -------
        Tuple(torch.Tensor, torch.Tensor)
        of shapes ((n_samples, 1), (n_samples, 1))
            Predicted outcomes and standard deviations of the predicted
            outcomes.
        """
        ys_bagged: torch.Tensor = torch.zeros_like(x)
        var_bagged: torch.Tensor = torch.zeros_like(x)
        yss = []
        for estimator in self.estimators:
            ys, sd = estimator.h(x, covars)
            ys_bagged += ys
            var_bagged += sd**2
            yss.append(ys)

        ys_bagged /= self.m
        np_yss = np.array(yss)
        var_bagged = var_bagged/self.m + sum(np_yss**2)/self.m - ys_bagged**2
        return ys_bagged, torch.sqrt(var_bagged)


class EnsembledQuantileIVLinearEstimator(MREstimatorWithUncertainty):
    """
    Ensembling of multiple QuantileIVLinearEstimator estimators using the
    Rubin's rule approach to calculating the variance of the predicted
    outcomes.
    """
    def __init__(
        self,
        estimators_path
    ):
        """
        Initializes a EnsembledQuantileIVLinearEstimator. This estimator
        ensembles multiple QuantileIVLinearEstimator estimators. To predict
        the outcomes, we compute the average predicted outcome over the
        QuantileIVLinearEstiamtor estimators. To get the variance of the
        predicted outcomes, we apply the Rubin's rule variance method
        presented in the report. See report for more details.

        Attributes
        ----------
        self.estimators: List[QuantileIVLinearEstimator]
            List of QuantileIVLinearEstimator objects used construct the
            GaussianMixtureVarianceEstimator
        self.m: int
            Number of QuantileIVLinearEstimator composing this instance.
        self.h: Callable
            Method to compute predicted outcomes from the estimated h function
            in the case of ensembling multiple QuantileIVLinearEstimator
            estimators.
        self.covars: torch.Tensor
            covariables of the parent QuantileIVEstimator.
        """
        self.estimators = QuantileIVLinearEstimator.create_instances(
            estimators_path)
        self.m = len(self.estimators)
        self.ens_h = self.__iv_reg

        self.covars = self.estimators[0].covars

    def compute_Vw(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ):
        """
        Computes the variance within from the Rubin's rule approach to
        calculating the variance of an ensembled model. The variance within
        refers to the average variance over all QuantileIVLinearEstimator
        estimators composing the ensembled model. See report for more
        information.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor, default=None, of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.

        Returns
        -------
        torch.Tensor os shape (n_samples, 1)
            Average variance over all QuantileIVLinearEstimator estimators
            composing the ensembled model.
        """
        Vw: torch.Tensor = torch.zeros_like(x)
        for estimator in self.estimators:
            _, sd = estimator.h(x, covars)
            Vw += sd**2
        Vw /= self.m
        return Vw

    def compute_Vb(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
    ):
        """
        Computes the variance between from the Rubin's rule approach to
        calculating the variance of an ensembled model. The variance between
        refers to the variance between each QuantileIVLinearEstimator
        estimators composing the ensembled model. See report for more details.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor, default=None, of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.

        Returns
        -------
        torch.Tensor os shape (n_samples, 1)
            Average variance between all QuantileIVLinearEstimator estimators
            composing the ensembled model.
        """
        ys_ensemble: torch.Tensor = torch.zeros_like(x)
        ys_list = []
        Vb: torch.Tensor = torch.zeros_like(x)

        for estimator in self.estimators:
            ys, _ = estimator.h(x, covars)
            ys_list.append(ys)
            ys_ensemble += ys

        ys_ensemble /= self.m

        for j in range(self.m):
            curr_Vb = (ys_list[j] - ys_ensemble) ** 2
            Vb += curr_Vb

        Vb /= (self.m - 1)
        return Vb

    def __compute_total_variance_and_ys(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        avg: Optional[bool] = False,
    ):
        """
        Computes the predicted outcomes of the ensembled model as well as the
        total variance using Rubin's rule approach to variance calculation. For
        more details on how the variance is calculated, see report.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor, default=None, of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.
        avg: bool, default=False
            Boolean value determining if we should average over the observed
            confounders or not.
        """
        ys_ensemble: torch.Tensor = torch.zeros_like(x)
        ys_list = []
        Vw: torch.Tensor = torch.zeros_like(x)
        Vb: torch.Tensor = torch.zeros_like(x)
        for estimator in self.estimators:
            if covars is None:
                if self.covars is not None:
                    covars = self.covars
            if avg:
                ys = estimator.avg_iv_reg_function(x, covars)[:, :, 1]
                var = torch.diag(estimator.compute_avg_h_variance(x, covars)).reshape(-1, 1)
            else:
                ys, sd = estimator.h(x, covars)
                var = sd**2

            ys_list.append(ys)
            Vw += var
            ys_ensemble += ys

        Vw /= self.m
        ys_ensemble /= self.m

        for j in range(self.m):
            curr_Vb = (ys_list[j] - ys_ensemble) ** 2
            Vb += curr_Vb

        Vb /= (self.m - 1)
        Vt = Vw + Vb + Vb/self.m

        return ys_ensemble, Vt

    def __iv_reg(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ):
        """
        Applies the h function formula in the case of ensembling multiple
        QuantileIVLinearEstimator estimators using the Rubin's rule strategy.
        See report for more details.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor of shape (n_samples, d_features)
            covariables data where d_features is the number of
            features.

        Returns
        -------
        Tuple(torch.Tensor, torch.Tensor)
        of shapes ((n_samples, 1), (n_samples, 1))
            Predicted outcomes and standard deviations of the predicted
            outcomes.
        """
        ys, Vt = self.__compute_total_variance_and_ys(x, covars)
        return ys, torch.sqrt(Vt)

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.05
    ):
        """
        Applies IV regression and calculates the confidence intervals of the
        predicted outcomes.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor, default=None, of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.
        alpha: float, default=0.05
            The significance level of the confidence interval, representing
            the probability of rejecting the null hypothesis when it is true.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1, 3)
            Lower bound of the predicted outcome in the first column,
            predicted outcome in the second column and upper bound of the
            predicted outcome in the third column.
        """
        ys_ensemble, Vt = self.__compute_total_variance_and_ys(x, covars)

        sd_vt = torch.sqrt(Vt)
        lower_ci = ys_ensemble - 1.96 * sd_vt
        upper_ci = ys_ensemble + 1.96 * sd_vt

        return torch.hstack([lower_ci, ys_ensemble, upper_ci]).view(-1, 1, 3)

    def avg_iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        low_memory: bool = True,
        alpha: float = 0.05
    ):
        """
        Applies IV regression in the case when we want to average over the
        covariables and calculates the confidence intervals of the
        predicted outcomes.

        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, 1)
            Exposure data.
        covars: torch.Tensor, default=None, of shape (n_samples, d_features)
            covariables data, where d_features is the number of
            covariables.
        low_memory: bool, default=True
            A boolean value indicating the use or not of a low memory approach
            to averaging over the covariables.
        alpha: float, default=0.05
            The significance level of the confidence interval, representing
            the probability of rejecting the null hypothesis when it is true.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1, 3)
            Lower bound of the predicted outcome in the first column,
            predicted outcome in the second column and upper bound of the
            predicted outcome in the third column.
        """
        ys_ensemble, Vt = self.__compute_total_variance_and_ys(x, covars, True)

        sd_vt = torch.sqrt(Vt)
        lower_ci = ys_ensemble + stats.norm.ppf(alpha/2) * sd_vt
        upper_ci = ys_ensemble + stats.norm.ppf(1-alpha/2) * sd_vt

        return torch.hstack([lower_ci, ys_ensemble, upper_ci]).view(-1, 1, 3)

    def ate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the ATE.

        NOTE: Variance of the ATE might not be calculated properly.

        Parameters
        ----------
        x0: torch.Tensor of shape (n_samples, 1)
            Baseline exposure data.
        x1: torch.Tensor of shape (n_samples,1 )
            Post-treatment exposure data.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1, 3)
            Lower bound of the predicted ATE in the first column,
            predicted ATE in the second column and upper bound of the
            predicted ATE in the third column.
        """
        y0, var0 = self.__compute_total_variance_and_ys(x0, self.covars, True)
        y1, var1 = self.__compute_total_variance_and_ys(x1, self.covars, True)

        ate = y1 - y0

        lower_ci = ate + stats.norm.ppf(0.05/2) * torch.sqrt(var0 + var1)
        upper_ci = ate + stats.norm.ppf(1 - 0.05/2) * torch.sqrt(var0 + var1)

        return torch.hstack([lower_ci, ate, upper_ci]).view(
            -1, 1, 3
        )

    def cate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        covars: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the CATE.

        NOTE: Variance of the CATE is not calculated properly.

        Parameters
        ----------
        x0: torch.Tensor of shape (n_samples, 1)
            Baseline exposure data.
        x1: torch.Tensor of shape (n_samples,1 )
            Post-treatment exposure data.
        covars: torch.Tensor of shape (n_samples, d_features)
            covariables, where d_features is the number of
            confounders.

        Returns
        -------
        torch.Tensor of shape (n_samples, 1, 3)
            Lower bound of the predicted CATE in the first column,
            predicted CATE in the second column and upper bound of the
            predicted CATE in the third column.
        """
        y1, vt1 = self.ens_h(x1, covars)
        y0, vt2 = self.ens_h(x0, covars)

        cate = y1 - y0

        var: torch.Tensor = torch.zeros_like(x0)
        avg_cate = 0
        cates = []
        for estimator in self.estimators:
            var += torch.diag(
                estimator.compute_cate_variance(x0, x1, covars)
            ).reshape(-1, 1)
            cate_ = estimator.cate(x0, x1, covars)[:, :, 1]
            cates.append(cate_)
            avg_cate += cate_
            
        avg_cate = avg_cate/self.m
            
        Vw = var/self.m
        
        Vb = 0
        for j in range(self.m):
            curr_Vb = (cates[j] - avg_cate) ** 2
            Vb += curr_Vb

        Vb /= (self.m - 1)
        
        Vt = Vw + Vb + Vb/self.m

        lower_ci = cate - 1.96 * torch.sqrt(Vt)
        upper_ci = cate + 1.96 * torch.sqrt(Vt)

        return torch.hstack([lower_ci, cate, upper_ci]).view(
            -1, 1, 3
        )