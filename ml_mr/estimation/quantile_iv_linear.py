"""
Implementation of an linear IV method
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from scipy import stats
import pickle as pk
import logging

from ..utils.data import IVDataset, FullBatchDataLoader
from ..utils import _cat
from .core import MREstimatorWithUncertainty
import uuid
import glob
from tqdm import tqdm

from quantile_iv import QuantileIVEstimator


class QuantileIVLinearEstimator(MREstimatorWithUncertainty):
    """
    QuantileIV estimator that uses the linear inference strategy, which
    consists of performing IV linear regression manually in the new
    representation of the exposure learned by the outcome neural network.
    """

    def __init__(
        self,
        estimator_path: str,
        id: str
    ):
        subdirs = ["parents", "pcas", "betas", "betas_variance"]

        subdirs_with_id = [
            os.path.join(subdir, id) for subdir in subdirs
        ]

        parent_estimator_path = os.path.join(estimator_path, subdirs_with_id[0])
        parent_estimator = os.readlink(parent_estimator_path)
        self.estimator = QuantileIVEstimator.from_results(parent_estimator)

        parent_pca = os.path.join(estimator_path, subdirs_with_id[1])
        self.pca = pk.load(open(parent_pca, "rb"))

        parent_betas = os.path.join(estimator_path, subdirs_with_id[2])
        self.betas = torch.load(parent_betas)

        parent_betas_variance = os.path.join(estimator_path, subdirs_with_id[3])
        self.betas_variance = torch.load(parent_betas_variance)

        self.covars = self.estimator.covars

        self.h = self.__iv_reg

    @classmethod
    def create_instances(self, estimator_path: str):
        QIVLinList = []
        if not os.path.isdir(estimator_path):
            raise ValueError("Path provided in not a directory")
        subdirs = ["parents", "pcas", "betas", "betas_variance"]

        path = os.path.join(estimator_path, subdirs[2])
        ids = [
            os.path.basename(s) for s in glob.glob(f"{path}/*")
            ]
        if len(ids) == 0:
            raise ValueError("Directory provided is empty")

        for id in ids:
            QIVLinList.append(QuantileIVLinearEstimator(estimator_path, id))
        return QIVLinList

    @classmethod
    def fit_and_save(
        cls,
        dataset: IVDataset,
        qiv_estimator_path: str,
        output_dir: str
    ):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        parents_folder = os.path.join(output_dir, "parents")
        betas_folder = os.path.join(output_dir, "betas")
        betas_variance_folder = os.path.join(output_dir, "betas_variance")
        pca_folder = os.path.join(output_dir, "pcas")

        os.mkdir(parents_folder)
        os.mkdir(betas_folder)
        os.mkdir(betas_variance_folder)
        os.mkdir(pca_folder)

        QIVLinList = []
        for e in tqdm(glob.glob(os.path.join(qiv_estimator_path, '*'))):
            if not os.path.isdir(e):
                logging.warn(f"{e} is not a directory. Skipping.")
                continue

            estimator = QuantileIVEstimator.from_results(
                e
            )

            if estimator is None:
                logging.warn(
                    f"""{e} is None.
                    Estimator does not contain meta.json. Skipping.""")
                continue

            id = cls.__generate_id()
            symlink_file = os.path.join(parents_folder, id)

            while os.path.exists(symlink_file):
                id = cls.__generate_id()
                symlink_file = os.path.join(parents_folder, id)

            os.symlink(os.path.abspath(e), symlink_file)

            pca = PCA(n_components=0.999, random_state=42)
            dl = FullBatchDataLoader(dataset)
            X, Y, ivs, covars = next(iter(dl))

            eta_bar = cls.__construct_eta_bar(estimator, covars, ivs)
            H = cls.__construct_H(estimator, X, covars)

            print("ETABAR CALC.")

            pca = cls.__fit_PCA(pca, eta_bar)

            eta_bar_pca = cls.__apply_PCA(eta_bar, pca)

            H_pca = cls.__apply_PCA(H, pca)

            print("PCA FIT AND APPLIED.")

            eta_bar_pca = cls.__add_intercept_col(eta_bar_pca)
            # eta_bar_pca = cls.__add_intercept_col(eta_bar)
            H_pca = cls.__add_intercept_col(H_pca)
            # H_pca = cls.__add_intercept_col(H)

            betas = cls.__compute_IV_betas(
                H_pca,
                eta_bar_pca,
                Y
            )

            print("BETAS CALC.")

            betas_variance = cls.__compute_IV_betas_variance(
                H_pca,
                eta_bar_pca,
                Y,
                betas
            )

            print("BETAS VARIANCE CALC.")

            output_dir_pca = os.path.join(pca_folder, id)
            output_dir_qivlinear_betas = os.path.join(betas_folder, id)
            output_dir_qivlinear_betas_variance = os.path.join(
                betas_variance_folder, id
            )

            print("SAVING FILES...")

            with open(output_dir_pca, "wb") as f:
                pk.dump(pca, f)

            torch.save(betas, output_dir_qivlinear_betas)
            torch.save(betas_variance, output_dir_qivlinear_betas_variance)

            torch.save(eta_bar, f"raw/complex/eta_bar/{id}")
            torch.save(H, f"raw/complex/H/{id}")

            QIVLinList.append(cls(output_dir, id))

            print("FILES SAVED")

        return QIVLinList

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.05
    ):
        ys, se_ys = self.h(x, covars)
        lower_ci = ys + stats.norm.ppf(alpha/2) * se_ys
        upper_ci = ys + stats.norm.ppf(1-alpha/2) * se_ys

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
        y_hats[:, :, 0] = y_hat + stats.norm.ppf(alpha/2) * avg_h_sd
        y_hats[:, :, 2] = y_hat + stats.norm.ppf(1 - alpha/2) * avg_h_sd

        return y_hats

    @staticmethod
    def __low_mem_avg_repr(
        estimator: QuantileIVEstimator,
        x: torch.Tensor,
        covars: torch.Tensor
    ) -> torch.Tensor:
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
        X: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ):
        if covars is None:
            print("No covars. No need to compute avg_h_variance")
            return None
        avg_eta = self.__low_mem_avg_repr(self.estimator, X, covars)
        avg_eta_pca = self.__apply_PCA(avg_eta, self.pca)
        avg_eta_pca = self.__add_intercept_col(avg_eta_pca)

        result = avg_eta_pca @ self.betas_variance @ avg_eta_pca.T

        return result

    def compute_avg_cate_variance(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        covars: torch.Tensor
    ):
        avg_eta_0 = self.__low_mem_avg_repr(self.estimator, x0, covars)
        avg_eta_1 = self.__low_mem_avg_repr(self.estimator, x1, covars)

        avg_eta_pca_0 = self.__apply_PCA(avg_eta_0, self.pca)
        avg_eta_pca_1 = self.__apply_PCA(avg_eta_1, self.pca)

        avg_eta_pca_0 = self.__add_intercept_col(avg_eta_pca_0)
        avg_eta_pca_1 = self.__add_intercept_col(avg_eta_pca_1)

        var0 = avg_eta_pca_0 @ self.betas_variance @ avg_eta_pca_0.T
        var1 = avg_eta_pca_1 @ self.betas_variance @ avg_eta_pca_1.T
        cross_term = (avg_eta_pca_1 @ self.betas_variance @ avg_eta_pca_0.T)
        + (avg_eta_pca_0 @ self.betas_variance @ avg_eta_pca_1.T)

        var = var0 + var1 - cross_term

        return var

    @staticmethod
    def __construct_eta_bar(
        estimator: QuantileIVEstimator,
        covars: torch.Tensor,
        ivs: torch.Tensor
    ):
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
        X: torch.Tensor,
        covars: torch.Tensor
    ):
        # Get the representation of the exposure x
        H = estimator.outcome_network.get_repr(
            _cat(X, covars)
        )

        return H

    @staticmethod
    def __compute_IV_betas(
        X: torch.Tensor,
        Z: torch.Tensor,
        Y: torch.Tensor
    ):
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
        # We compute the variance of the coefficients beta_IV_hat
        ZTX_inv = torch.linalg.lstsq(Z.T @ X, torch.eye(Z.size(1))).solution
        print("INV CALC.")
        xby = ((X @ betas - Y) ** 2).numpy().flatten()
        var_beta_IV_hat = (
            ZTX_inv
            @ Z.T
            * xby
            @ Z
            @ ZTX_inv
        )
        print("VAR CALC.")

        return var_beta_IV_hat

    @staticmethod
    def __fit_PCA(pca: PCA, to_fit: torch.Tensor):
        if to_fit.requires_grad:
            to_fit = to_fit.detach()
        return pca.fit(to_fit.numpy())

    @staticmethod
    def __apply_PCA(x: torch.Tensor, pca: PCA):
        x_pca = torch.from_numpy(
            pca.transform(x.detach().numpy())
        )
        return x_pca

    @staticmethod
    def __add_intercept_col(x):
        return torch.hstack((torch.ones((x.size(0), 1)), x))

    def compute_eta(self, x, covars):
        eta = self.estimator.outcome_network.get_repr(
            _cat(x, covars)
        )
        return eta

    def compute_eta_pca(self, x, covars):
        eta = self.estimator.outcome_network.get_repr(
            _cat(x, covars)
        )
        eta_pca = self.__apply_PCA(eta, self.pca)

        return eta_pca

    # Estimated h_hat function
    def __iv_reg(self, x, covars):
        betas = self.betas
        betas_var = self.betas_variance

        pca = self.pca

        eta = self.estimator.outcome_network.get_repr(
            _cat(x, covars)
        )

        eta_pca = self.__apply_PCA(eta, pca)
        eta_pca = self.__add_intercept_col(eta_pca)
        # eta_pca = self.__add_intercept_col(eta)

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

    # TODO: add alpha as parameter to ate and cate functions
    def ate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
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
        y1, _ = self.h(x1, covars)
        y0, _ = self.h(x0, covars)

        cate = y1 - y0

        var = torch.diag(
            self.compute_avg_cate_variance(x0, x1, covars)
        ).reshape(-1, 1)

        # add 2eta(covars,x1)V(u_hat)eta(covars,x0)
        lower_ci = cate + stats.norm.ppf(0.05/2) * torch.sqrt(var)
        upper_ci = cate + stats.norm.ppf(1 - 0.05/2) * torch.sqrt(var)

        return torch.hstack([lower_ci, cate, upper_ci]).view(
            -1, 1, 3
        )

    @classmethod
    def __generate_id(cls):
        id = uuid.uuid4()
        comp_id = str(id).split("-")
        return comp_id[1]


class EnsembledQuantileIVLinearEstimator(MREstimatorWithUncertainty):
    def __init__(
        self,
        estimators_path: str
    ):
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
        Vw: torch.Tensor = torch.zeros_like(x)
        for estimator in self.estimators:
            #Vw += torch.diag(estimator.compute_avg_h_variance(x, covars)).reshape(-1, 1)
            _, sd = estimator.h(x, covars)
            Vw += sd**2
        Vw /= self.m
        return Vw

    def compute_Vb(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
    ):
        ys_ensemble: torch.Tensor = torch.zeros_like(x)
        ys_list = []
        Vb: torch.Tensor = torch.zeros_like(x)

        for estimator in self.estimators:
            #ys = estimator.avg_iv_reg_function(x, covars)[:, :, 1]
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

        return ys, Vt

    def __iv_reg(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ):
        ys, Vt = self.__compute_total_variance_and_ys(x, covars)
        return ys, torch.sqrt(Vt)

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.05
    ):
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
        y1, vt1 = self.ens_h(x1, covars)
        y0, vt2 = self.ens_h(x0, covars)

        cate = y1 - y0

        var: torch.Tensor = torch.zeros_like(x0)
        cate_ = 0
        for estimator in self.estimators:
            var += torch.diag(
                estimator.compute_avg_cate_variance(x0, x1, covars)
            ).reshape(-1, 1)
            cate_ += estimator.cate(x0, x1, covars)[:, :, 1]

        var /= self.m

        var = var + ((1/self.m)*cate_**2) - ((1/self.m**2)*cate_**2)

        lower_ci = cate + stats.norm.ppf(0.05/2) * torch.sqrt(var)
        upper_ci = cate + stats.norm.ppf(1 - 0.05/2) * torch.sqrt(var)

        return torch.hstack([lower_ci, cate, upper_ci]).view(
            -1, 1, 3
        )