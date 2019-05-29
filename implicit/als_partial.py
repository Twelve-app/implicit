""" Implicit Alternating Least Squares """
import logging
import time

import implicit.cuda
import numpy as np
import tqdm

from .recommender_base import MatrixFactorizationBase
from collections import namedtuple

log = logging.getLogger("implicit")


MatrixGenerator = namedtuple("MatrixGenerator", ["user_items_generator", "item_users_generator"])


class PartialAlternatingLeastSquares(MatrixFactorizationBase):
    """ Alternating Least Squares

    A Recommendation Model based off the algorithms described in the paper 'Collaborative
    Filtering for Implicit Feedback Datasets' with performance optimizations described in
    'Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative
    Filtering.'

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    regularization : float, optional
        The regularization factor to use
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    iterations : int, optional
        The number of ALS iterations to use when fitting data
    calculate_training_loss : bool, optional
        Whether to log out the training loss at each iteration

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    """

    def __init__(self, item_users_shape, factors=128, regularization=0.01, dtype=np.float32,
                 iterations=15, calculate_training_loss=False):
        super(PartialAlternatingLeastSquares, self).__init__()

        # currently there are some issues when training on the GPU when some of the warps
        # don't have full factors. Round up to be warp aligned.
        # TODO: figure out where the issue is (best guess is in the
        # the 'dot' function in 'implicit/cuda/utils/cuh)
        if factors % 32:
            padding = 32 - factors % 32
            log.warning("GPU training requires factor size to be a multiple of 32."
                        " Increasing factors from %i to %i.", factors, factors + padding)
            factors += padding

        # parameters on how to factorize
        self.item_users_shape = item_users_shape
        self.factors = factors
        self.regularization = regularization

        # options on how to fit the model
        self.dtype = dtype
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.fit_callback = None
        self.cg_steps = 3

        # cache for item factors squared
        self._YtY = None
        self._init_fit()

    def _init_fit(self):
        items, users = self.item_users_shape

        self.user_factors = np.random.rand(users, self.factors).astype(self.dtype) * 0.01
        self.item_factors = np.random.rand(items, self.factors).astype(self.dtype) * 0.01

        # 2.5 GB x 2 (3M users * ~200 features * 4 bytes * [users and items])
        self.gpu_user_factors = implicit.cuda.CuDenseMatrix(self.user_factors.astype(np.float32))
        self.gpu_item_factors = implicit.cuda.CuDenseMatrix(self.item_factors.astype(np.float32))

        self.solver = implicit.cuda.CuPartialLeastSquaresSolver(self.factors)

    def fit_generators(self, matrix_generator, show_progress=True):
        """ Factorizes the item_users matrix.

        After calling this method, the members 'user_factors' and 'item_factors' will be
        initialized with a latent factor model of the input data.

        The item_users matrix does double duty here. It defines which items are liked by which
        users (P_iu in the original paper), as well as how much confidence we have that the user
        liked the item (C_iu).

        The negative items are implicitly defined: This code assumes that non-zero items in the
        item_users matrix means that the user liked the item. The negatives are left unset in this
        sparse matrix: the library will assume that means Piu = 0 and Ciu = 1 for all these items.

        Parameters
        ----------
        matrix_generator: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the item, the columns are the users that liked that item,
            and the value is the confidence that the user liked the item.
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        """
        X = self.gpu_user_factors
        Y = self.gpu_item_factors

        log.debug("Running %i ALS iterations", self.iterations)
        with tqdm.tqdm(total=self.iterations, disable=not show_progress) as progress:
            for iteration in range(self.iterations):
                iteration_data = matrix_generator()
                s = time.time()
                self.solver.least_squares_init(Y)
                for start_user, size, user_items in iteration_data.user_items():
                    Cui = implicit.cuda.CuCSRMatrix(user_items)
                    self.solver.least_squares(start_user, size, Cui, X, Y, self.regularization, self.cg_steps)
                progress.update(.5)

                self.solver.least_squares_init(X)
                for start_item, size, item_users in iteration_data.item_users():
                    Ciu = implicit.cuda.CuCSRMatrix(item_users)
                    self.solver.least_squares(start_item, size, Ciu, Y, X, self.regularization, self.cg_steps)
                progress.update(.5)

                if self.fit_callback:
                    self.fit_callback(iteration, time.time() - s)

                if self.calculate_training_loss:
                    loss = self.solver.calculate_loss(Cui, X, Y, self.regularization)
                    progress.set_postfix({"loss": loss})

        if self.calculate_training_loss:
            log.info("Final training loss %.4f", loss)

        X.to_host(self.user_factors)
        Y.to_host(self.item_factors)
