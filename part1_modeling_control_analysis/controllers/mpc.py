import numpy as np





def build_prediction_matrices(Ad, Bd, horizon):

    """

    Build stacked discrete-time prediction matrices



        X = F x_k + G U



    where



        X = [x_{k+1};

             x_{k+2};

             ...

             x_{k+N}]



        U = [u_k;

             u_{k+1};

             ...

             u_{k+N-1}]

    """



    Ad = np.asarray(Ad, dtype=float)

    Bd = np.asarray(Bd, dtype=float)



    nx = Ad.shape[0]

    nu = Bd.shape[1]

    N = int(horizon)



    if Ad.shape != (nx, nx):

        raise ValueError("Ad must be square.")



    if Bd.shape[0] != nx:

        raise ValueError("Ad and Bd dimensions are inconsistent.")



    if N <= 0:

        raise ValueError("horizon must be positive.")



    F = np.zeros(

        (N * nx, nx)

    )



    G = np.zeros(

        (N * nx, N * nu)

    )



    # Precompute powers:

    #

    # A^0, A^1, ..., A^N



    A_power = [

        np.eye(nx)

    ]



    for _ in range(N):

        A_power.append(

            A_power[-1] @ Ad

        )



    # --------------------------------------------------------

    # F matrix

    # --------------------------------------------------------



    for i in range(N):



        row = slice(

            i * nx,

            (i + 1) * nx,

        )



        F[row, :] = A_power[i + 1]



    # --------------------------------------------------------

    # G matrix

    #

    # x_{k+i+1}

    # =

    # A^(i+1) x_k

    # +

    # sum_j A^(i-j) B u_{k+j}

    # --------------------------------------------------------



    for i in range(N):



        row = slice(

            i * nx,

            (i + 1) * nx,

        )



        for j in range(i + 1):



            col = slice(

                j * nu,

                (j + 1) * nu,

            )



            G[row, col] = (

                A_power[i - j] @ Bd

            )



    return F, G



class UnconstrainedMPC:

    """

    Finite-horizon discrete linear MPC without constraints.



    Prediction:

        X = F x_k + G U



    Cost:

        J =

        (X - X_ref)^T Qbar (X - X_ref)

        + U^T Rbar U

    """



    def __init__(

        self,

        Ad,

        Bd,

        Q,

        R,

        horizon,

    ):

        self.Ad = np.asarray(

            Ad,

            dtype=float,

        )



        self.Bd = np.asarray(

            Bd,

            dtype=float,

        )



        self.Q = np.asarray(

            Q,

            dtype=float,

        )



        self.R = np.asarray(

            R,

            dtype=float,

        )



        self.N = int(horizon)



        self.nx = self.Ad.shape[0]

        self.nu = self.Bd.shape[1]



        self.F, self.G = (

            build_prediction_matrices(

                self.Ad,

                self.Bd,

                self.N,

            )

        )



        # Repeated stage costs

        self.Qbar = np.kron(

            np.eye(self.N),

            self.Q,

        )



        self.Rbar = np.kron(

            np.eye(self.N),

            self.R,

        )



        # Hessian-like matrix

        self.H = (

            self.G.T

            @ self.Qbar

            @ self.G

            + self.Rbar

        )





    def solve(

        self,

        x,

        X_ref,

    ):

        """

        Solve the unconstrained finite-horizon problem.



        Parameters

        ----------

        x : shape (nx,)

            Current state.



        X_ref : shape (N*nx,)

            Stacked reference states:

            [x_ref(k+1), ..., x_ref(k+N)]



        Returns

        -------

        U : shape (N*nu,)

            Optimal input sequence.

        """



        x = np.asarray(

            x,

            dtype=float,

        )



        X_ref = np.asarray(

            X_ref,

            dtype=float,

        )



        if x.shape != (self.nx,):

            raise ValueError(

                f"x must have shape ({self.nx},)"

            )



        if X_ref.shape != (

            self.N * self.nx,

        ):

            raise ValueError(

                "X_ref has incorrect shape."

            )



        prediction_error = (

            self.F @ x

            - X_ref

        )



        rhs = (

            -self.G.T

            @ self.Qbar

            @ prediction_error

        )



        U = np.linalg.solve(

            self.H,

            rhs,

        )



        return U





    def compute(

        self,

        x,

        X_ref,

    ):

        """

        Return only the first receding-horizon input.

        """



        U = self.solve(

            x,

            X_ref,

        )



        return U[:self.nu]



import scipy.sparse as sp

import osqp





class OSQPMPC:

    """

    Finite-horizon linear MPC solved as a QP with OSQP.



    Prediction:

        X = F x + G U



    Cost:

        J =

        (X-Xref)^T Qbar (X-Xref)

        + U^T Rbar U



    OSQP convention:

        min 0.5 U^T P U + q^T U



    Therefore:

        P = 2 (G^T Qbar G + Rbar)

        q = 2 G^T Qbar (F x - Xref)

    """



    def __init__(

        self,

        Ad,

        Bd,

        Q,

        R,

        horizon,

        input_lower=None,

        input_upper=None,

    ):

        self.Ad = np.asarray(Ad, dtype=float)

        self.Bd = np.asarray(Bd, dtype=float)

        self.Q = np.asarray(Q, dtype=float)

        self.R = np.asarray(R, dtype=float)



        self.N = int(horizon)



        self.nx = self.Ad.shape[0]

        self.nu = self.Bd.shape[1]



        self.F, self.G = build_prediction_matrices(

            self.Ad,

            self.Bd,

            self.N,

        )



        self.Qbar = np.kron(

            np.eye(self.N),

            self.Q,

        )



        self.Rbar = np.kron(

            np.eye(self.N),

            self.R,

        )



        self.H = (

            self.G.T

            @ self.Qbar

            @ self.G

            + self.Rbar

        )



        # OSQP uses:

        # 0.5 U^T P U + q^T U

        self.P = sp.csc_matrix(

            2.0 * self.H

        )



        nU = self.N * self.nu



        # ----------------------------------------------------

        # Input constraints

        # ----------------------------------------------------



        if input_lower is None:

            lower = np.full(

                nU,

                -np.inf,

            )

        else:

            input_lower = np.asarray(

                input_lower,

                dtype=float,

            )



            lower = np.tile(

                input_lower,

                self.N,

            )



        if input_upper is None:

            upper = np.full(

                nU,

                np.inf,

            )

        else:

            input_upper = np.asarray(

                input_upper,

                dtype=float,

            )



            upper = np.tile(

                input_upper,

                self.N,

            )



        self.lower = lower

        self.upper = upper



        # A = I gives direct bounds on U

        self.A_constraint = sp.eye(

            nU,

            format="csc",

        )



        self.solver = osqp.OSQP()



        # q is state/reference dependent,

        # initialize with zeros.

        self.solver.setup(

            P=self.P,

            q=np.zeros(nU),

            A=self.A_constraint,

            l=self.lower,

            u=self.upper,

            verbose=False,

            polishing=True,

            eps_abs=1e-10,

            eps_rel=1e-10,

        )





    def solve(

        self,

        x,

        X_ref,

    ):

        x = np.asarray(

            x,

            dtype=float,

        )



        X_ref = np.asarray(

            X_ref,

            dtype=float,

        )



        prediction_error = (

            self.F @ x

            - X_ref

        )



        q_qp = (

            2.0

            * self.G.T

            @ self.Qbar

            @ prediction_error

        )



        self.solver.update(

            q=q_qp

        )



        result = self.solver.solve()

        self.last_result = result



        if result.info.status not in (

            "solved",

            "solved inaccurate",

        ):

            raise RuntimeError(

                "OSQP failed: "

                + result.info.status

            )



        return result.x.copy()





    def compute(

        self,

        x,

        X_ref,

    ):

        U = self.solve(

            x,

            X_ref,

        )



        return U[:self.nu]



    def update_input_bounds(

        self,

        lower,

        upper,

    ):

        """

        Update stacked input bounds.



        lower / upper can be either:

            shape (nu,)

        or:

            shape (N, nu)

        """



        lower = np.asarray(

            lower,

            dtype=float,

        )



        upper = np.asarray(

            upper,

            dtype=float,

        )



        if lower.shape == (self.nu,):

            lower = np.tile(

                lower,

                self.N,

            )



        elif lower.shape == (

            self.N,

            self.nu,

        ):

            lower = lower.reshape(-1)



        else:

            raise ValueError(

                "Invalid lower-bound shape."

            )



        if upper.shape == (self.nu,):

            upper = np.tile(

                upper,

                self.N,

            )



        elif upper.shape == (

            self.N,

            self.nu,

        ):

            upper = upper.reshape(-1)



        else:

            raise ValueError(

                "Invalid upper-bound shape."

            )



        if np.any(lower > upper):

            raise ValueError(

                "Lower bound exceeds upper bound."

            )



        self.lower = lower

        self.upper = upper



        self.solver.update(

            l=self.lower,

            u=self.upper,

        )