import numpy._core.numeric as N
import matplotlib.pyplot as plt
import numpy.linalg as nlg
import numpy as np
import sympy as sp


class Linear2DSystem:
    """
    Creates a Linear Dynamical System 2x2:
    dX/dt = A*X + b

    Parameters
    ----------
    A: array_like or str
    b: array_like or str
    symbols: str

    Returns
    -------
    linsys: Linear Dynamical System

    Examples
    --------
    >>> from dpws_plotter.linear_system import Linear2DSystem
    >>> F = Linear2DSystem(((1,0), (0,1)), (0, 0))

    >>> F.print()
    ⎡x⎤
    ⎢ ⎥
    ⎣y⎦
    >>> F.plot()
    Singularity: (0.0,0.0)
    """
    def __init__(self, A, b, symbols='x y'):
        # type checking for A
        if isinstance(A, str):
            self.A: np.ndarray = np.array(np.asmatrix(A))
        else:
            self.A: np.ndarray = np.array(A)
        # type checking for b
        if isinstance(b, str):
            self.b: np.ndarray = np.array(np.asmatrix(b))
        else:
            self.b: np.ndarray = np.array(b)

        # Sanity Checks
        if self.A.shape != (2, 2):
            raise ValueError
        self.b = self.b.reshape(2, 1) # raise error if not possible

        self.sym_A: sp.Matrix = sp.Matrix(self.A)
        self.sym_b = sp.Matrix(self.b)
        x, y = sp.symbols(symbols)
        self.sym_X = sp.Matrix([[x], [y]])

        self.singularity = None
        try:
            x0, y0 = np.array(nlg.inv(self.A).dot(-self.b))  # calcula a singularidade
            self.singularity = (x0, y0)
        except nlg.LinAlgError:
            pass

        self.eigenvectors = self.sym_A.eigenvects()

    def print(self):
        sp.pprint(self.symrepr(), )

    def eval(self, xy):
        """Return flatten np.array of F(X)"""
        return np.array(self.__call__(xy)).flatten()

    def symrepr(self):
        return self.sym_A * self.sym_X + self.b

    def plot(self,
            x_range=(-10, 10),
            y_range=(-10, 10),
            n_grid=25, density=0.8):

        # Sanity Checks:
        if len(x_range) != 2:
            raise ValueError
        if len(y_range) != 2:
            raise ValueError
        if density < 0:
            raise ValueError

        fig, ax = plt.subplots(figsize=(5, 5))
        x_min, x_max = x_range
        y_min, y_max = y_range
        ax.grid(True)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # draw singular points
        if self.singularity is None:
            # por padrão -b é singular para os casos degenerados
            x0, y0 = np.array(-self.b).flatten()
            singular_pts = False
        else:
            x0, y0 = self.singularity
            singular_pts = True
            print(f'Singularity: ({x0[0]},{y0[0]})')

        # Draw streamplot
        # Create a grid for the vector field
        x_grid = np.linspace(x_min, x_max, n_grid)
        y_grid = np.linspace(y_min, y_max, n_grid)
        x, y = np.meshgrid(x_grid, y_grid)

        u = np.zeros(x.shape)
        v = np.zeros(y.shape)
        # Evaluate the vector field at each grid point
        for i in range(n_grid):
            for j in range(n_grid):
                x_vec = np.array([x[i, j], y[i, j]])
                dx_dt = self.eval(x_vec)
                u[i, j] = dx_dt[0]
                v[i, j] = dx_dt[1]

        props = dict(
            color='gray',
            linewidth=0.9,
            density=density,
            arrowstyle='->',
            arrowsize=1.5,
            minlength=0.1,
            broken_streamlines=False
        )
        plt.streamplot(x_grid, y_grid, u, v, **props)

        # Draw auto-spaces
        for lbd, mul, vecs in self.eigenvectors:
            # If there is no singularity, skip drawing auto-space process
            if self.singularity is None: break
            # If lbd is complex, skip drawing auto-space
            if sp.im(lbd) != 0: continue

            for a, b in vecs:
                # Find min and max values for the auto-space parameter t
                # auto-space line: (x0,y0) + t*(a,b)
                if np.abs(a) > np.abs(b):
                    # dx/dt > dy/dt: x is faster then y
                    t1 = (x_min - x0) / a
                    t2 = (x_max - x0) / a
                else:
                    # y is faster then x
                    t1 = (y_min - y0) / b
                    t2 = (y_max - y0) / b

                # Draw auto-space line
                formating = 'r--' if lbd < 0 else 'b--'
                ax.plot([x0 + a * t1, x0 + a * t2], [y0 + b * t1, y0 + b * t2], formating)
        # Draw singularity
        if self.singularity:
            ax.plot(x0, y0, 'ko')
        # Draw figure
        fig.show()

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return self.__call__(self.sym_X).__str__()
    def __call__(self, X):
        if isinstance(X, list | tuple):
            X = np.array(X).reshape(2,1)
            return self.A.dot(X) + self.b
        if isinstance(X, N.ndarray):
            return self.A.dot(X) + self.b
        if isinstance(X, sp.Matrix):
            return self.sym_A * X + self.b
        else:
            raise ValueError
    def __add__(self, other):
        return self.symrepr() + other
    def __sub__(self, other):
        return self.symrepr() - other
    def __mul__(self, other):
        return self.symrepr() * other
    def __truediv__(self, other):
        return self.symrepr() / other
