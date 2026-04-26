from typing import Union

import matplotlib.pyplot as plt
import numpy.linalg as nplg
import numpy.typing as npt
import numpy as np
import sympy as sp

type NumArray = npt.NDArray[np.number]

class Linear2DSystem:
    def __init__(self, A: Union[NumArray|str], b: Union[NumArray|str], symbols = 'x y'):
        """ Creates a Lienar Dinamical System 2x2:
        dX/dt = A*X + b
        """
        if type(A) is str:
            A = np.asmatrix(A)
        if type(b) is str:
            b = np.asmatrix(b)

        # check argument dimensions
        self.A = np.array(A)
        if self.A.shape != (2, 2):
            raise ValueError
        self.b = np.array(b).reshape(2, 1)

        self.sym_A = sp.Matrix(A)
        self.sym_b = sp.Matrix(b)

        x, y = sp.symbols(symbols)
        self.sym_X = sp.Matrix([[x],[y]])
    
    def __repr__(self):
        return self.__str__()
    
    def __symrepr__(self):
        return self.__call__(self.sym_X)

    def show(self):
        sp.pprint(self.__symrepr__())
    
    def __str__(self):
        return self.__call__(self.sym_X).__str__()

    def __symrepr__(self):
        return self.sym_A*self.sym_X + self.b

    def __call__(self, X):
        if type(X) is list or type(X) is tuple:
            X = np.array(X)
        X = X.reshape(2,1)
        if X is NumArray:
            return self.A.dot(X) + self.b
        return self.sym_A*X + self.b

    def eval(self, X: NumArray):
        """Return flatten np.array of F(X)
        """
        return np.array(self.__call__(X)).flatten()
    
    def plot(self, x_range = (-10, 10), y_range = (-10, 10), n_grid = 25, density=1):
        fig, ax = plt.subplots(figsize=(5, 5))
        # draw singular points
        x_min, x_max = x_range
        y_min, y_max = y_range
        ax.grid(True)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

        try:
            x0, y0 = np.array(nplg.inv(self.A).dot(-self.b)) # calcula a singularidade
            singular_pts = True
            print(f'Singularidade: ({x0[0]},{y0[0]})')
        except nplg.LinAlgError: 
            x0, y0 = np.array(-self.b).flatten() # por padrao (0,0) é singular para os casos degenerados
            singular_pts = False
        
        # Draw streamplot
        # Create a grid for the vector field
        x_grid = np.linspace(x_min, x_max, n_grid)
        y_grid = np.linspace(y_min, y_max, n_grid)
        X, Y = np.meshgrid(x_grid, y_grid)

        U = np.zeros(X.shape)
        V = np.zeros(Y.shape)
        # Evaluate the vector field at each grid point
        for i in range(n_grid):
            for j in range(n_grid):
                X_vec = np.array([X[i, j], Y[i, j]])
                dx_dt = self.eval(X_vec)
                U[i, j] = dx_dt[0]
                V[i, j] = dx_dt[1]
        props = dict(
            color='gray',
            linewidth=0.7,
            density=density,
            arrowstyle='->',
            arrowsize=1.5,
            minlength=1,
            broken_streamlines=False
        )
        ax.streamplot(X, Y, U, V, **props)

        # Draw autovects lines
        print('Autovalores:')
        for lbd, mul, vecs in self.sym_A.eigenvects():
            print(f'\tlbd: {lbd}, mul: {mul}:')
            # Se não tem ponto singularunico
            # entao se o autovalor é zero o vetor indica as singularidade
            # se nao for, então pula para o proximo vetor
            if sp.im(lbd) != 0: continue
            if (not singular_pts) and lbd: continue
            for a, b in vecs:
                print(f'\t\tVetor: ({a}, {b})')
                if np.abs(a) > np.abs(b):
                    # x cresce mais
                    t1 = (x_min - x0)/a
                    t2 = (x_max - x0)/a
                else:
                    # y cresce mais
                    t1 = (y_min - y0)/b
                    t2 = (y_max - y0)/b
                # Isso nao garante que saia do frame range
                # Mas se sair é pq o ponto singular é mal posicionado nos limites
                # dai a linha 67 corta o que estiver fora do range
                formating = 'r--' if lbd < 0 else 'b--'
                ax.plot([x0 + a*t1, x0 + a*t2], [y0 + b*t1, y0 + b*t2], formating)
        if singular_pts:
            ax.plot(x0, y0, 'ko')
        fig.show()

    def __add__(self, other):
        return self.__symrepr__() + other
    def __sub__(self, other):
        return self.__symrepr__() - other
    def __mul__(self, other):
        return self.__symrepr__() * other
    def __truediv__(self, other):
        return self.__symrepr__() / other
    def __floordiv__(self, other):
        return self.__symrepr__() // other

