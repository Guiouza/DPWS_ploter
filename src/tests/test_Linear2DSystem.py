from dpws_ploter import Linear2DSystem
from pytest import raises
import sympy as sp


def test_Linear2DSystem():
    # Linear system initialization
    # wrong matrix:
    with raises(ValueError):
        # Bad matrix definition
        Linear2DSystem([[1, 1], [1]], [0, 0]) 
    with raises(ValueError):
        # Matrix != from 2x2
        Linear2DSystem([[1, 1, 1], [1, 1, 1]], [0, 0])
    
    Linear2DSystem([[1, 0 ], [0, 1]], [0, 0])
    F = Linear2DSystem("[1 0; 0 1]", "[1, 1]")

    # test aritmetics
    v = sp.Matrix([[1],[1]])
    assert str(F) == 'Matrix([[x + 1], [y + 1]])'
    assert F(v)[0,0] == 2
    assert F(v)[1,0] == 2