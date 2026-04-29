from dpws_plotter.linear_system import Linear2DSystem
from pytest import raises
import sympy as sp


def test_class_declaration():
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
    assert type(F) == Linear2DSystem

    # Test arithmetics:
    v = sp.Matrix([[1],[1]])
    assert str(F) == 'Matrix([[x + 1], [y + 1]])'
    assert F(v)[0,0] == 2
    assert F(v)[1,0] == 2

def test_plotting():
    # Identity (proper node or star)
    F = Linear2DSystem([[1, 0], [0, 1]], [0, 0])
    F.print()
    F.plot()
    print('=============')
    # proper node
    F = Linear2DSystem([[1, 0], [0, 2]], [0, 0])
    F.print()
    F.plot()
    print('=============')
    # improper node
    F = Linear2DSystem([[1, 0], [1, 1]], [0, 0])
    F.print()
    F.plot()
    print('=============')
    # Line Fixed Points
    F = Linear2DSystem([[0, 0], [0, 1]], [0, 0])
    F.print()
    F.plot()
    print('=============')
    # Saddle
    F = Linear2DSystem([[1, 0], [0, -1]], [0, 0])
    F.print()
    F.plot()
    print('=============')
    # center
    F = Linear2DSystem([[0, -1], [1, 0]], [0, 0])
    F.print()
    F.plot()
    print('=============')
    # spiral
    F = Linear2DSystem([[1, -2], [2, 1]], [0, 0])
    F.print()
    F.plot()
    print('=============')
