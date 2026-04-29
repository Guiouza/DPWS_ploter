""" Porcelain functions for plotting tools
author: Guilherme Meneghin de Souza (guilherme.meneghinsouza@gmail.com)
"""

import dpws_plotter.linear_system as ls

def plot(A, b, A2=None, b2=None):
    # Create Linear Systems
    F1 = ls.Linear2DSystem(A, b)
    print('F1(x,y) = ')
    F1.print()
    F2 = None
    if A2 is not None:
        if b2 is not None:
            b2 = '[0, 0]'
        F2 = ls.Linear2DSystem(A2, b2)
        print('F2(x,y) = ')
        F2.print()
    
    f1_range = (-10, 10)
    f2_range = (-10, 10)

    if F2 is None:
        F1.plot()
    else:
        print('Not coded yet')
        raise