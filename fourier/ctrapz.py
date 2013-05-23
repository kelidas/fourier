
from scipy import weave
from scipy import trapz
import numpy as np



def ctrapz(y, x):
    x_shape = x.shape[0]
    y_shape = 1
    if y.ndim > 1:
        y_shape = y.shape[0]

    res = np.zeros(y_shape)

    C_code = '''
            #line 29 "ctrapz.py"
            double result=0;
            for(int j = 0; j < y_shape; j++){
                for(int i = 0; i < x_shape-1; i++){
                    result += (y(i) + y(i+1)) / 2.0 * (x(i+1) - x(i));  
                    };
                 res(j) = result;
                 result=0;
            };
            '''

    weave.inline(C_code, ['x', 'y', 'x_shape', 'y_shape', 'res'], type_converters=weave.converters.blitz, compiler='gcc')
    return res

if __name__ == '__main__':
    x = np.array([1, 2, 3, 4])
    y = np.array([3, 2, 3, 4])
    print ctrapz(y, x)
    print trapz(y, x)
