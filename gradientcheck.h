#ifndef GRADIENTCHECK_H
#define GRADIENTCHECK_H
#include <eigen3/Eigen/Dense>
using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMajorMat;

inline RowMajorMat eval_numerical_gradient_matrix(RowMajorMat (*func)( )){

}

#endif // GRADIENTCHECK_H
