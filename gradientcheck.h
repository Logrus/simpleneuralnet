#ifndef GRADIENTCHECK_H
#define GRADIENTCHECK_H
#include "layer.h"
#include <eigen3/Eigen/Dense>
using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMajorMat;

inline RowMajorMat eval_numerical_gradient_matrix(std::shared_ptr<Layer> l, RowMajorMat X){
    RowMajorMat grad(X);
}

#endif // GRADIENTCHECK_H
