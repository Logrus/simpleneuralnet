#ifndef GRADIENTCHECK_H
#define GRADIENTCHECK_H
#include "layer.h"
#include <eigen3/Eigen/Dense>
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMajorMatDouble;

inline RowMajorMat eval_numerical_gradient_matrix(Layer* l, RowMajorMatDouble &X, RowMajorMatDouble df, double h=1e-5){
    RowMajorMat grad(X.rows(), X.cols());
    for(int c=0;c<X.cols();++c){
        for(int r=0;r<X.rows();++r){
            double oldval = X(r,c);
            X (r,c) = oldval + h;
            l->forward();
            auto pos = l->out;
            X (r,c) = oldval - h;
            l->forward();
            auto neg = l->out;
            X(r,c) = oldval;

            auto mat = ((*pos-*neg).cwiseProduct(df)) / (2*h);

            grad(r,c) = mat.sum();
        }
    }
    return grad;

}

#endif // GRADIENTCHECK_H
