#ifndef GRADIENTCHECK_H
#define GRADIENTCHECK_H
#include "layer.h"
#include <eigen3/Eigen/Dense>
using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMajorMat;

inline RowMajorMat eval_numerical_gradient_matrix(Layer* l, RowMajorMat &X, RowMajorMat df, float h=1e-5){
    RowMajorMat grad(X.rows(), X.cols());
    for(int c=0;c<X.cols();++c){
        for(int r=0;r<X.rows();++r){
            float oldval = X(r,c);
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
