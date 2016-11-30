#ifndef AFFINELAYER_H
#define AFFINELAYER_H
#include "layer.h"
#include <memory>
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

typedef Matrix<float,Dynamic,Dynamic, RowMajor> RowMajorMatrix;

class AffineLayer : public Layer
{
public:
    AffineLayer();
    void forward();
    void backward();

    // Inputs
    std::shared_ptr<RowMajorMatrix> X;
    std::shared_ptr<RowMajorMatrix> W;
    std::shared_ptr<VectorXf> b;

    // Forward pass output
    std::shared_ptr<RowMajorMatrix> out;

    // Backward pass gradients
    std::shared_ptr<RowMajorMatrix> dX;
    std::shared_ptr<RowMajorMatrix> dW;
    std::shared_ptr<VectorXf> db;

};

#endif // AFFINELAYER_H
