#ifndef LAYER_H
#define LAYER_H
#include <memory>
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

template <typename T>
class Layer
{
    typedef Matrix<T, Dynamic, Dynamic, RowMajor> RowMajorMatrix;
    typedef Matrix<T, 1, Dynamic> RowMajorVector;
public:

    virtual void forward() = 0;
    virtual void backward() = 0;

    // Inputs
    std::shared_ptr<RowMajorMatrix> X;
    std::shared_ptr<RowMajorMatrix> W;
    std::shared_ptr<RowMajorVector> b;

    // Forward pass output
    std::shared_ptr<RowMajorMatrix> out;

    // Backward pass input radient
    std::shared_ptr<RowMajorMatrix> dout;

    // Backward pass gradients
    std::shared_ptr<RowMajorMatrix> dX;
    std::shared_ptr<RowMajorMatrix> dW;
    std::shared_ptr<RowMajorVector> db;
};

#endif // LAYER_H
