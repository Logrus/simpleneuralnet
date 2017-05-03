#ifndef AFFINELAYER_H
#define AFFINELAYER_H
#include "layer.h"
#include <memory>
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

template <typename T>
class AffineLayer : public Layer<T>
{
public:
    void forward();
    void backward();

    using Layer<T>::RowMajorMatrix;
    using Layer<T>::RowMajorVector;
};


#endif // AFFINELAYER_H
