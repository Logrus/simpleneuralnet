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
};

#endif // AFFINELAYER_H
