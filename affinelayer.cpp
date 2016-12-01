#include "affinelayer.h"

AffineLayer::AffineLayer()
{

}

void AffineLayer::forward()
{
    Eigen::MatrixXf ans = (*X)*(*W);
    ans.rowwise() += (*b).transpose();
    out = std::make_shared<RowMajorMatrix> (ans);
}

void AffineLayer::backward()
{
    // dX

    // dW

    // db
}
