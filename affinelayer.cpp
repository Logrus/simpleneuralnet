#include "affinelayer.h"

template <typename T>
void AffineLayer<T>::forward()
{
    auto ans = (*X)*(*W);
    ans.rowwise() += (*b).transpose();
    out = std::make_shared<::RowMajorMatrix> (ans);
}


template <typename T>
void AffineLayer<T>::backward()
{
    // dX
    // std::cout << W->rows() << " " << W->cols() << std::endl;
    // std::cout << X->rows() << " " << X->cols() << std::endl;
    // std::cout << dout->rows() << " " << dout->cols() << std::endl;

    dX = std::make_shared<::RowMajorMatrix>(*W * (*dout).transpose());
    // dW
    dW = std::make_shared<::RowMajorMatrix>((*X).transpose() * *dout);
    // db
    db = std::make_shared<::VectorXf>((*dout).rowwise().sum());
}
