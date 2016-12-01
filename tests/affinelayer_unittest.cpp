#include "../affinelayer.h"
#include "../gradientcheck.h"
#include "gtest/gtest.h"
#include <memory>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

TEST(AffineLayerTest, ForwardPass) {
  MatrixXf correct_out(2,3);
  correct_out << 1.49834967,  1.70660132,  1.91485297,
                 3.25553199,  3.5141327,   3.77273342;

  int out_size = 3;
  int in_size  = 4*5*6;
  int batch_size = 2;

  AffineLayer a;
  a.X = std::make_shared<RowMajorMatrix>(batch_size,in_size);
  a.W = std::make_shared<RowMajorMatrix>(in_size, out_size);
  a.b = std::make_shared<VectorXf>(out_size);

  // Fill values
  int x_size = a.X->rows()*a.X->cols();
  Map<RowVectorXf> xvec (a.X->data(), x_size);
  xvec = Eigen::RowVectorXf::LinSpaced(x_size, -0.1, 0.5);

  int w_size = a.W->rows()*a.W->cols();
  Map<RowVectorXf> wvec (a.W->data(), w_size);
  wvec = RowVectorXf::LinSpaced(w_size, -0.2, 0.3);

  (*a.b) = VectorXf::LinSpaced(out_size, -0.3, 0.1);

  a.forward();

  for (int r=0; r<correct_out.rows(); ++r)
      for (int c=0; c<correct_out.cols(); ++c)
          EXPECT_FLOAT_EQ((*a.out)(r,c),correct_out(r,c));

}


TEST(AffineLayerTest, BackwardPass) {

  int out_size = 5;
  int in_size  = 2*3;
  int batch_size = 10;

  AffineLayer a;
  a.X = std::make_shared<RowMajorMatrix>(RowMajorMatrix::Random(batch_size, in_size));
  a.W = std::make_shared<RowMajorMatrix>(RowMajorMatrix::Random(in_size, out_size));
  a.b = std::make_shared<VectorXf>(VectorXf::Random(out_size));

  a.dout = std::make_shared<RowMajorMatrix>(RowMajorMatrix::Random(batch_size,out_size));

  a.backward();
  auto grad = a.out;

  // Do gradcheck
  auto num_grad = eval_numerical_gradient_matrix(&a, *a.X, *a.dout);
  std::cout << "Showing numerical gradient" << std::endl;
  std::cout << num_grad << std::endl;

}
