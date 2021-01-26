#include <alien/utils/Precomp.h>
#include <cmath>

#include <gtest/gtest.h>
#include <alien/ref/AlienRefSemantic.h>

#include <alien/ref/AlienImportExport.h>
#include <alien/ref/mv_expr/MVExpr.h>


namespace Environment {
extern Alien::IMessagePassingMng* parallelMng();
extern Alien::ITraceMng* traceMng();
}

// Tests the default c'tor.
TEST(TestMVExpr, ExprEngine)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = Environment::traceMng();
  Integer global_size = 10;
  const Alien::Space s(10, "MySpace");
  Alien::MatrixDistribution mdist(s, s, Environment::parallelMng());
  Alien::VectorDistribution vdist(s, Environment::parallelMng());
  Alien::Matrix A(mdist); // A.setName("A") ;
  Alien::Vector x(vdist); // x.setName("x") ;
  Alien::Vector y(vdist); // y.setName("y") ;
  Alien::Vector r(vdist); // r.setName("r") ;
  Alien::Real lambda = 0.5;

  auto local_size = vdist.localSize();
  auto offset = vdist.offset();
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  {
    Alien::DirectMatrixBuilder builder(A, tag);
    builder.reserve(3 * local_size);
    builder.allocate();
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      builder(row, row) = 2.;
      if (row + 1 < global_size)
        builder(row, row + 1) = -1.;
      if (row - 1 >= 0)
        builder(row, row - 1) = -1.;
    }
  }
  {
    Alien::LocalVectorWriter writer(x);
    for (Integer i = 0; i < local_size; ++i)
      writer[i] = 1.;
  }

  using namespace Alien::MVExpr;

  {
    auto expr = mul(ref(A), ref(x));
    auto evaluated = eval(expr);
  }

  {
    auto expr = add(ref(x), ref(y));
    auto evaluated = eval(expr);
  }

  {
    auto expr = mul(ref(A), add(ref(x), ref(y)));
    auto evaluated = eval(expr);
  }

  {
    // auto expr = mul(ref(A), add(ref(x),mul(cst(lambda),ref(y)))) ;
    // auto evaluated = eval(expr);
  }

  Alien::SimpleCSRLinearAlgebra alg;
  // trace_mng->info()<<" NORME : "<<x.name()<<" = "<<alg.norm2(x) ;
  ASSERT_EQ(alg.norm2(x), std::sqrt(global_size));
  {
    trace_mng->info() << "y = lambda*x";
    assign(y, lambda * x);
    y = lambda * x;
    // trace_mng->info()<<" NORME : "<<y.name()<<" = "<<alg.norm2(y)  ;
    ASSERT_EQ(alg.norm2(y), std::sqrt(global_size * 0.25));

    trace_mng->info() << "y=A*x";
    assign(y, mul(ref(A), ref(x)));
    y = A * x;
    // trace_mng->info()<<" NORME : "<<y.name()<<" = "<<alg.norm2(y)  ;
    ASSERT_EQ(alg.norm2(y), std::sqrt(2));

    trace_mng->info() << "r=y-A*x";
    r = y - A * x;
    // trace_mng->info()<<" NORME : "<<r.name()<<" = "<<alg.norm2(r)  ;
    ASSERT_EQ(alg.norm2(r), 0.);

    trace_mng->info() << "x=x+lambda*r";
    y = y + (lambda * x);
    // trace_mng->info()<<" NORME : "<<y.name()<<" = "<<alg.norm2(y)  ;
    ASSERT_EQ(alg.norm2(y), std::sqrt(2 * 1.5 * 1.5 + (global_size - 2) * 0.25));

    trace_mng->info() << "y=A*lambda*r";
    y = A * (lambda * x);
    // trace_mng->info()<<" NORME : "<<y.name()<<" = "<<alg.norm2(y)  ;
    ASSERT_EQ(alg.norm2(y), std::sqrt(0.5));

    {
      Alien::VectorWriter writer(y);
      for (Integer i = 0; i < local_size; ++i)
        writer[offset + i] = offset + i;
    }
    Real value1 = eval(dot(x, y));
    trace_mng->info() << " DOT(x,y) " << value1;
    ASSERT_EQ(value1, 45);

    Real value2 = eval(dot(lambda * x, y));
    trace_mng->info() << " DOT(lambda*x,y) " << value2;
    ASSERT_EQ(value2, 22.5);

    Real value3 = eval(dot(x, r + y));
    trace_mng->info() << " DOT(x,r+y) " << value3;
    ASSERT_EQ(value3, 45);

    Real value4 = eval(dot(x, A * y));
    trace_mng->info() << " DOT(x,A*y) " << value4;
    ASSERT_EQ(value4, 9);
  }
  ////////////////////////////////////////////////
  //
  // TESTS PIPELINE ENABLING ASYNCHRONISM
  {
    pipeline(vassign(y, A * x), vassign(r, y - A * x));
  }
}
