// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <arcane/utils/Array.h>

#include "Numerics/DiscreteOperator/TensorAlgebra.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real DiscreteOperator::MinimumEigenvalue::compute(const Real & A)
{
  return A;
}

/*---------------------------------------------------------------------------*/

Real DiscreteOperator::MinimumEigenvalue::compute(const Real3 & A)
{
  Real mineig = A[0];
  for (Integer i = 1; i < 3 ; i++)
    mineig = math::min(mineig,A[i]);
  return mineig;
}

/*---------------------------------------------------------------------------*/

Real DiscreteOperator::MinimumEigenvalue::compute(const Real3x3 & A)
{
  Array<Real> a(3);
  // compute the coefficients in charachteristic polynomial
  a[0] = -A.x.x - A.y.y - A.z.z;
  a[1] = -A.x.y * A.x.y - A.x.z * A.x.z  - A.y.z * A.y.z + A.x.x * A.y.y + A.x.x * A.z.z + A.y.y * A.z.z;
  a[2] =  A.x.y * A.x.y * A.z.z + A.x.z * A.x.z * A.y.y + A.y.z * A.y.z * A.x.x - A.x.x * A.y.y * A.z.z -2* A.x.y * A.x.z * A.y.z;

  Real q = -(3*a[1] - a[0]*a[0])/(9);
  Real r = -(9*a[0]*a[1] - 27*a[2] - 2*a[0]*a[0]*a[0])/54;

  Real q3 = q*q*q;
  Real r2 = r*r;
  Real root = 0.;
  if (q3-r2<-1e-14)
    std::cout<<"wrong discriminant in eigenvalue computaion(GradM scheme)!   D="<<q3-r2<<"\n"; //Here the program must crash!!!!
  else if (math::abs(q3-r2) < 1e-14){ // Zero discriminant
    Real A;
    if (r<0)
      A = pow(-r,1./3);
    else
      A = -pow(r,1./3);
    double tmp =  a[0]/3;
    root = math::min(2*A -  tmp, -A - tmp);
  }
  return root;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
