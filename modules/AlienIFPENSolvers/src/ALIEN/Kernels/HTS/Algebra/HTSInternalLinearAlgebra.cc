
#include "HTSInternalLinearAlgebra.h"

#include <ALIEN/Kernels/HTS/HTSBackEnd.h>

#include <ALIEN/Core/Backend/LinearAlgebraT.h>
#include <ALIEN/Kernels/SimpleCSR/Algebra/CBLASMPIKernel.h>

#include <ALIEN/Data/Space.h>


#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRMatrix.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/Algebra/SimpleCSRInternalLinearAlgebra.h>

#include <ALIEN/Kernels/HTS/DataStructure/HTSMatrix.h>

#include <arccore/base/NotImplementedException.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

template class ALIEN_IFPENSOLVERS_EXPORT LinearAlgebra<BackEnd::tag::htssolver>;
template class ALIEN_IFPENSOLVERS_EXPORT
    LinearAlgebra<BackEnd::tag::hts, BackEnd::tag::simplecsr>;

/*---------------------------------------------------------------------------*/
IInternalLinearAlgebra<SimpleCSRMatrix<Arccore::Real>, SimpleCSRVector<Arccore::Real>>*
HTSSolverInternalLinearAlgebraFactory()
{
  return new HTSSolverInternalLinearAlgebra();
}

IInternalLinearAlgebra<HTSMatrix<Arccore::Real>, SimpleCSRVector<Arccore::Real>>*
HTSInternalLinearAlgebraFactory()
{
  return new HTSInternalLinearAlgebra();
}

/*---------------------------------------------------------------------------*/
HTSInternalLinearAlgebra::HTSInternalLinearAlgebra()
{
  // Devrait faire le HTSInitialize qui est actuellement dans le solveur
  // Attention, cette initialisation serait globale et non restreinte à cet objet
}

/*---------------------------------------------------------------------------*/

HTSInternalLinearAlgebra::~HTSInternalLinearAlgebra()
{
}

/*---------------------------------------------------------------------------*/

Arccore::Real
HTSInternalLinearAlgebra::norm0(const Vector& x) const
{
  return 0.;
}

/*---------------------------------------------------------------------------*/

Arccore::Real
HTSInternalLinearAlgebra::norm1(const Vector& x) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

Arccore::Real
HTSInternalLinearAlgebra::norm2(const Vector& x) const
{
  return CBLASMPIKernel::nrm2(x.distribution(), x);
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::mult(const Matrix& a, const Vector& x, Vector& r) const
{
  a.mult(x.getDataPtr(),r.getDataPtr()) ;
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::axpy(const Arccore::Real& alpha, const Vector& x, Vector& r) const
{
  CBLASMPIKernel::axpy(x.distribution(), alpha, x, r);
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::aypx(const Arccore::Real& alpha, Vector& y, const Vector& x) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::copy(const Vector& x, Vector& r) const
{
  CBLASMPIKernel::copy(x.distribution(), x, r);
}

/*---------------------------------------------------------------------------*/

Arccore::Real
HTSInternalLinearAlgebra::dot(const Vector& x, const Vector& y) const
{
  return CBLASMPIKernel::dot(x.distribution(), x, y);
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::scal(const Arccore::Real& alpha, Vector& x) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::scal not implemented");
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::diagonal(const Matrix& a, Vector& x) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::diagonal not implemented");
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::reciprocal(Vector& x) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::reciprocal not implemented");
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::pointwiseMult(const Vector& x, const Vector& y, Vector& w) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::pointwiseMult not implemented");
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
