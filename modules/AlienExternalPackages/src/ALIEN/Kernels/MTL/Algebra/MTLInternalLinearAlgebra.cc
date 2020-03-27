#include <ALIEN/Kernels/MTL/Algebra/MTLInternalLinearAlgebra.h>

#include <alien/core/backend/LinearAlgebraT.h>

#include <ALIEN/Kernels/MTL/DataStructure/MTLInternal.h>
#include <ALIEN/Kernels/MTL/DataStructure/MTLMatrix.h>
#include <ALIEN/Kernels/MTL/DataStructure/MTLVector.h>
#include <ALIEN/Kernels/MTL/MTLBackEnd.h>
#include <boost/numeric/mtl/mtl.hpp>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>

#include <arccore/base/NotImplementedException.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ALIEN_EXTERNALPACKAGES_EXPORT LinearAlgebra<BackEnd::tag::mtl>;

IInternalLinearAlgebra<MTLMatrix, MTLVector>*
MTLInternalLinearAlgebraFactory()
{
  return new MTLInternalLinearAlgebra();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MTLInternalLinearAlgebra::MTLInternalLinearAlgebra()
{
  // Devrait faire le MTLInitialize qui est actuellement dans le solveur
  // Attention, cette initialisation serait globale et non restreinte à cet objet
}

/*---------------------------------------------------------------------------*/

MTLInternalLinearAlgebra::~MTLInternalLinearAlgebra()
{
}

/*---------------------------------------------------------------------------*/

Arccore::Real
MTLInternalLinearAlgebra::norm0(const Vector& x) const
{
  return infinity_norm(x.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

Arccore::Real
MTLInternalLinearAlgebra::norm1(const Vector& x) const
{
  return one_norm(x.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

Arccore::Real
MTLInternalLinearAlgebra::norm2(const Vector& x) const
{
  return two_norm(x.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearAlgebra::mult(const Matrix& a, const Vector& x, Vector& r) const
{
  r.internal()->m_internal = a.internal()->m_internal * x.internal()->m_internal;
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearAlgebra::axpy(
    const Arccore::Real& alpha, const Vector& x, Vector& r) const
{
  r.internal()->m_internal += alpha * x.internal()->m_internal;
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearAlgebra::aypx(
    const Arccore::Real& alpha, Vector& y, const Vector& x) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "MTLInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearAlgebra::copy(const Vector& x, Vector& r) const
{
  r.internal()->m_internal = x.internal()->m_internal;
}

/*---------------------------------------------------------------------------*/

Arccore::Real
MTLInternalLinearAlgebra::dot(const Vector& x, const Vector& y) const
{
  return dot_real(x.internal()->m_internal, y.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearAlgebra::scal(const Arccore::Real& alpha, Vector& x) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "MTLInternalLinearAlgebra::scal not implemented");
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearAlgebra::diagonal(const Matrix& a, Vector& x) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "MTLInternalLinearAlgebra::diagonal not implemented");
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearAlgebra::reciprocal(Vector& x) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "MTLInternalLinearAlgebra::reciprocal not implemented");
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearAlgebra::pointwiseMult(const Vector& x, const Vector& y, Vector& w) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "MTLInternalLinearAlgebra::pointwiseMult not implemented");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
MTLInternalLinearAlgebra::mult(const Matrix& a, const UniqueArray<Real>& x, UniqueArray<Real>& r) const {
 throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::mult not implemented");
}
void
MTLInternalLinearAlgebra::axpy(const Real & alpha, const UniqueArray<Real>& x, UniqueArray<Real>& r) const {
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::axpy not implemented");
}
void
MTLInternalLinearAlgebra::aypx(const Real & alpha, UniqueArray<Real>& y, const UniqueArray<Real>& x) const {
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::aypx not implemented");
}
void
MTLInternalLinearAlgebra::copy(const UniqueArray<Real>& x, UniqueArray<Real>& r) const {
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::copy not implemented");
}
Real
MTLInternalLinearAlgebra::dot(Integer local_size, const UniqueArray<Real>& x, const UniqueArray<Real>& y) const {
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::dot not implemented");
return Real() ;
}
void
MTLInternalLinearAlgebra::scal(const Real & alpha, UniqueArray<Real>& x) const {
  throw NotImplementedException(A_FUNCINFO, "HypreLinearAlgebra::scal not implemented");
}

} // END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
