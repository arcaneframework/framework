#include "mpi.h"

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/core/backend/LinearAlgebraT.h>

#include "alien/kernels/mcg/data_structure/MCGMatrix.h"
#include "alien/kernels/mcg/data_structure/MCGVector.h"
#include "alien/kernels/mcg/data_structure/MCGInternal.h"
#include "alien/kernels/mcg/algebra/MCGInternalLinearAlgebra.h"

namespace Alien {

ILinearAlgebra*
MCGInternalLinearAlgebraFactory()
{
  return new MCGInternalLinearAlgebra();
}

/*---------------------------------------------------------------------------*/

MCGInternalLinearAlgebra::MCGInternalLinearAlgebra()
{
  // Devrait faire le MCGInitialize qui est actuellement dans le solveur
  // Attention, cette initialisation serait globale et non restreinte à cet objet
}

/*---------------------------------------------------------------------------*/

MCGInternalLinearAlgebra::~MCGInternalLinearAlgebra() {}

/*---------------------------------------------------------------------------*/

Real
MCGInternalLinearAlgebra::norm0(const VectorType& x) const
{
  const MCGVector& vx = x.impl()->get<BackEnd::tag::mcgsolver>();
  return norm0(vx);
}

/*---------------------------------------------------------------------------*/

Real
MCGInternalLinearAlgebra::norm1(const VectorType& x) const
{
  const MCGVector& vx = x.impl()->get<BackEnd::tag::mcgsolver>();
  return norm1(vx);
}

/*---------------------------------------------------------------------------*/

Real
MCGInternalLinearAlgebra::norm2(const VectorType& x) const
{
  const MCGVector& vx = x.impl()->get<BackEnd::tag::mcgsolver>();
  return norm2(vx);
}

/*---------------------------------------------------------------------------*/

void
MCGInternalLinearAlgebra::mult(
    const MatrixType& a, const VectorType& x, VectorType& r) const
{
  const MCGMatrix& ma = a.impl()->get<BackEnd::tag::mcgsolver>();
  const MCGVector& vx = x.impl()->get<BackEnd::tag::mcgsolver>();
  MCGVector& vr = r.impl()->get<BackEnd::tag::mcgsolver>(true);
  ALIEN_ASSERT(
      (ma.colSpace() == vx.space() && vx.space() == vr.space()), ("Incompatible spaces"));
  return mult(ma, vx, vr);
}

/*---------------------------------------------------------------------------*/

void
MCGInternalLinearAlgebra::axpy(
    Real alpha, const VectorType& x, VectorType& r) const
{
  const MCGVector& vx = x.impl()->get<BackEnd::tag::mcgsolver>();
  MCGVector& vr = r.impl()->get<BackEnd::tag::mcgsolver>(true);
  ALIEN_ASSERT((vx.space() == vr.space()), ("Incompatible spaces"));
  return axpy(alpha, vx, vr);
}

/*---------------------------------------------------------------------------*/
void
MCGInternalLinearAlgebra::copy(const VectorType& x, VectorType& r) const
{
  const MCGVector& vx = x.impl()->get<BackEnd::tag::mcgsolver>();
  MCGVector& vr = r.impl()->get<BackEnd::tag::mcgsolver>(true);
  ALIEN_ASSERT((vx.space() == vr.space()), ("Incompatible spaces"));
  return copy(vx, vr);
}

/*---------------------------------------------------------------------------*/

Real
MCGInternalLinearAlgebra::dot(const VectorType& x, const VectorType& y) const
{
  const MCGVector& vx = x.impl()->get<BackEnd::tag::mcgsolver>();
  const MCGVector& vy = y.impl()->get<BackEnd::tag::mcgsolver>();
  ALIEN_ASSERT((vx.space() == vy.space()), ("Incompatible space"));
  return dot(vx, vy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real
MCGInternalLinearAlgebra::norm0(const VectorImpl& vx) const
{
  Real result = 0;
  // result = infinity_norm(vx.internal()->m_internal);
  return result;
}

/*---------------------------------------------------------------------------*/

Real
MCGInternalLinearAlgebra::norm1(const VectorImpl& vx) const
{
  Real result = 0;
  // result = one_norm(vx.internal()->m_internal);
  return result;
}

/*---------------------------------------------------------------------------*/

Real
MCGInternalLinearAlgebra::norm2(const VectorImpl& vx) const
{
  Real result = 0;
  // result = two_norm(vx.internal()->m_internal);
  return result;
}

/*---------------------------------------------------------------------------*/

void
MCGInternalLinearAlgebra::mult(
    const MatrixImpl& ma, const VectorImpl& vx, VectorImpl& vr) const
{
  // mult(ma.internal()->m_internal,vx.internal()->m_internal,vr.internal()->m_internal);
  // vr.internal()->m_internal= ma.internal()->m_internal*vx.internal()->m_internal;
}

/*---------------------------------------------------------------------------*/

void
MCGInternalLinearAlgebra::axpy(
    Real alpha, const VectorImpl& vx, VectorImpl& vr) const
{
  // vr.internal()->m_internal += alpha*vx.internal()->m_internal;
}

/*---------------------------------------------------------------------------*/
void
MCGInternalLinearAlgebra::copy(const VectorImpl& vx, VectorImpl& vr) const
{
  // copy(vx.internal()->m_internal,vr.internal()->m_internal);
  // vr.internal()->m_internal=vx.internal()->m_internal;
}

/*---------------------------------------------------------------------------*/

Real
MCGInternalLinearAlgebra::dot(const VectorImpl& vx, const VectorImpl& vy) const
{
  Real result = 0;
  // result = dot_real(vx.internal()->m_internal,vy.internal()->m_internal);
  return result;
}

} // namespace Alien
