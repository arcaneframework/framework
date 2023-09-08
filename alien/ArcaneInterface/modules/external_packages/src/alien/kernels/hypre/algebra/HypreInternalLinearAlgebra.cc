#include "HypreInternalLinearAlgebra.h"

#include <HYPRE_parcsr_mv.h>
#include <HYPRE_IJ_mv.h>

#include <alien/kernels/hypre/HypreBackEnd.h>

#include <alien/core/backend/LinearAlgebraT.h>
#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>
#include <alien/kernels/hypre/data_structure/HypreInternal.h>
#include <alien/data/Space.h>

#include <arccore/base/NotImplementedException.h>

#include <cmath>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace {
  HYPRE_ParVector hypre_implem(const HypreVector& v)
  {
    HYPRE_ParVector res;
    HYPRE_IJVectorGetObject(v.internal()->internal(), reinterpret_cast<void**>(&res));
    return res;
  }

  HYPRE_ParCSRMatrix hypre_implem(const HypreMatrix& m)
  {
    HYPRE_ParCSRMatrix res;
    HYPRE_IJMatrixGetObject(m.internal()->internal(), reinterpret_cast<void**>(&res));
    return res;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ALIEN_EXTERNAL_PACKAGES_EXPORT LinearAlgebra<BackEnd::tag::hypre>;

extern ALIEN_EXTERNAL_PACKAGES_EXPORT IInternalLinearAlgebra<HypreMatrix, HypreVector>*
HypreInternalLinearAlgebraFactory()
{
  return new HypreInternalLinearAlgebra();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HypreInternalLinearAlgebra::HypreInternalLinearAlgebra()
{
  // Devrait faire le HypreInitialize qui est actuellement dans le solveur
  // Attention, cette initialisation serait globale et non restreinte à cet objet
}

/*---------------------------------------------------------------------------*/

HypreInternalLinearAlgebra::~HypreInternalLinearAlgebra()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arccore::Real
HypreInternalLinearAlgebra::norm0(const HypreVector& vx ALIEN_UNUSED_PARAM) const
{
  Arccore::Real result = 0;
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HypreLinearAlgebra::norm0 not implemented");
  // VecNorm(vx.internal()->m_internal, NORM_INFINITY, &result);
  return result;
}

/*---------------------------------------------------------------------------*/

Arccore::Real
HypreInternalLinearAlgebra::norm1(const HypreVector& vx ALIEN_UNUSED_PARAM) const
{
  Arccore::Real result = 0;
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HypreLinearAlgebra::norm1 not implemented");
  // VecNorm(vx.internal()->m_internal, NORM_1, &result);
  return result;
}

/*---------------------------------------------------------------------------*/

Arccore::Real
HypreInternalLinearAlgebra::norm2(const HypreVector& vx) const
{
  return std::sqrt(dot(vx, vx));
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearAlgebra::mult(
    const HypreMatrix& ma, const HypreVector& vx, HypreVector& vr) const
{
  HYPRE_ParCSRMatrixMatvec(
      1.0, hypre_implem(ma), hypre_implem(vx), 0.0, hypre_implem(vr));
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearAlgebra::axpy(Real alpha ALIEN_UNUSED_PARAM,
    const HypreVector& vx ALIEN_UNUSED_PARAM, HypreVector& vr ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HypreLinearAlgebra::axpy not implemented");
}

/*---------------------------------------------------------------------------*/
void
HypreInternalLinearAlgebra::copy(const HypreVector& vx, HypreVector& vr) const
{
  HYPRE_ParVectorCopy(hypre_implem(vx), hypre_implem(vr));
}

/*---------------------------------------------------------------------------*/

Arccore::Real
HypreInternalLinearAlgebra::dot(const HypreVector& vx, const HypreVector& vy) const
{
  double dot_prod = 0;
  HYPRE_ParVectorInnerProd(hypre_implem(vx), hypre_implem(vy), &dot_prod);
  return static_cast<Arccore::Real>(dot_prod);
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearAlgebra::diagonal(
    HypreMatrix const& m ALIEN_UNUSED_PARAM, HypreVector& v ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HypreLinearAlgebra::diagonal not implemented");
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearAlgebra::reciprocal(HypreVector& v ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HypreLinearAlgebra::reciprocal not implemented");
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearAlgebra::aypx(Real alpha ALIEN_UNUSED_PARAM,
    HypreVector& y ALIEN_UNUSED_PARAM, const HypreVector& x ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HypreLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearAlgebra::pointwiseMult(const HypreVector& x ALIEN_UNUSED_PARAM,
    const HypreVector& y ALIEN_UNUSED_PARAM, HypreVector& w ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(
      A_FUNCINFO, "HypreLinearAlgebra::pointwiseMult not implemented");
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearAlgebra::scal(Real alpha, HypreVector& x) const
{
  HYPRE_ParVectorScale(static_cast<double>(alpha), hypre_implem(x));
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearAlgebra::mult(
    const Matrix& a, const UniqueArray<Real>& x, UniqueArray<Real>& r) const
{
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::mult not implemented");
}

void
HypreInternalLinearAlgebra::axpy(
    Real alpha, const UniqueArray<Real>& x, UniqueArray<Real>& r) const
{
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::norm0 not implemented");
}

void
HypreInternalLinearAlgebra::aypx(
    Real alpha, UniqueArray<Real>& y, const UniqueArray<Real>& x) const
{
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::axpy not implemented");
}
void
HypreInternalLinearAlgebra::copy(const UniqueArray<Real>& x, UniqueArray<Real>& r) const
{
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::copy not implemented");
}
Real
HypreInternalLinearAlgebra::dot(
    Integer local_size, const UniqueArray<Real>& x, const UniqueArray<Real>& y) const
{
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::dot not implemented");
  return Real();
}
void
HypreInternalLinearAlgebra::scal(Real alpha, UniqueArray<Real>& x) const
{
  throw NotImplementedException(A_FUNCINFO, "HypreLinearAlgebra::scal not implemented");
}
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
