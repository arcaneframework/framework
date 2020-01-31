#include "hypre_internal_linear_algebra.h"
#include "hypre_matrix.h"
#include "hypre_vector.h"
#include "hypre_internal.h"

#include <cmath>

#include <arccore/base/NotImplementedException.h>
#include <arccore/base/TraceInfo.h>

#include <HYPRE_parcsr_mv.h>
#include <_hypre_parcsr_mv.h>
#include <HYPRE_IJ_mv.h>

#include <ALIEN/Core/Backend/LinearAlgebraT.h>
#include <ALIEN/Data/Space.h>

namespace Alien {

  namespace {
    HYPRE_ParVector
    hypre_implem(const Hypre::Vector &v) {
      HYPRE_ParVector res;
      HYPRE_IJVectorGetObject(v.internal()->internal(), reinterpret_cast<void **>(&res));
      return res;
    }

    HYPRE_ParCSRMatrix
    hypre_implem(const Hypre::Matrix &m) {
      HYPRE_ParCSRMatrix res;
      HYPRE_IJMatrixGetObject(m.internal()->internal(), reinterpret_cast<void **>(&res));
      return res;
    }
  }

  template
  class ALIEN_HYPRE_EXPORT LinearAlgebra<BackEnd::tag::hypre>;

  IInternalLinearAlgebra<Hypre::Matrix, Hypre::Vector> *
  HypreInternalLinearAlgebraFactory() {
    return new Hypre::HypreInternalLinearAlgebra();
  }

  namespace Hypre {

    Arccore::Real
    HypreInternalLinearAlgebra::norm0(const Vector &vx ALIEN_UNUSED_PARAM) const {
      Arccore::Real result = 0;
      throw Arccore::NotImplementedException(A_FUNCINFO, "HypreLinearAlgebra::norm0 not implemented");
    }

    Arccore::Real
    HypreInternalLinearAlgebra::norm1(const Vector &vx ALIEN_UNUSED_PARAM) const {
      Arccore::Real result = 0;
      throw Arccore::NotImplementedException(A_FUNCINFO, "HypreLinearAlgebra::norm1 not implemented");
    }

    Arccore::Real
    HypreInternalLinearAlgebra::norm2(const Vector &vx) const {
      return std::sqrt(dot(vx, vx));
    }

    void
    HypreInternalLinearAlgebra::mult(
            const Matrix &ma, const Vector &vx, Vector &vr) const {
      HYPRE_ParCSRMatrixMatvec(1.0, hypre_implem(ma), hypre_implem(vx), 0.0, hypre_implem(vr));
    }

    void
    HypreInternalLinearAlgebra::axpy(
            const Arccore::Real &alpha ALIEN_UNUSED_PARAM, const Vector &vx ALIEN_UNUSED_PARAM, Vector &vr
            ALIEN_UNUSED_PARAM) const {
      HYPRE_ParVectorAxpy(alpha,  hypre_implem(vx), hypre_implem(vr));
    }

    void
    HypreInternalLinearAlgebra::copy(const Vector &vx, Vector &vr) const {
      HYPRE_ParVectorCopy(hypre_implem(vx), hypre_implem(vr));
    }

    Arccore::Real
    HypreInternalLinearAlgebra::dot(const Vector &vx, const Vector &vy) const {
      double dot_prod = 0;
      HYPRE_ParVectorInnerProd(hypre_implem(vx), hypre_implem(vy), &dot_prod);
      return static_cast<Arccore::Real>(dot_prod);
    }

    void
    HypreInternalLinearAlgebra::diagonal(Matrix const &m ALIEN_UNUSED_PARAM, Vector &v
                                         ALIEN_UNUSED_PARAM) const {
      throw Arccore::NotImplementedException(
              A_FUNCINFO, "HypreLinearAlgebra::diagonal not implemented");
    }

    void
    HypreInternalLinearAlgebra::reciprocal(Vector &v ALIEN_UNUSED_PARAM) const {
      throw Arccore::NotImplementedException(
              A_FUNCINFO, "HypreLinearAlgebra::reciprocal not implemented");
    }

    void
    HypreInternalLinearAlgebra::aypx(
            const double &alpha ALIEN_UNUSED_PARAM, Vector &y ALIEN_UNUSED_PARAM, const Vector &x
            ALIEN_UNUSED_PARAM) const {
      throw Arccore::NotImplementedException(A_FUNCINFO, "HypreLinearAlgebra::aypx not implemented");
    }

    void
    HypreInternalLinearAlgebra::pointwiseMult(
            const Vector &x ALIEN_UNUSED_PARAM, const Vector &y ALIEN_UNUSED_PARAM, Vector &w
            ALIEN_UNUSED_PARAM) const {
      throw Arccore::NotImplementedException(
              A_FUNCINFO, "HypreLinearAlgebra::pointwiseMult not implemented");
    }

    void
    HypreInternalLinearAlgebra::scal(const Arccore::Real &alpha, Vector &x) const {
      HYPRE_ParVectorScale(static_cast<double>(alpha), hypre_implem(x));
    }
  }
}