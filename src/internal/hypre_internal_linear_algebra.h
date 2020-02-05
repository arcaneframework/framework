#pragma once

#include "hypre_backend.h"

#include <ALIEN/hypre/export.h>

#include <ALIEN/Utils/Precomp.h>
#include <ALIEN/Core/Backend/IInternalLinearAlgebraT.h>

namespace Alien::Hypre {

  class ALIEN_HYPRE_EXPORT InternalLinearAlgebra
          : public IInternalLinearAlgebra<Matrix, Vector> {
  public:

    InternalLinearAlgebra() {}

    virtual ~InternalLinearAlgebra() {}

  public:

    // IInternalLinearAlgebra interface.
    Arccore::Real norm0(const Vector &x) const;

    Arccore::Real norm1(const Vector &x) const;

    Arccore::Real norm2(const Vector &x) const;

    void mult(const Matrix &a, const Vector &x, Vector &r) const;

    void axpy(const Arccore::Real &alpha, const Vector &x, Vector &r) const;

    void aypx(const Arccore::Real &alpha, Vector &y, const Vector &x) const;

    void copy(const Vector &x, Vector &r) const;

    Arccore::Real dot(const Vector &x, const Vector &y) const;

    void scal(const Arccore::Real &alpha, Vector &x) const;

    void diagonal(const Matrix &a, Vector &x) const;

    void reciprocal(Vector &x) const;

    void pointwiseMult(const Vector &x, const Vector &y, Vector &w) const;
  };

}