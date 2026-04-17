// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RelaxationRuntime.h                                         (C) 2000-2026 */
/*                                                                           */
/* Runtime configurable relaxation.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_RELAXATIONRUNTIME_H
#define ARCCORE_ALINA_RELAXATIONRUNTIME_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/AlinaGlobal.h"

#include <type_traits>

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/Relaxation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Relaxation schemes.
enum class eRelaxationType
{
  gauss_seidel, ///< Gauss-Seidel smoothing
  GaussSeidelRelaxation = gauss_seidel,
  ilu0, ///< Incomplete LU with zero fill-in
  ILU0Relaxation = ilu0,
  iluk, ///< Level-based incomplete LU
  ILUKRelaxation = iluk,
  ilup, ///< Level-based incomplete LU (fill-in is determined from A^p pattern)
  ILUPRelaxation = ilup,
  ilut, ///< Incomplete LU with thresholding
  ILUTRelaxation = ilut,
  damped_jacobi, ///< Damped Jacobi
  DampedJacobiRelaxation = damped_jacobi,
  spai0, ///< Sparse approximate inverse of 0th order
  SPAI0Relaxation = spai0,
  spai1, ///< Sparse approximate inverse of 1st order
  SPAI1Relaxation = spai1,
  chebyshev, ///< Chebyshev relaxation
  ChebyshevRelaxation = chebyshev
};

extern "C++" ARCCORE_ALINA_EXPORT
std::ostream& operator<<(std::ostream& os, eRelaxationType r);

extern "C++" ARCCORE_ALINA_EXPORT
std::istream& operator>>(std::istream& in, eRelaxationType& r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_ALINA_ALL_RUNTIME_RELAXATION() \
  ARCCORE_ALINA_RUNTIME_RELAXATION(GaussSeidelRelaxation); \
  ARCCORE_ALINA_RUNTIME_RELAXATION(ILU0Relaxation); \
  ARCCORE_ALINA_RUNTIME_RELAXATION(ILUKRelaxation); \
  ARCCORE_ALINA_RUNTIME_RELAXATION(ILUPRelaxation); \
  ARCCORE_ALINA_RUNTIME_RELAXATION(ILUTRelaxation); \
  ARCCORE_ALINA_RUNTIME_RELAXATION(DampedJacobiRelaxation); \
  ARCCORE_ALINA_RUNTIME_RELAXATION(SPAI0Relaxation); \
  ARCCORE_ALINA_RUNTIME_RELAXATION(SPAI1Relaxation); \
  ARCCORE_ALINA_RUNTIME_RELAXATION(ChebyshevRelaxation)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Runtime configurable relaxation.
 */
template <class Backend>
struct RelaxationRuntime
{
  typedef Alina::PropertyTree params;
  typedef typename Backend::params backend_params;

  template <class Matrix>
  explicit RelaxationRuntime(const Matrix& A, params prm = params(),
                             const backend_params& bprm = backend_params())
  : m_relaxation_type(prm.get("type", eRelaxationType::spai0))
  {
    if (!prm.erase("type"))
      ARCCORE_ALINA_PARAM_MISSING("type");
    switch (m_relaxation_type) {

#define ARCCORE_ALINA_RUNTIME_RELAXATION(type) \
  case eRelaxationType::type: \
    m_relaxation = call_constructor<type>(A, prm, bprm); \
    break

      ARCCORE_ALINA_ALL_RUNTIME_RELAXATION();

#undef ARCCORE_ALINA_RUNTIME_RELAXATION

    default:
      _throwBadTypeType();
    }
  }

  ~RelaxationRuntime()
  {
    delete m_relaxation;
  }

  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    switch (m_relaxation_type) {

#define ARCCORE_ALINA_RUNTIME_RELAXATION(type) \
  case eRelaxationType::type: \
    call_apply_pre<type>(A, rhs, x, tmp); \
    break

      ARCCORE_ALINA_ALL_RUNTIME_RELAXATION();

#undef ARCCORE_ALINA_RUNTIME_RELAXATION

    default:
      _throwBadTypeType();
    }
  }

  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    switch (m_relaxation_type) {

#define ARCCORE_ALINA_RUNTIME_RELAXATION(type) \
  case eRelaxationType::type: \
    call_apply_post<type>(A, rhs, x, tmp); \
    break

      ARCCORE_ALINA_ALL_RUNTIME_RELAXATION();

#undef ARCCORE_ALINA_RUNTIME_RELAXATION

    default:
      _throwBadTypeType();
    }
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix& A, const VectorRHS& rhs, VectorX& x) const
  {
    switch (m_relaxation_type) {

#define ARCCORE_ALINA_RUNTIME_RELAXATION(type) \
  case eRelaxationType::type: \
    call_apply<type>(A, rhs, x); \
    break

      ARCCORE_ALINA_ALL_RUNTIME_RELAXATION();

#undef ARCCORE_ALINA_RUNTIME_RELAXATION

    default:
      _throwBadTypeType();
    }
  }

  size_t bytes() const
  {
    return m_relaxation->bytes();
  }

  template <template <class> class Relaxation, class Matrix>
  typename std::enable_if_t<backend::relaxation_is_supported<Backend, Relaxation>::value, RelaxationBase*>
  call_constructor(const Matrix& A, const params& prm, const backend_params& bprm)
  {
    return new Relaxation<Backend>(A, prm, bprm);
  }

  template <template <class> class Relaxation, class Matrix>
  typename std::enable_if_t<!backend::relaxation_is_supported<Backend, Relaxation>::value, RelaxationBase*>
  call_constructor(const Matrix&, const params&, const backend_params&)
  {
    _throwUnsupportedBackendType();
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  typename std::enable_if<backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    static_cast<Relaxation<Backend>*>(m_relaxation)->apply_pre(A, rhs, x, tmp);
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  typename std::enable_if<!backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply_pre(const Matrix&, const VectorRHS&, VectorX&, VectorTMP&) const
  {
    _throwUnsupportedBackendType();
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  typename std::enable_if<backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    static_cast<Relaxation<Backend>*>(m_relaxation)->apply_post(A, rhs, x, tmp);
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  typename std::enable_if<!backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply_post(const Matrix&, const VectorRHS&, VectorX&, VectorTMP&) const
  {
    _throwUnsupportedBackendType();
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX>
  typename std::enable_if<backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply(const Matrix& A, const VectorRHS& rhs, VectorX& x) const
  {
    static_cast<Relaxation<Backend>*>(m_relaxation)->apply(A, rhs, x);
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX>
  typename std::enable_if<!backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply(const Matrix&, const VectorRHS&, VectorX&) const
  {
    _throwUnsupportedBackendType();
  }

  void _throwBadTypeType [[noreturn]] () const
  {
    int v = static_cast<int>(m_relaxation_type);
    ARCANE_FATAL("Unsupported relaxation type '{0}'", v);
  }
  void _throwUnsupportedBackendType [[noreturn]] () const
  {
    String err_message = String::format("The relaxation '{0}' is not supported by the backend", m_relaxation_type);
    //NOTE: We need to do a 'logic_error' because this is catched is some tests
    throw std::logic_error(err_message.localstr());
  }

 private:

  eRelaxationType m_relaxation_type = eRelaxationType::SPAI0Relaxation;
  RelaxationBase* m_relaxation = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
