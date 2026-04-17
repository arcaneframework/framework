// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedRelaxationRuntime.h                              (C) 2000-2026 */
/*                                                                           */
/* Distributed memory sparse approximate inverse relaxation scheme.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MPI_DISTRIBUTEDRELAXATIONRUNTIME_H
#define ARCCORE_ALINA_MPI_DISTRIBUTEDRELAXATIONRUNTIME_H
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

#include "arccore/alina/BackendInterface.h"
#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/DistributedRelaxation.h"
#include "arccore/alina/DistributedMatrix.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Distributed memory sparse approximate inverse relaxation scheme.
 */
template <class Backend>
struct DistributedRelaxationRuntime
{
  typedef Backend backend_type;
  using BackendType = backend_type;
  typedef typename Backend::params backend_params;
  typedef Alina::PropertyTree params;

  eRelaxationType r;
  void* handle = nullptr;

  DistributedRelaxationRuntime(const DistributedMatrix<Backend>& A,
                               params prm, const backend_params& bprm = backend_params())
  : r(prm.get("type", eRelaxationType::spai0))
  {
    if (!prm.erase("type"))
      ARCCORE_ALINA_PARAM_MISSING("type");

    switch (r) {

#define ARCCORE_ALINA_RELAX_DISTR(type) \
  case eRelaxationType::type: \
    handle = static_cast<void*>(new ::Arcane::Alina::Distributed##type<Backend>(A, prm, bprm)); \
    break

#define ARCCORE_ALINA_RELAX_LOCAL_DISTR(type) \
  case eRelaxationType::type: \
    handle = call_constructor<type>(A, prm, bprm); \
    break;

#define ARCCORE_ALINA_RELAX_LOCAL_LOCAL(type) \
  case eRelaxationType::type: \
    handle = call_constructor<type>(*A.local(), prm, bprm); \
    break;

      ARCCORE_ALINA_RELAX_DISTR(SPAI0Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ChebyshevRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_LOCAL(DampedJacobiRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_LOCAL(ILU0Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_LOCAL(ILUKRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_LOCAL(ILUPRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_LOCAL(ILUTRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_LOCAL(SPAI1Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_LOCAL(GaussSeidelRelaxation);

#undef ARCCORE_ALINA_RELAX_LOCAL_LOCAL
#undef ARCCORE_ALINA_RELAX_LOCAL_DISTR
#undef ARCCORE_ALINA_RELAX_DISTR

    default:
      throw std::invalid_argument("Unsupported relaxation type");
    }
  }

  ~DistributedRelaxationRuntime()
  {
    switch (r) {
#define ARCCORE_ALINA_RELAX_DISTR(type) \
  case eRelaxationType::type: \
    delete static_cast<Distributed##type<Backend>*>(handle); \
    break

#define ARCCORE_ALINA_RELAX_LOCAL(type) \
  case eRelaxationType::type: \
    delete static_cast<type<Backend>*>(handle); \
    break;

      ARCCORE_ALINA_RELAX_DISTR(SPAI0Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL(DampedJacobiRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL(ILU0Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL(ILUKRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL(ILUPRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL(ILUTRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL(SPAI1Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL(ChebyshevRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL(GaussSeidelRelaxation);

#undef ARCCORE_ALINA_RELAX_LOCAL
#undef ARCCORE_ALINA_RELAX_DISTR

    default:
      break;
    }
  }

  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    switch (r) {

#define ARCCORE_ALINA_RELAX_DISTR(type) \
  case eRelaxationType::type: \
    static_cast<const Distributed##type<Backend>*>(handle)->apply_pre(A, rhs, x, tmp); \
    break

#define ARCCORE_ALINA_RELAX_LOCAL_DISTR(type) \
  case eRelaxationType::type: \
    call_apply_pre<type>(A, rhs, x, tmp); \
    break;

#define ARCCORE_ALINA_RELAX_LOCAL_LOCAL(type) \
  case eRelaxationType::type: \
    call_apply_pre<type>(*A.local_backend(), rhs, x, tmp); \
    break;

      ARCCORE_ALINA_RELAX_DISTR(SPAI0Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(DampedJacobiRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILU0Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILUKRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILUPRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILUTRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(SPAI1Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ChebyshevRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_LOCAL(GaussSeidelRelaxation);

#undef ARCCORE_ALINA_RELAX_LOCAL_LOCAL
#undef ARCCORE_ALINA_RELAX_LOCAL_DISTR
#undef ARCCORE_ALINA_RELAX_DISTR

    default:
      throw std::invalid_argument("Unsupported relaxation type");
    }
  }

  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    switch (r) {

#define ARCCORE_ALINA_RELAX_DISTR(type) \
  case eRelaxationType::type: \
    static_cast<const ::Arcane::Alina::Distributed##type<Backend>*>(handle)->apply_post(A, rhs, x, tmp); \
    break

#define ARCCORE_ALINA_RELAX_LOCAL_DISTR(type) \
  case eRelaxationType::type: \
    call_apply_post<type>(A, rhs, x, tmp); \
    break;

#define ARCCORE_ALINA_RELAX_LOCAL_LOCAL(type) \
  case eRelaxationType::type: \
    call_apply_post<type>(*A.local_backend(), rhs, x, tmp); \
    break;

      ARCCORE_ALINA_RELAX_DISTR(SPAI0Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(DampedJacobiRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILU0Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILUKRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILUPRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILUTRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(SPAI1Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ChebyshevRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_LOCAL(GaussSeidelRelaxation);

#undef ARCCORE_ALINA_RELAX_LOCAL_LOCAL
#undef ARCCORE_ALINA_RELAX_LOCAL_DISTR
#undef ARCCORE_ALINA_RELAX_DISTR

    default:
      throw std::invalid_argument("Unsupported relaxation type");
    }
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix& A, const VectorRHS& rhs, VectorX& x) const
  {
    switch (r) {

#define ARCCORE_ALINA_RELAX_DISTR(type) \
  case eRelaxationType::type: \
    static_cast<const Distributed##type<Backend>*>(handle)->apply(A, rhs, x); \
    break

#define ARCCORE_ALINA_RELAX_LOCAL_DISTR(type) \
  case eRelaxationType::type: \
    call_apply<type>(A, rhs, x); \
    break;

#define ARCCORE_ALINA_RELAX_LOCAL_LOCAL(type) \
  case eRelaxationType::type: \
    call_apply<type>(*A.local_backend(), rhs, x); \
    break;

      ARCCORE_ALINA_RELAX_DISTR(SPAI0Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(DampedJacobiRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_LOCAL(GaussSeidelRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILU0Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILUKRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILUPRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ILUTRelaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(SPAI1Relaxation);
      ARCCORE_ALINA_RELAX_LOCAL_DISTR(ChebyshevRelaxation);

#undef ARCCORE_ALINA_RELAX_LOCAL_LOCAL
#undef ARCCORE_ALINA_RELAX_LOCAL_DISTR
#undef ARCCORE_ALINA_RELAX_DISTR

    default:
      throw std::invalid_argument("Unsupported relaxation type");
    }
  }

  template <template <class> class Relaxation, class Matrix>
  typename std::enable_if<backend::relaxation_is_supported<Backend, Relaxation>::value, void*>::type
  call_constructor(const Matrix& A, const params& prm, const backend_params& bprm)
  {
    return static_cast<void*>(new Relaxation<Backend>(A, prm, bprm));
  }

  template <template <class> class Relaxation, class Matrix>
  typename std::enable_if<!backend::relaxation_is_supported<Backend, Relaxation>::value, void*>::type
  call_constructor(const Matrix&, const params&, const backend_params&)
  {
    throw std::logic_error("The relaxation is not supported by the backend");
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  typename std::enable_if<backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    static_cast<Relaxation<Backend>*>(handle)->apply_pre(A, rhs, x, tmp);
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  typename std::enable_if<!backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply_pre(const Matrix&, const VectorRHS&, VectorX&, VectorTMP&) const
  {
    throw std::logic_error("The relaxation is not supported by the backend");
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  typename std::enable_if<backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    static_cast<Relaxation<Backend>*>(handle)->apply_post(A, rhs, x, tmp);
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  typename std::enable_if<!backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply_post(const Matrix&, const VectorRHS&, VectorX&, VectorTMP&) const
  {
    throw std::logic_error("The relaxation is not supported by the backend");
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX>
  typename std::enable_if<backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply(const Matrix& A, const VectorRHS& rhs, VectorX& x) const
  {
    static_cast<Relaxation<Backend>*>(handle)->apply(A, rhs, x);
  }

  template <template <class> class Relaxation, class Matrix, class VectorRHS, class VectorX>
  typename std::enable_if<!backend::relaxation_is_supported<Backend, Relaxation>::value, void>::type
  call_apply(const Matrix&, const VectorRHS&, VectorX&) const
  {
    throw std::logic_error("The relaxation is not supported by the backend");
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
