// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DummyPreconditioner.h                                       (C) 2000-2026 */
/*                                                                           */
/* Dummy preconditioner using identity matrix.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_DUMMYPRECONDITIONER_H
#define ARCCORE_ALINA_DUMMYPRECONDITIONER_H
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

#include <memory>
#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/AlinaUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::preconditioner
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Dummy preconditioner using identity matrix.
 */
template <class Backend>
class DummyPreconditioner
{
 public:

  typedef Backend backend_type;

  typedef typename Backend::matrix matrix;
  typedef typename Backend::vector vector;
  typedef typename Backend::value_type value_type;
  typedef typename Backend::col_type col_type;
  typedef typename Backend::ptr_type ptr_type;
  typedef typename BuiltinBackend<value_type, col_type, ptr_type>::matrix build_matrix;

  typedef Alina::detail::empty_params params;
  typedef typename Backend::params backend_params;

  template <class Matrix>
  DummyPreconditioner(const Matrix& M,
                      const params& = params(),
                      const backend_params& bprm = backend_params())
  : A(Backend::copy_matrix(std::make_shared<build_matrix>(M), bprm))
  {
  }

  DummyPreconditioner(std::shared_ptr<build_matrix> M,
                      const params& = params(),
                      const backend_params& bprm = backend_params())
  : A(Backend::copy_matrix(M, bprm))
  {
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    backend::copy(rhs, x);
  }

  std::shared_ptr<matrix> system_matrix_ptr() const
  {
    return A;
  }

  const matrix& system_matrix() const
  {
    return *A;
  }

  size_t bytes() const
  {
    return 0;
  }

 private:

  std::shared_ptr<matrix> A;

  friend std::ostream& operator<<(std::ostream& os, const DummyPreconditioner& p)
  {
    os << "identity matrix as preconditioner" << std::endl;
    os << "  unknowns: " << backend::nbRow(p.system_matrix()) << std::endl;
    os << "  nonzeros: " << backend::nonzeros(p.system_matrix()) << std::endl;

    return os;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::preconditioner

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif
