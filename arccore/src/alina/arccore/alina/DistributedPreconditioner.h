// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedPreconditioner.h                                 (C) 2000-2026 */
/*                                                                           */
/* Runtime wrapper around mpi preconditioners.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MPI_DISTRIBUTEDPRECONDITIONER_H
#define ARCCORE_ALINA_MPI_DISTRIBUTEDPRECONDITIONER_H
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

#include <iostream>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/DistributedAMG.h"
#include "arccore/alina/DistributedInnerProduct.h"
#include "arccore/alina/DistributedCoarseningRuntime.h"
#include "arccore/alina/DistributedRelaxationRuntime.h"
#include "arccore/alina/DistributedDirectSolverRuntime.h"
#include "arccore/alina/MatrixPartitionerRuntime.h"
#include "arccore/alina/DistributedRelaxation.h"
#include "arccore/alina/DistributedMatrix.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Preconditioner kinds.
enum class eDistributedPreconditionerType
{
  amg, ///< AMG
  relaxation ///< Single-level relaxation
};

inline std::ostream& operator<<(std::ostream& os, eDistributedPreconditionerType p)
{
  switch (p) {
  case eDistributedPreconditionerType::amg:
    return os << "amg";
  case eDistributedPreconditionerType::relaxation:
    return os << "relaxation";
  default:
    return os << "???";
  }
}

inline std::istream& operator>>(std::istream& in, eDistributedPreconditionerType& p)
{
  std::string val;
  in >> val;

  if (val == "amg")
    p = eDistributedPreconditionerType::amg;
  else if (val == "relaxation")
    p = eDistributedPreconditionerType::relaxation;
  else
    throw std::invalid_argument("Invalid preconditioner class. "
                                "Valid choices are: amg, relaxation");

  return in;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Distributed Preconditioner.
 */
template <class Backend>
class DistributedPreconditioner
{
 public:

  using backend_type = Backend;
  using BackendType = Backend;
  typedef typename backend_type::params backend_params;
  typedef PropertyTree params;
  typedef typename backend_type::value_type value_type;
  typedef DistributedMatrix<backend_type> matrix;

  using AMGPrecondType = DistributedAMG<Backend,
                                        DistributedCoarseningRuntime<Backend>,
                                        DistributedRelaxationRuntime<Backend>,
                                        DistributedDirectSolverRuntime<Backend>,
                                        MatrixPartitionerRuntime<Backend>>;

  template <class Matrix>
  DistributedPreconditioner(mpi_communicator comm,
                            const Matrix& Astrip,
                            params prm = params(),
                            const backend_params& bprm = backend_params())
  : _class(prm.get("class", eDistributedPreconditionerType::amg))
  , handle(0)
  {
    init(std::make_shared<matrix>(comm, Astrip, backend::nbRow(Astrip)), prm, bprm);
  }

  DistributedPreconditioner(mpi_communicator,
                            std::shared_ptr<matrix> A,
                            params prm = params(),
                            const backend_params& bprm = backend_params())
  : _class(prm.get("class", eDistributedPreconditionerType::amg))
  , handle(0)
  {
    init(A, prm, bprm);
  }

  ~DistributedPreconditioner()
  {
    switch (_class) {
    case eDistributedPreconditionerType::amg: {
      delete static_cast<AMGPrecondType*>(handle);
    } break;
    case eDistributedPreconditionerType::relaxation: {
      typedef Alina::AsDistributedPreconditioner<DistributedRelaxationRuntime<Backend>>
      Precond;

      delete static_cast<Precond*>(handle);
    } break;
    default:
      break;
    }
  }

  template <class Matrix>
  void rebuild(const Matrix& A,
               const backend_params& bprm = backend_params())
  {
    switch (_class) {
    case eDistributedPreconditionerType::amg: {
      static_cast<AMGPrecondType*>(handle)->rebuild(A, bprm);
    } break;
    default:
      std::cerr << "rebuild is a noop unless the preconditioner is AMG" << std::endl;
      return;
    }
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    switch (_class) {
    case eDistributedPreconditionerType::amg: {
      static_cast<AMGPrecondType*>(handle)->apply(rhs, x);
    } break;
    case eDistributedPreconditionerType::relaxation: {
      typedef Alina::AsDistributedPreconditioner<DistributedRelaxationRuntime<Backend>> Precond;

      static_cast<Precond*>(handle)->apply(rhs, x);
    } break;
    default:
      throw std::invalid_argument("Unsupported preconditioner class");
    }
  }

  /// Returns the system matrix from the finest level.
  std::shared_ptr<matrix> system_matrix_ptr() const
  {
    switch (_class) {
    case eDistributedPreconditionerType::amg: {
      return static_cast<AMGPrecondType*>(handle)->system_matrix_ptr();
    }
    case eDistributedPreconditionerType::relaxation: {
      typedef AsDistributedPreconditioner<DistributedRelaxationRuntime<Backend>> Precond;

      return static_cast<Precond*>(handle)->system_matrix_ptr();
    }
    default:
      throw std::invalid_argument("Unsupported preconditioner class");
    }
  }

  const matrix& system_matrix() const
  {
    return *system_matrix_ptr();
  }

  friend std::ostream& operator<<(std::ostream& os, const DistributedPreconditioner& p)
  {
    switch (p._class) {
    case eDistributedPreconditionerType::amg: {
      return os << *static_cast<AMGPrecondType*>(p.handle);
    }
    case eDistributedPreconditionerType::relaxation: {
      typedef AsDistributedPreconditioner<DistributedRelaxationRuntime<Backend>> Precond;

      return os << *static_cast<Precond*>(p.handle);
    }
    default:
      throw std::invalid_argument("Unsupported preconditioner class");
    }
  }

 private:

  eDistributedPreconditionerType _class;
  void* handle;

  void init(std::shared_ptr<matrix> A, params& prm, const backend_params& bprm)
  {
    if (!prm.erase("class"))
      ARCCORE_ALINA_PARAM_MISSING("class");

    switch (_class) {
    case eDistributedPreconditionerType::amg: {
      handle = static_cast<void*>(new AMGPrecondType(A->comm(), A, prm, bprm));
    } break;
    case eDistributedPreconditionerType::relaxation: {
      typedef AsDistributedPreconditioner<DistributedRelaxationRuntime<Backend>> Precond;

      handle = static_cast<void*>(new Precond(A->comm(), A, prm, bprm));
    } break;
    default:
      throw std::invalid_argument("Unsupported preconditioner class");
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Distributed block preconditioner.
 */
template <class Precond>
class DistributedBlockPreconditioner
{
 public:

  typedef typename Precond::params params;
  typedef typename Precond::backend_type backend_type;
  using BackendType = backend_type;
  typedef typename backend_type::params backend_params;

  typedef typename backend_type::value_type value_type;
  typedef typename backend_type::matrix bmatrix;
  typedef DistributedMatrix<backend_type> matrix;

  template <class Matrix>
  DistributedBlockPreconditioner(mpi_communicator comm,
                                 const Matrix& Astrip,
                                 const params& prm = params(),
                                 const backend_params& bprm = backend_params())
  {
    A = std::make_shared<matrix>(comm, Astrip, backend::nbRow(Astrip));
    P = std::make_shared<Precond>(A->local(), prm, bprm);
    A->set_local(P->system_matrix_ptr());
    A->move_to_backend(bprm);
  }

  DistributedBlockPreconditioner(mpi_communicator,
                                 std::shared_ptr<matrix> A,
                                 const params& prm = params(),
                                 const backend_params& bprm = backend_params())
  : A(A)
  {
    P = std::make_shared<Precond>(A->local(), prm, bprm);
    A->set_local(P->system_matrix_ptr());
    A->move_to_backend(bprm);
  }

  std::shared_ptr<matrix> system_matrix_ptr() const
  {
    return A;
  }

  const matrix& system_matrix() const
  {
    return *A;
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    P->apply(rhs, x);
  }

 private:

  std::shared_ptr<matrix> A;
  std::shared_ptr<Precond> P;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
