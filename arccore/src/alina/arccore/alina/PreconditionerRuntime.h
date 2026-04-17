// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PreconditionerRuntime.h                                     (C) 2000-2026 */
/*                                                                           */
/* Runtime-configurable preconditioners.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_PRECONDITIONERRUNTIME_H
#define ARCCORE_ALINA_PRECONDITIONERRUNTIME_H
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

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/SolverRuntime.h"
#include "arccore/alina/CoarseningRuntime.h"
#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/Relaxation.h"
#include "arccore/alina/DummyPreconditioner.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/AMG.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Preconditioner kinds.
enum class ePreconditionerType
{
  amg, ///< AMG
  relaxation, ///< Single-level relaxation
  dummy, ///< Identity matrix as preconditioner.
  nested ///< Nested solver as preconditioner.
};

inline std::ostream& operator<<(std::ostream& os, ePreconditionerType p)
{
  switch (p) {
  case ePreconditionerType::amg:
    return os << "amg";
  case ePreconditionerType::relaxation:
    return os << "relaxation";
  case ePreconditionerType::dummy:
    return os << "dummy";
  case ePreconditionerType::nested:
    return os << "nested";
  default:
    return os << "???";
  }
}

inline std::istream& operator>>(std::istream& in, ePreconditionerType& p)
{
  std::string val;
  in >> val;

  if (val == "amg")
    p = ePreconditionerType::amg;
  else if (val == "relaxation")
    p = ePreconditionerType::relaxation;
  else if (val == "dummy")
    p = ePreconditionerType::dummy;
  else if (val == "nested")
    p = ePreconditionerType::nested;
  else
    throw std::invalid_argument("Invalid preconditioner class. Valid choices are: "
                                "amg, relaxation, dummy, nested");

  return in;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Runtime-configurable preconditioners.
 */
template <class Backend>
class PreconditionerRuntime
{
 public:

  using backend_type = Backend;
  using BackendType = Backend;

  typedef typename Backend::value_type value_type;
  typedef typename Backend::matrix matrix;
  typedef typename Backend::vector vector;
  typedef typename Backend::params backend_params;

  typedef Alina::PropertyTree params;

  template <class Matrix>
  PreconditionerRuntime(const Matrix& A,
                        params prm = params(),
                        const backend_params& bprm = backend_params())
  : _class(prm.get("class", ePreconditionerType::amg))
  , handle(0)
  {
    if (!prm.erase("class"))
      ARCCORE_ALINA_PARAM_MISSING("class");
    std::cout << "PreconditionerClass=" << _class << "\n";
    switch (_class) {
    case ePreconditionerType::amg: {
      typedef Alina::AMG<Backend, CoarseningRuntime, RelaxationRuntime> Precond;
      handle = static_cast<void*>(new Precond(A, prm, bprm));
    } break;
    case ePreconditionerType::relaxation: {
      typedef Alina::RelaxationAsPreconditioner<Backend, RelaxationRuntime> Precond;
      handle = static_cast<void*>(new Precond(A, prm, bprm));
    } break;
    case ePreconditionerType::dummy: {
      typedef Alina::preconditioner::DummyPreconditioner<Backend> Precond;
      handle = static_cast<void*>(new Precond(A, prm, bprm));
    } break;
    case ePreconditionerType::nested: {
      typedef PreconditionedSolver<PreconditionerRuntime, SolverRuntime<Backend>> Precond;
      handle = static_cast<void*>(new Precond(A, prm, bprm));
    } break;
    default:
      throw std::invalid_argument("Unsupported preconditioner class");
    }
  }

  ~PreconditionerRuntime()
  {
    switch (_class) {
    case ePreconditionerType::amg: {
      typedef Alina::AMG<Backend, CoarseningRuntime, RelaxationRuntime> Precond;
      delete static_cast<Precond*>(handle);
    } break;
    case ePreconditionerType::relaxation: {
      typedef Alina::RelaxationAsPreconditioner<Backend, RelaxationRuntime> Precond;
      delete static_cast<Precond*>(handle);
    } break;
    case ePreconditionerType::dummy: {
      typedef Alina::preconditioner::DummyPreconditioner<Backend> Precond;
      delete static_cast<Precond*>(handle);
    } break;
    case ePreconditionerType::nested: {
      typedef PreconditionedSolver<PreconditionerRuntime, SolverRuntime<Backend>> Precond;
      delete static_cast<Precond*>(handle);
    } break;
    default:
      break;
    }
  }

  template <class Matrix>
  void rebuild(const Matrix& A, const backend_params& bprm = backend_params())
  {
    switch (_class) {
    case ePreconditionerType::amg: {
      typedef Alina::AMG<Backend, CoarseningRuntime, RelaxationRuntime> Precond;
      static_cast<Precond*>(handle)->rebuild(A, bprm);
    } break;
    default:
      std::cerr << "rebuild is a noop unless the preconditioner is AMG" << std::endl;
      return;
    }
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2& x) const
  {
    switch (_class) {
    case ePreconditionerType::amg: {
      typedef Alina::AMG<Backend, CoarseningRuntime, RelaxationRuntime> Precond;
      static_cast<Precond*>(handle)->apply(rhs, x);
    } break;
    case ePreconditionerType::relaxation: {
      typedef Alina::RelaxationAsPreconditioner<Backend, RelaxationRuntime> Precond;
      static_cast<Precond*>(handle)->apply(rhs, x);
    } break;
    case ePreconditionerType::dummy: {
      typedef Alina::preconditioner::DummyPreconditioner<Backend> Precond;
      static_cast<Precond*>(handle)->apply(rhs, x);
    } break;
    case ePreconditionerType::nested: {
      typedef PreconditionedSolver<PreconditionerRuntime, SolverRuntime<Backend>> Precond;
      static_cast<Precond*>(handle)->apply(rhs, x);
    } break;
    default:
      throw std::invalid_argument("Unsupported preconditioner class");
    }
  }

  std::shared_ptr<matrix> system_matrix_ptr() const
  {
    switch (_class) {
    case ePreconditionerType::amg: {
      typedef Alina::AMG<Backend, CoarseningRuntime, RelaxationRuntime> Precond;
      return static_cast<Precond*>(handle)->system_matrix_ptr();
    }
    case ePreconditionerType::relaxation: {
      typedef Alina::RelaxationAsPreconditioner<Backend, RelaxationRuntime> Precond;
      return static_cast<Precond*>(handle)->system_matrix_ptr();
    }
    case ePreconditionerType::dummy: {
      typedef Alina::preconditioner::DummyPreconditioner<Backend> Precond;
      return static_cast<Precond*>(handle)->system_matrix_ptr();
    }
    case ePreconditionerType::nested: {
      typedef PreconditionedSolver<PreconditionerRuntime, SolverRuntime<Backend>> Precond;
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

  size_t size() const
  {
    return backend::nbRow(system_matrix());
  }

  size_t bytes() const
  {
    switch (_class) {
    case ePreconditionerType::amg: {
      typedef Alina::AMG<Backend, CoarseningRuntime, RelaxationRuntime> Precond;
      return backend::bytes(*static_cast<Precond*>(handle));
    }
    case ePreconditionerType::relaxation: {
      typedef Alina::RelaxationAsPreconditioner<Backend, RelaxationRuntime> Precond;
      return backend::bytes(*static_cast<Precond*>(handle));
    }
    case ePreconditionerType::dummy: {
      typedef Alina::preconditioner::DummyPreconditioner<Backend> Precond;
      return backend::bytes(*static_cast<Precond*>(handle));
    }
    case ePreconditionerType::nested: {
      typedef PreconditionedSolver<PreconditionerRuntime, SolverRuntime<Backend>> Precond;
      return backend::bytes(*static_cast<Precond*>(handle));
    }
    default:
      throw std::invalid_argument("Unsupported preconditioner class");
    }
  }

  friend std::ostream& operator<<(std::ostream& os, const PreconditionerRuntime& p)
  {
    switch (p._class) {
    case ePreconditionerType::amg: {
      typedef Alina::AMG<Backend, CoarseningRuntime, RelaxationRuntime> Precond;
      return os << *static_cast<Precond*>(p.handle);
    }
    case ePreconditionerType::relaxation: {
      typedef Alina::RelaxationAsPreconditioner<Backend, RelaxationRuntime> Precond;
      return os << *static_cast<Precond*>(p.handle);
    }
    case ePreconditionerType::dummy: {
      typedef Alina::preconditioner::DummyPreconditioner<Backend> Precond;
      return os << *static_cast<Precond*>(p.handle);
    }
    case ePreconditionerType::nested: {
      typedef PreconditionedSolver<PreconditionerRuntime, SolverRuntime<Backend>> Precond;
      return os << *static_cast<Precond*>(p.handle);
    }
    default:
      throw std::invalid_argument("Unsupported preconditioner class");
    }
  }

 private:

  const ePreconditionerType _class;

  void* handle = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
