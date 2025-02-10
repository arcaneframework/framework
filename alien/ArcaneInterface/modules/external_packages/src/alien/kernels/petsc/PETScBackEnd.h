// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MTLBackEnd                                                  (C) 2000-2025 */
/*                                                                           */
/* Tools for PETSc backend                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ALIEN_PETSCIMPL_PETSCBACKEND_H
#define ALIEN_PETSCIMPL_PETSCBACKEND_H

#include <arccore/message_passing/MessagePassingGlobal.h>
#include <alien/core/backend/BackEnd.h>
#include <alien/utils/Precomp.h>
#include <alien/AlienExternalPackagesExport.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IOptionsPETScLinearSolver;

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PETScInternalLinearSolver;
class PETScMatrix;
class PETScVector;
class Space;
template <class Matrix, class Vector> class IInternalLinearAlgebra;
template <class Matrix, class Vector> class IInternalLinearSolver;

extern ALIEN_EXTERNAL_PACKAGES_EXPORT IInternalLinearAlgebra<PETScMatrix, PETScVector>*
PETScInternalLinearAlgebraFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng = nullptr);

extern IInternalLinearSolver<PETScMatrix, PETScVector>* PETScInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng,
    IOptionsPETScLinearSolver* options);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace BackEnd {
  namespace tag {
    struct petsc
    {
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <> struct AlgebraTraits<BackEnd::tag::petsc>
{
  typedef PETScMatrix matrix_type;
  typedef PETScVector vector_type;
  typedef IOptionsPETScLinearSolver options_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
  typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;
  static algebra_type* algebra_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng = nullptr)
  {
    return PETScInternalLinearAlgebraFactory(p_mng);
  }
  static solver_type* solver_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng, options_type* options)
  {
    return PETScInternalLinearSolverFactory(p_mng, options);
  }
  static BackEndId name() { return "petsc"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_PETSCIMPL_PETSCBACKEND_H */
