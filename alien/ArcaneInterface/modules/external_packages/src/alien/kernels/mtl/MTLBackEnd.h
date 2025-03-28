// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MTLBackEnd                                                  (C) 2000-2025 */
/*                                                                           */
/* Tools for MTL backend                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ALIEN_MTLIMPL_MTLBACKEND_H
#define ALIEN_MTLIMPL_MTLBACKEND_H

#include <arccore/message_passing/MessagePassingGlobal.h>
#include <alien/core/backend/BackEnd.h>
#include <alien/utils/Precomp.h>
#include <alien/AlienExternalPackagesExport.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class IOptionsMTLLinearSolver;

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MultiVectorImpl;
class MTLMatrix;
class MTLVector;
class Space;
template <class Matrix, class Vector> class IInternalLinearAlgebra;
template <class Matrix, class Vector> class IInternalLinearSolver;

extern ALIEN_EXTERNAL_PACKAGES_EXPORT IInternalLinearAlgebra<MTLMatrix, MTLVector>* MTLInternalLinearAlgebraFactory();

extern IInternalLinearSolver<MTLMatrix, MTLVector>* MTLInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsMTLLinearSolver* options);

/*---------------------------------------------------------------------------*/

namespace BackEnd {
  namespace tag {
    struct mtl
    {
    };
  }
}

template <> struct AlgebraTraits<BackEnd::tag::mtl>
{
  typedef MTLMatrix matrix_type;
  typedef MTLVector vector_type;
  typedef IOptionsMTLLinearSolver options_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
  typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;
  static algebra_type* algebra_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng = nullptr)
  {
    return MTLInternalLinearAlgebraFactory();
  }
  static solver_type* solver_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng, options_type* options)
  {
    return MTLInternalLinearSolverFactory(p_mng, options);
  }
  static BackEndId name() { return "mtl"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_MTLIMPL_MTLBACKEND_H */
