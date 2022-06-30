// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableComputeFunction.h                       (C) 2000-2022 */
/*                                                                           */
/* Classe fonctor de recalcul d'une variable matériau.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLECOMPUTEFUNCTION_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLECOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FunctorWithArgument.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/core/materials/IMeshMaterialVariableComputeFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 \brief Interface de la classe fonctor de recalcul d'une variable.
 */
class MeshMaterialVariableComputeFunction
: public IMeshMaterialVariableComputeFunction
{
 public:

  template<typename ClassType>
  MeshMaterialVariableComputeFunction(ClassType* instance,void (ClassType::*func)(IMeshMaterial* mat))
  : m_functor(new FunctorWithArgumentT<ClassType,IMeshMaterial*>(instance,func))
  {
  }
  template<typename ClassType>
  MeshMaterialVariableComputeFunction(ClassType* instance,void (ClassType::*func)(IMeshMaterial* mat),const TraceInfo& tinfo)
  : m_functor(new FunctorWithArgumentT<ClassType,IMeshMaterial*>(instance,func)), m_trace_info(tinfo)
  {
  }

  //! Libère les ressources
  ~MeshMaterialVariableComputeFunction() override
  {
    delete m_functor;
  }

 public:

  //! Exécute la fonction de calcul
  void execute(IMeshMaterial* mat) override
  {
    m_functor->executeFunctor(mat);
  }

  const TraceInfo& traceInfo() const override
  {
    return m_trace_info;
  }

 private:

  IFunctorWithArgumentT<IMeshMaterial*>* m_functor;
  TraceInfo m_trace_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

