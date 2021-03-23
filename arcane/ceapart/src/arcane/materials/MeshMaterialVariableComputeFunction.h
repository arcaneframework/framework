// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableComputeFunction.h                       (C) 2000-2014 */
/*                                                                           */
/* Classe fonctor de recalcul d'une variable matériau.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLECOMPUTEFUNCTION_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLECOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FunctorWithArgument.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/materials/IMeshMaterialVariableComputeFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

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
  virtual ~MeshMaterialVariableComputeFunction()
  {
    delete m_functor;
  }

 public:

  //! Exécute la fonction de calcul
  virtual void execute(IMeshMaterial* mat)
  {
    m_functor->executeFunctor(mat);
  }

  virtual const TraceInfo& traceInfo() const
  {
    return m_trace_info;
  }

 private:

  IFunctorWithArgumentT<IMeshMaterial*>* m_functor;
  TraceInfo m_trace_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

