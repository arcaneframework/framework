// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableComputeFunction.h                                   (C) 2000-2013 */
/*                                                                           */
/* Classe fonctor de recalcul d'une variable.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLECOMPUTEFUNCTION_H
#define ARCANE_VARIABLECOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Functor.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/IVariableComputeFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 \brief Interface de la classe fonctor de recalcul d'une variable.
 */
class VariableComputeFunction
: public IVariableComputeFunction
{
 public:

  template<typename ClassType>
  VariableComputeFunction(ClassType* instance,void (ClassType::*func)())
  : m_functor(new FunctorT<ClassType>(instance,func))
    {
    }
  template<typename ClassType>
  VariableComputeFunction(ClassType* instance,void (ClassType::*func)(),const TraceInfo& tinfo)
  : m_functor(new FunctorT<ClassType>(instance,func)), m_trace_info(tinfo)
    {
    }

  //! Libère les ressources
  virtual ~VariableComputeFunction()
  {
    delete m_functor;
  }

 public:

  //! Exécute la fonction de calcul
  virtual void execute()
  {
    m_functor->executeFunctor();
  }

  virtual const TraceInfo& traceInfo() const
  {
    return m_trace_info;
  }

 private:

  IFunctor* m_functor;
  TraceInfo m_trace_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

