// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RangeFunctor.h                                              (C) 2000-2010 */
/*                                                                           */
/* Fonctor sur un interval d'itération.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_RANGEFUNCTOR_H
#define ARCANE_UTILS_RANGEFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IRangeFunctor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor sur un interval d'itération.
 */
template<typename InstanceType>
class RangeFunctorT
: public IRangeFunctor
{
 private:

  typedef void (InstanceType::*FunctionType)(Integer i0,Integer size);

 public:
  RangeFunctorT(InstanceType* instance,FunctionType function)
  : m_instance(instance), m_function(function)
  {
  }
 
 public:
  
  virtual void executeFunctor(Integer begin,Integer size)
  {
    (m_instance->*m_function)(begin,size);
  }
 
 private:
  InstanceType* m_instance;
  FunctionType m_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor sur un interval d'itération instancié via une lambda fonction.
 *
 * Cette classe est utilisée avec le mécanisme des lambda fonctions du C++1x.
 */
template<typename LambdaType>
class LambdaRangeFunctorT
: public IRangeFunctor
{
 public:
  LambdaRangeFunctorT(const LambdaType& lambda_function)
  : m_lambda_function(lambda_function)
  {
  }
 
 public:
  
  virtual void executeFunctor(Integer begin,Integer size)
  {
    m_lambda_function(begin,size);
  }
 
 private:
  const LambdaType& m_lambda_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

