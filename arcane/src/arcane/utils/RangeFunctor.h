// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RangeFunctor.h                                              (C) 2000-2021 */
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

namespace Arcane
{

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
 * Cette classe est utilisée avec le mécanisme des lambda fonctions du C++11.
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
  
  void executeFunctor(Integer begin,Integer size) override
  {
    m_lambda_function(begin,size);
  }
 
 private:
  const LambdaType& m_lambda_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor sur un interval d'itération instancié via une lambda fonction.
 *
 * Cette classe est utilisée avec le mécanisme des lambda fonctions du C++11.
 */
template<int RankValue,typename LambdaType>
class LambdaMDRangeFunctor
: public IMDRangeFunctor<RankValue>
{
 public:
  LambdaMDRangeFunctor(const LambdaType& lambda_function)
  : m_lambda_function(lambda_function)
  {
  }
 
 public:
  
  void executeFunctor(const ComplexLoopRanges<RankValue>& loop_range) override
  {
    m_lambda_function(loop_range);
  }
 
 private:
  const LambdaType& m_lambda_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

