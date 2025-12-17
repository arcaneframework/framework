// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RangeFunctor.h                                              (C) 2000-2025 */
/*                                                                           */
/* Fonctor sur un interval d'itération.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_RANGEFUNCTOR_H
#define ARCCORE_BASE_RANGEFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/IRangeFunctor.h"

#include <tuple>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor sur un interval d'itération.
 */
template <typename InstanceType>
class RangeFunctorT
: public IRangeFunctor
{
 private:

  typedef void (InstanceType::*FunctionType)(Integer i0, Integer size);

 public:

  RangeFunctorT(InstanceType* instance, FunctionType function)
  : m_instance(instance)
  , m_function(function)
  {
  }

 public:

  virtual void executeFunctor(Integer begin, Integer size)
  {
    (m_instance->*m_function)(begin, size);
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
template <typename LambdaType>
class LambdaRangeFunctorT
: public IRangeFunctor
{
 public:

  LambdaRangeFunctorT(const LambdaType& lambda_function)
  : m_lambda_function(lambda_function)
  {
  }

 public:

  void executeFunctor(Integer begin, Integer size) override
  {
    m_lambda_function(begin, size);
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
template <int RankValue, typename LambdaType>
class LambdaMDRangeFunctor
: public IMDRangeFunctor<RankValue>
{
 public:

  LambdaMDRangeFunctor(const LambdaType& lambda_function)
  : m_lambda_function(lambda_function)
  {
  }

 public:

  void executeFunctor(const ComplexForLoopRanges<RankValue>& loop_range) override
  {
    m_lambda_function(loop_range);
  }

 private:

  const LambdaType& m_lambda_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor sur un interval d'itération instancié via une lambda fonction.
 *
 * Cette classe est utilisée avec le mécanisme des lambda fonctions du C++1x.
 * Elle permet la gestion de plusieurs vues en paramètres de la lambda
 * 
 */
template <typename LambdaType, typename... Views>
class LambdaRangeFunctorTVa
: public IRangeFunctor
{
 public:

  LambdaRangeFunctorTVa(Views... views, const LambdaType& lambda_function)
  : m_lambda_function(lambda_function)
  , m_views(std::forward_as_tuple(views...))
  {
  }

 public:

  void executeFunctor(Integer begin, Integer size) override
  {
    std::tuple<Views...> sub_views;
    getSubView(sub_views, begin, size, std::make_index_sequence<sizeof...(Views)>{});
    std::apply(m_lambda_function, sub_views);
  }

 private:

  //! méthode interne pour découper les vues
  template <size_t... I>
  void getSubView(std::tuple<Views...>& sub_views, Integer begin, Integer size, std::index_sequence<I...>)
  {
    ((std::get<I>(std::forward<decltype(sub_views)>(sub_views)) =
      std::get<I>(std::forward<decltype(m_views)>(m_views)).subView(begin, size)),
     ...);
  }

 private:

  const LambdaType& m_lambda_function;
  std::tuple<Views...> m_views;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
