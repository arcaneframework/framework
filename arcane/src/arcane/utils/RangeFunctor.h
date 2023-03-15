// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
#include "arcane/core/MathUtils.h"

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
 * \brief Classe de base des fonctors sur un container.
 *
 * Cette classe permet de scinder une itération sur un container en
 * garantissant que les itérations se font sur un multiple de \a m_block_size.
 * Pour l'instant cette valeur vaut toujours 8 et donc les itérations sur
 * entités se font par bloc de 8 valeurs. Cela permet de garantir pour la
 * vectorisation que les sous-vues de \a m_view seront correctement alignées.
 */
template <typename T>
class AbstractRangeFunctorT
: public IRangeFunctor
{
 public:

  static const Integer DEFAULT_GRAIN_SIZE = 400;

  struct SubViewRange {
   Integer m_begin;
   Integer m_size;
  };

  AbstractRangeFunctorT(T view,Integer grain_size)
  : m_view(view)
  , m_block_size(SIMD_PADDING_SIZE)
  , m_nb_block(view.size())
  , m_block_grain_size(grain_size)
  {
    // NOTE: si le range functor est utilisé pour la vectorisation, il faut
    // que items_view.localIds() soit aligné. Le problème est qu'on ne sait
    // pas exactement quel est l'alignement requis. On pourrait se base sur
    // \a m_block_size et dire que l'alignement est m_block_size * sizeof(Int32).
    // De toute facon, le problème éventuel d'alignement sera détecté par
    // SimdItemEnumerator.
    Integer nb_item = m_view.size();
    m_nb_block = nb_item / m_block_size;
    if ( (nb_item % m_block_size)!=0 )
      ++m_nb_block;

    m_block_grain_size = grain_size / m_block_size;
  }

 public:

  //! Nombre d'indexs.
  Integer nbBlock() { return m_nb_block; }
  //! Taille souhaitée d'un intervalle d'itération.
  Integer blockGrainSize() const { return m_block_grain_size; }
  //! Range de la subview sous la forme (begin,size) en convertissant (begin_block,nb_block) à m_items.
  SubViewRange getSubViewRange(Integer begin_block,Integer nb_block) const
  {
    Integer begin = begin_block * m_block_size;
    Integer nb_item = m_view.size();
    Integer size = math::min(nb_block * m_block_size, nb_item - begin);
    return SubViewRange{begin, size};
  }

 protected:

  T m_view;
  Integer m_block_size;
  Integer m_nb_block;
  Integer m_block_grain_size;

 protected:

  T _view(Integer begin_block,Integer nb_block) const
  {
    // Converti (begin_block,nb_block) en (begin,size) correspondant à m_items.
    Integer begin = begin_block * m_block_size;
    Integer nb_item = m_view.size();
    Integer size = math::min(nb_block * m_block_size, nb_item - begin);
    return m_view.subView(begin, size);
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Alias sur le type du 1er variadic argument pour plus de lisibilité
// TODO: surement à déplacer ailleurs...
template<typename... Types>
using FirstVariadicType = std::tuple_element_t<0, std::tuple<Types...>>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor sur un interval d'itération instancié via une lambda fonction.
 *
 * Cette classe est utilisée avec le mécanisme des lambda fonctions du C++1x.
 * Elle permet la gestion de plusieurs vues en paramètres de la lambda
 * 
 * \remark L'héritage est préservé via le type de la 1ere vue (paramètre de la lambda)
 * et permet de garder les fonctionnalités de découpage en blocs des vues
 * 
 */
template<typename LambdaType, typename... Views>
class LambdaRangeFunctorTVa
: public AbstractRangeFunctorT<FirstVariadicType<Views...>>
{
 public:
  LambdaRangeFunctorTVa(Views... views,const LambdaType& lambda_function,
      Integer grain_size = AbstractRangeFunctorT<FirstVariadicType<Views...>>::DEFAULT_GRAIN_SIZE)
  : AbstractRangeFunctorT<FirstVariadicType<Views...>>(std::get<0>(std::forward_as_tuple(views...)),grain_size)
  , m_lambda_function(lambda_function)
  , m_views(std::forward_as_tuple(views...))
  {
  }
 
 public:
  void executeFunctor(Integer begin,Integer size) override
  {
    std::tuple<Views...> sub_views;
    getSubView(sub_views, begin, size, std::make_index_sequence<sizeof...(Views)>{});
    std::apply(m_lambda_function, sub_views);
  }

 private:
  //! méthode interne pour récupérer l'intervalle pour découper les vues en blocs à traiter en parallèle
  template <size_t... I>
  void getSubView(std::tuple<Views...>& sub_views, Integer begin_block, Integer nb_block, std::index_sequence<I...>)
  {
    auto [begin, size] = this->getSubViewRange(begin_block,nb_block);
    ((std::get<I>(std::forward<decltype(sub_views)>(sub_views)) =
      std::get<I>(std::forward<decltype(m_views)>(m_views)).subView(begin,size)), ...);
  }

 private:
  const LambdaType& m_lambda_function;
  std::tuple<Views...> m_views;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

