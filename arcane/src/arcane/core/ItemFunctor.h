// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFunctor.h                                               (C) 2000-2023 */
/*                                                                           */
/* Fonctor sur les entités.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMFUNCTOR_H
#define ARCANE_ITEMFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/RangeFunctor.h"
#include "arcane/utils/Functor.h"

#include "arcane/Item.h"
#include "arcane/ItemVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des fonctors sur un container.
 *
 * Cette classe permet de scinder une itération sur un container en
 * garantissant que les itérations se font sur un multiple de \a m_block_size.
 * Pour l'instant cette valeur vaut toujours 8 et donc les itérations sur
 * entités se font par bloc de 8 valeurs. Cela permet de garantir pour la
 * vectorisation que les sous-vues de \a m_items seront correctement alignées.
 */
template <typename T>
class ARCANE_CORE_EXPORT AbstractItemRangeFunctorT
: public IRangeFunctor
{
 public:

  static const Integer DEFAULT_GRAIN_SIZE = 400;

  struct SubViewRange {
   Integer m_begin;
   Integer m_size;
  };

  AbstractItemRangeFunctorT(T items_view,Integer grain_size)
  : m_items(items_view)
  , m_block_size(SIMD_PADDING_SIZE)
  , m_nb_block(items_view.size())
  , m_block_grain_size(grain_size)
  {
    // NOTE: si le range functor est utilisé pour la vectorisation, il faut
    // que items_view.localIds() soit aligné. Le problème est qu'on ne sait
    // pas exactement quel est l'alignement requis. On pourrait se base sur
    // \a m_block_size et dire que l'alignement est m_block_size * sizeof(Int32).
    // De toute facon, le problème éventuel d'alignement sera détecté par
    // SimdItemEnumerator.
    Integer nb_item = m_items.size();
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
    Integer nb_item = m_items.size();
    Integer size = math::min(nb_block * m_block_size,nb_item-begin);
    return SubViewRange{begin,size};
  }

 protected:

  T m_items;
  Integer m_block_size;
  Integer m_nb_block;
  Integer m_block_grain_size;

 protected:

  T _view(Integer begin_block,Integer nb_block) const
  {
    // Converti (begin_block,nb_block) en (begin,size) correspondant à m_items.
    Integer begin = begin_block * m_block_size;
    Integer nb_item = m_items.size();
    Integer size = math::min(nb_block * m_block_size,nb_item-begin);
    return m_items.subView(begin,size);
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des fonctors sur une liste d'entités.
 *
 * Cette classe permet de scinder une itération sur un ItemVector en
 * garantissant que les itérations se font sur un multiple de \a m_block_size.
 * Pour l'instant cette valeur vaut toujours 8 et donc les itérations sur
 * entités se font par bloc de 8 valeurs. Cela permet de garantir pour la
 * vectorisation que les sous-vues de \a m_items seront correctement alignées.
 */
// TODO: Y'a t'il besoin d'un ARCANE_CORE_EXPORT sur un typedef ???
using AbstractItemRangeFunctor = AbstractItemRangeFunctorT<ItemVectorView>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor pour itérer sur une liste d'entités.
 */
template<typename InstanceType,typename ItemType>
class ItemRangeFunctorT
: public AbstractItemRangeFunctor
{
 private:
  
  typedef void (InstanceType::*FunctionType)(ItemVectorViewT<ItemType>);
 
 public:
  ItemRangeFunctorT(ItemVectorView items_view,InstanceType* instance,
                    FunctionType function,Integer grain_size = DEFAULT_GRAIN_SIZE)
  : AbstractItemRangeFunctor(items_view,grain_size), m_instance(instance),
    m_function(function)
  {
  }

 private:

  InstanceType* m_instance;
  FunctionType m_function;

 public:

  virtual void executeFunctor(Integer begin,Integer size)
  {
    //cout << "** BLOCKED RANGE! range=" << range.begin() << " end=" << range.end() << " size=" << range.size() << "\n";
    //CellVectorView sub_view = m_cells.subView(range.begin(),range.size());
    ItemVectorViewT<ItemType> sub_view(this->_view(begin,size));
    //cout << "** SUB_VIEW v=" << sub_view.size();
    (m_instance->*m_function)(sub_view);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor sur un interval d'itération instancié via une lambda fonction.
 *
 * Cette classe est utilisée avec le mécanisme des lambda fonctions du C++1x.
 */
template<typename LambdaType>
class LambdaItemRangeFunctorT
: public AbstractItemRangeFunctor
{
 public:
  LambdaItemRangeFunctorT(ItemVectorView items_view,const LambdaType& lambda_function,
                          Integer grain_size = DEFAULT_GRAIN_SIZE)
  : AbstractItemRangeFunctor(items_view,grain_size), m_lambda_function(lambda_function)
  {
  }
 
 public:
  
  void executeFunctor(Integer begin,Integer size) override
  {
    ItemVectorView sub_view(this->_view(begin,size));
    m_lambda_function(sub_view);
  }
 
 private:
  const LambdaType& m_lambda_function;
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
 */
template<typename LambdaType, typename... Views>
class LambdaItemRangeFunctorT_FL
: public AbstractItemRangeFunctorT<FirstVariadicType<Views...>>
{
 public:
  LambdaItemRangeFunctorT_FL(Views... views,const LambdaType& lambda_function,
      Integer grain_size = AbstractItemRangeFunctorT<FirstVariadicType<Views...>>::DEFAULT_GRAIN_SIZE)
  : AbstractItemRangeFunctorT<FirstVariadicType<Views...>>(std::get<0>(std::forward_as_tuple(views...)),grain_size)
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
/*!
 * \brief Fonctor pour le calcul des éléments d'un groupe.
 */
class ItemGroupComputeFunctor
: public IFunctor
{
 public:
  ItemGroupComputeFunctor() : m_group(0) { }
  virtual ~ItemGroupComputeFunctor() { }
 public:
  void setGroup(ItemGroupImpl* group) { m_group = group; }
 public:
 protected:
  ItemGroupImpl* m_group;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

