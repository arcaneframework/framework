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
 * \brief Classe de base des fonctors sur une liste d'entités.
 *
 * Cette classe permet de scinder une itération sur un ItemVector en
 * garantissant que les itérations se font sur un multiple de \a m_block_size.
 * Pour l'instant cette valeur vaut toujours 8 et donc les itérations sur
 * entités se font par bloc de 8 valeurs. Cela permet de garantir pour la
 * vectorisation que les sous-vues de \a m_items seront correctement alignées.
 */
class ARCANE_CORE_EXPORT AbstractItemRangeFunctor
: public IRangeFunctor
{
 public:

  static const Integer DEFAULT_GRAIN_SIZE = 400;

  AbstractItemRangeFunctor(ItemVectorView items_view,Integer grain_size);

 public:

  //! Nombre d'indexs.
  Integer nbBlock() { return m_nb_block; }
  //! Taille souhaitée d'un intervalle d'itération.
  Integer blockGrainSize() const { return m_block_grain_size; }

 protected:

  ItemVectorView m_items;
  Integer m_block_size;
  Integer m_nb_block;
  Integer m_block_grain_size;

 protected:

  ItemVectorView _view(Integer begin_block,Integer nb_block) const;

 private:
};

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

