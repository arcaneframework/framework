// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFunctor.h                                               (C) 2000-2024 */
/*                                                                           */
/* Fonctor sur les entités.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMFUNCTOR_H
#define ARCANE_ITEMFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/RangeFunctor.h"
#include "arcane/utils/Functor.h"

#include "arcane/core/Item.h"
#include "arcane/core/ItemVectorView.h"

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

  AbstractItemRangeFunctor(ItemVectorView items_view, Int32 grain_size);

 public:

  //! Nombre de blocs.
  Int32 nbBlock() const { return m_nb_block; }

  //! Taille souhaitée d'un intervalle d'itération.
  Int32 blockGrainSize() const { return m_block_grain_size; }

 protected:

  ItemVectorView m_items;
  Int32 m_block_size = 0;
  Int32 m_nb_block = 0;
  Int32 m_block_grain_size = 0;

 protected:

  ItemVectorView _view(Int32 begin_block, Int32 nb_block, Int32* true_begin = nullptr) const;

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

  virtual void executeFunctor(Int32 begin, Int32 size)
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
                          Int32 grain_size = DEFAULT_GRAIN_SIZE)
  : AbstractItemRangeFunctor(items_view,grain_size), m_lambda_function(lambda_function)
  {
  }
 
 public:

  void executeFunctor(Int32 begin, Int32 size) override
  {
    Int32 true_begin = 0;
    ItemVectorView sub_view(this->_view(begin, size, &true_begin));
    // La lambda peut avoir deux prototypes :
    // - elle prend uniquement un ItemVectorView en argument (version historique)
    // - elle prend un ItemVectorView et l'indice du début du vecteur. Cela
    // permet de connaitre l'index de l'itération
    if constexpr (std::is_invocable_v<LambdaType, ItemVectorView>)
      m_lambda_function(sub_view);
    else
      m_lambda_function(sub_view, true_begin);
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

  ItemGroupComputeFunctor() = default;

 public:

  void setGroup(ItemGroupImpl* group) { m_group = group; }

 protected:

  ItemGroupImpl* m_group = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

