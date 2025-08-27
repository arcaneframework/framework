// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLoop.h                                                  (C) 2000-2018 */
/*                                                                           */
/* Classes utilitaires pour gérer les boucles sur les entités.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMLOOP_H
#define ARCANE_ITEMLOOP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file ItemLoop.h
 *
 * \brief Types et macros pour gérer les boucles sur les entités du maillage.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*!
 * \brief Espace de nom contenant les différentes classes gérant les boucles
 * sur les entités.
 */
namespace Loop
{
/*!
 * \internal
 * \brief Fonctor de boucle d'entité permettant de supprimer les
 * indirections si les indices locaux de \a view sont consécutifs.
 */
template<typename IterType,typename Lambda> inline void
_InternalSimpleItemLoop(ItemVectorView view,const Lambda& lambda)
{
  if (view.size()==0)
    return;
  bool is_contigous = view.indexes().isContigous();
  //is_contigous = false;
  if (is_contigous){
    Int32 x0 = view.localIds()[0];
    // Suppose que les itérations sont indépendantes
    ARCANE_PRAGMA_IVDEP
    for( Int32 i=0, n=view.size(); i<n; ++i )
      lambda(IterType(x0+i));
  }
  else{
    ENUMERATE_ITEM(iitem,view){
      lambda(IterType(iitem.localId()));
    }
  }
}
/*!
 * \brief Classe template pour encapsuler une boucle sur les entités.
 */
template<typename ItemType>
class ItemLoopFunctor
{
 public:
  typedef typename ItemType::Index IterType;
  typedef ItemVectorViewT<ItemType> VectorViewType;
  typedef ItemGroupT<ItemType> ItemGroupType;
  typedef ItemLoopFunctor<ItemType> ThatClass;
 private:
  ItemLoopFunctor(ItemVectorView items)
  : m_items(items){}
 public:
  static ThatClass create(const ItemGroupType& items)
  { return ThatClass(items.view()); }
  static ThatClass create(VectorViewType items)
  { return ThatClass(items); }
 public:
  template<typename Lambda>
  void operator<<(Lambda&& lambda)
  {
    _InternalSimpleItemLoop<IterType>(m_items,lambda);
  }
 private:
  ItemVectorView m_items;
};

typedef ItemLoopFunctor<Cell> ItemLoopFunctorCell;
typedef ItemLoopFunctor<Node> ItemLoopFunctorNode;

} // End of namespace Loop

} // End of namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur une entité via une fonction lambda.
 *
 * \param item_type type de l'entité (Arcane::Node, Arcane::Cell, Arcane::Edge, ....)
 * \param iter nom de l'itérateur
 * \param container conteneur associé (de type Arcane::ItemGroup ou Arcane::ItemVectorView).
 *
 * Cette macro génère une lambda et il faut donc terminer
 * l'expression par un ';'.
 *
 * Par exemple, pour itérer sur toutes les mailles:
 * \code
 * Real gamma = 1.4;
 * ENUMERATE_ITEM_LAMBDA(Cell,icell,allCells()){
 *   Real pressure = pressure[icell];
 *   Real adiabatic_cst = adiabatic_cst[icell];
 *   Real density = density[icell];
 *   internal_energy[icell] = pressure / ((gamma-1.0) * density);
 * };
 * \endcode
 * L'itérateur est de type \a item_type :: Index (par exemple Cell::Index
 * pour une maille). Il ne dispose donc pas des méthodes classiques
 * sur les entités (comme par exemple Arcane::Cell::nbNode()). L'itérateur
 * permet uniquement d'accéder aux valeurs des variables.
 *
 * La lambda est déclarée avec [=] et il est donc interdit de modifier
 * les variables capturées.
 *
 * \warning La syntaxe et la sémantique de cette macro sont expérimentales.
 * Cette macro ne doit être utilisée que pour des tests.
 */
#define ENUMERATE_ITEM_LAMBDA(item_type,iter,container) \
  Arcane::Loop:: ItemLoopFunctor ## item_type :: create ( (container) ) << [=]( Arcane::Loop:: ItemLoopFunctor ## item_type :: IterType iter)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
