// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLoop.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Utility classes for managing loops over entities.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMLOOP_H
#define ARCANE_CORE_ITEMLOOP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file ItemLoop.h
 *
 * \brief Types and macros for managing loops over mesh entities.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*!
 * \brief Namespace containing various classes managing loops
 * over entities.
 */
namespace Loop
{

  /*!
 * \internal
 * \brief Entity loop functor that allows for the removal of
 * indirections if the local indices of a view are consecutive.
 */
  template <typename IterType, typename Lambda> inline void
  _InternalSimpleItemLoop(ItemVectorView view, const Lambda& lambda)
  {
    if (view.size() == 0)
      return;
    bool is_contigous = view.indexes().isContigous();
    //is_contigous = false;
    if (is_contigous) {
      Int32 x0 = view.localIds()[0];
      // Assuming iterations are independent
      ARCANE_PRAGMA_IVDEP
      for (Int32 i = 0, n = view.size(); i < n; ++i)
        lambda(IterType(x0 + i));
    }
    else {
      ENUMERATE_ITEM (iitem, view) {
        lambda(IterType(iitem.localId()));
      }
    }
  }

  /*!
 * \brief Template class to encapsulate a loop over entities.
 */
  template <typename ItemType>
  class ItemLoopFunctor
  {
   public:

    typedef typename ItemType::Index IterType;
    typedef ItemVectorViewT<ItemType> VectorViewType;
    typedef ItemGroupT<ItemType> ItemGroupType;
    typedef ItemLoopFunctor<ItemType> ThatClass;

   private:

    ItemLoopFunctor(ItemVectorView items)
    : m_items(items)
    {}

   public:

    static ThatClass create(const ItemGroupType& items)
    {
      return ThatClass(items.view());
    }
    static ThatClass create(VectorViewType items)
    {
      return ThatClass(items);
    }

   public:

    template <typename Lambda>
    void operator<<(Lambda&& lambda)
    {
      _InternalSimpleItemLoop<IterType>(m_items, lambda);
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
 * \brief Enumerator over an entity via a lambda function.
 *
 * \param item_type entity type (Arcane::Node, Arcane::Cell, Arcane::Edge, ....)
 * \param iter name of the iterator
 * \param container associated container (of type Arcane::ItemGroup or Arcane::ItemVectorView).
 *
 * This macro generates a lambda and therefore the expression must be terminated
 * by a ';'.
 *
 * For example, to iterate over all cells:
 * \code
 * Real gamma = 1.4;
 * ENUMERATE_ITEM_LAMBDA(Cell,icell,allCells()){
 *   Real pressure = pressure[icell];
 *   Real adiabatic_cst = adiabatic_cst[icell];
 *   Real density = density[icell];
 *   internal_energy[icell] = pressure / ((gamma-1.0) * density);
 * };
 * \endcode
 * The iterator is of type \a item_type :: Index (for example Cell::Index
 * for a cell). It therefore does not have the classic methods
 * on entities (such as Arcane::Cell::nbNode()). The iterator
 * only allows access to variable values.
 *
 * The lambda is declared with [=] and it is therefore forbidden to modify
 * the captured variables.
 *
 * \warning The syntax and semantics of this macro are experimental.
 * This macro should only be used for testing.
 */
#define ENUMERATE_ITEM_LAMBDA(item_type, iter, container) \
  Arcane::Loop::ItemLoopFunctor##item_type ::create((container)) << [=](Arcane::Loop::ItemLoopFunctor##item_type ::IterType iter)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
