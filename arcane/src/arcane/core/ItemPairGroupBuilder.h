// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroupBuilder.h                                      (C) 2000-2025 */
/*                                                                           */
/* Construction of the entity lists for the ItemPairGroup.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMPAIRGROUPBUILDER_H
#define ARCANE_CORE_ITEMPAIRGROUPBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemPairGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Construction of the entity lists for the ItemPairGroup.
 *
 * This class is used when recalculating the entities of an ItemPairGroup.
 *
 * The user code must call the addNextItem() method for each entity
 * of group().itemGroup() by specifying the localId() of the added entities.
 * For example:
 *
 \code
 * void functor(ItemPairGroupBuilder& builder)
 * {
 *    Int32Array local_ids;
 *    ENUMERATE_ITEM(iitem.builder.group().itemGroup()){
 *      local_ids.clear();
 *      // Calculates the entities connected to \a iitem and adds them to \a local_ids.
 *      ...
 *      builder.addNextItem(local_ids);
 *    }
 * }
 \endcode
 *
 * For a more complete usage example, refer to the ItemPairGroup documentation.
 */
class ARCANE_CORE_EXPORT ItemPairGroupBuilder
{
 public:

  //! \internal
  explicit ItemPairGroupBuilder(const ItemPairGroup& group);
  ~ItemPairGroupBuilder();

 public:

  //! Associated group.
  const ItemPairGroup& group() { return m_group; }
  //! Adds the entities \a sub_items to
  void addNextItem(Int32ConstArrayView sub_items);

 private:

  ItemPairGroup m_group;
  Int64 m_index = 0;
  Array<Int64>& m_unguarded_indexes;
  Array<Int32>& m_unguarded_local_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
