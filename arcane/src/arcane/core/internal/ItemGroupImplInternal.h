// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupImplInternal.h                                     (C) 2000-2025 */
/*                                                                           */
/* Internal API of ItemGroupImpl in Arcane.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_ITEMGROUPIMPLINTERNAL_H
#define ARCANE_CORE_INTERNAL_ITEMGROUPIMPLINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemGroupImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal API of ItemGroupImpl in Arcane
 */
class ARCANE_CORE_EXPORT ItemGroupImplInternal
{
 public:

  explicit ItemGroupImplInternal(ItemGroupInternal* p)
  : m_p(p)
  {}

 public:

  //! Indicates that the group is associated with a constituent.
  void setAsConstituentGroup();

  //! List of localId() of the group's entities.
  SmallSpan<Int32> itemsLocalId();

  /*!
   * \brief Notifies the instance that the list of group entities has been directly modified.
   *
   * \a nb_remaining is the number of remaining entities and \a removed_ids is the list
   * of removed entities.
   */
  void notifyDirectRemoveItems(SmallSpan<const Int32> removed_ids, Int32 nb_remaining);

  //! Indicates that the SIMD padding of the entities has been performed
  void notifySimdPaddingDone();

  //! Changes the memory resource used to store the localId() of the entities
  void setMemoryRessourceForItemLocalId(eMemoryRessource mem);

 private:

  ItemGroupInternal* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
