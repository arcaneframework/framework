// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupImplInternal.h                                     (C) 2000-2025 */
/*                                                                           */
/* API interne à Arcane de ItemGroupImpl.                                    */
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
 * \brief API interne à Arcane de ItemGroupImpl
 */
class ARCANE_CORE_EXPORT ItemGroupImplInternal
{
 public:

  explicit ItemGroupImplInternal(ItemGroupInternal* p)
  : m_p(p)
  {}

 public:

  //! Indique que le groupe est associé à un constituant.
  void setAsConstituentGroup();

  //! Liste des localId() des entités du groupe.
  SmallSpan<Int32> itemsLocalId();

  /*!
   * \brief Notifie l'instance qu'on a directement modifié la liste des entités du groupe.
   *
   * \a nb_remaining est le nombre d'entités restantes et \a removed_ids la liste
   * des entités supprimées.
   */
  void notifyDirectRemoveItems(SmallSpan<const Int32> removed_ids, Int32 nb_remaining);

  //! Indique que le padding SIMD des entités à été effectué
  void notifySimdPaddingDone();

  //! Change la ressource mémoire utilisée pour conserver les localId() des entités
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
