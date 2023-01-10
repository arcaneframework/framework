// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemIndexedListView.h                                       (C) 2000-2023 */
/*                                                                           */
/* Vue sur une liste indexée d'entités.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINDEXEDLISTVIEW_H
#define ARCANE_ITEMINDEXEDLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/ItemTypes.h"
#include "arcane/ItemSharedInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Vue interne sur un tableau d'entités.
 *
 * Celle classe n'est utile que pour construire des listes d'entités utilisées
 * en interne de %Arcane. La version utilisateur de cette classe est
 * ItemVectorView.
 *
 * \sa ItemVectorView
 */
template <int Extent>
class ARCANE_CORE_EXPORT ItemIndexedListView
{
  static_assert(Extent == (-1), "only dynamic (-1) extent is currently supported");

  friend ItemInternalConnectivityList;
  friend ItemBase;
  friend ItemVectorView;

 public:

  ItemIndexedListView() = default;

 private:

  constexpr ItemIndexedListView(ItemSharedInfo* si, SmallSpan<const Int32> local_ids)
  : m_local_ids(local_ids)
  , m_shared_info(si)
  {
    ARCANE_ASSERT(m_shared_info, ("null shared_info"));
  }

  constexpr ItemIndexedListView(ItemSharedInfo* si, const Int32* local_ids, Int32 count)
  : m_local_ids(local_ids, count)
  , m_shared_info(si)
  {
    ARCANE_ASSERT(m_shared_info, ("null shared info"));
  }

 private:

  //! Nombre d'éléments du vecteur
  constexpr Int32 size() const { return m_local_ids.size(); }

  //! Tableau des numéros locaux des entités
  constexpr SmallSpan<const Int32, Extent> localIds() const { return m_local_ids; }

  //! Tableau des numéros locaux des entités
  ConstArrayView<Int32> constLocalIds() const { return m_local_ids.constSmallView(); }

 private:

  SmallSpan<const Int32, Extent> m_local_ids;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
