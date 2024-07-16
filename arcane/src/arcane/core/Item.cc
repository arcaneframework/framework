// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Item.cc                                                     (C) 2000-2024 */
/*                                                                           */
/* Classe de base d'un élément du maillage.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Item.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/ItemCompare.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/MeshItemInternalList.h"
#include "arcane/core/IndexedItemConnectivityView.h"
#include "arcane/core/ItemConnectedEnumerator.h"

#include "arcane/core/ItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

std::atomic<int> Item::m_nb_created_from_internal = 0;
std::atomic<int> Item::m_nb_created_from_internalptr = 0;
std::atomic<int> Item::m_nb_set_from_internal = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \defgroup GroupItem Les éléments du maillage
 *
 * Il s'agit de l'ensemble des classes traitant les entités de maillage.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Voir Item.h et ItemSharedInfo.h pour les #ifdef associés

String Item::
typeName(Integer t)
{
  return ItemTypeMng::_legacyTypeName(t);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal ItemInternal::nullItemInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Item::
_badConversion() const
{
#ifdef ARCANE_DEBUG
  ARCANE_FATAL("Bad conversion from {0}",kind());
#else /* ARCANE_DEBUG */
  ARCANE_FATAL("Bad conversion");
#endif /* ARCANE_DEBUG */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IndexedItemConnectivityViewBase::
_badConversion(eItemKind k1,eItemKind k2) const
{
  ARCANE_FATAL("Can not convert connectivity view ({0},{1}) to ({2},{3})",
               m_source_kind,m_target_kind,k1,k2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Item::
dumpStats(ITraceMng* tm)
{
  tm->info() << "ItemStats: nb_created_from_internal = " << m_nb_created_from_internal;
  tm->info() << "ItemStats: nb_created_from_internalptr = " << m_nb_created_from_internalptr;
  tm->info() << "ItemStats: nb_set_from_internal = " << m_nb_set_from_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Item::
resetStats()
{
  m_nb_created_from_internal = 0;
  m_nb_created_from_internalptr = 0;
  m_nb_set_from_internal = 0;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Pour tester l'instantiation de ces classes
template class ItemConnectedListViewT<Node>;
template class ItemConnectedEnumeratorBaseT<Node>;
template class ItemConnectedEnumeratorT<Node>;
template class ItemConnectedEnumeratorT<Item>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namesapce Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
