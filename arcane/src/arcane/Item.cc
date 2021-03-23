// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Item.cc                                                     (C) 2000-2020 */
/*                                                                           */
/* Classe de base d'un élément du maillage.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotSupportedException.h"

#include "arcane/Item.h"
#include "arcane/ItemCompare.h"
#include "arcane/ItemPrinter.h"
#include "arcane/MeshItemInternalList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
  if (t >= 0 && t < ItemTypeMng::nbBasicItemType())
    return ItemTypeMng::singleton()->typeFromId(t)->typeName();
  else
    return "InvalidType";
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
  throw FatalErrorException("Item::_badConversion",String("Bad conversion from ")+itemKindName(kind()));
#else /* ARCANE_DEBUG */
  throw FatalErrorException("Item::_badConversion","Bad conversion");
#endif /* ARCANE_DEBUG */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Positionne l'item dual dont le noeud est dual
void DualNode::
setDualItem(const Item& item)
{
  ARCANE_UNUSED(item);
  ARCANE_THROW(NotSupportedException,"");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namesapce Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
