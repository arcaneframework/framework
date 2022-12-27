// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexedItemConnectivityAccessor.h                           (C) 2000-2022 */
/*                                                                           */
/* Connectivité incrémentale des entités.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_INDEXEDITEMCONNECTIVITYACCESSOR_H
#define ARCANE_MESH_INDEXEDITEMCONNECTIVITYACCESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/IItemFamily.h"
#include "arcane/ItemVector.h"
#include "arcane/VariableTypes.h"
#include "arcane/IIncrementalItemConnectivity.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT IndexedItemConnectivityAccessor
: public IndexedItemConnectivityViewBase
{
 public:

  IndexedItemConnectivityAccessor(IndexedItemConnectivityViewBase view, IItemFamily* target_item_family);
  IndexedItemConnectivityAccessor(IIncrementalItemConnectivity* connectivity);
  IndexedItemConnectivityAccessor() = default;

  ItemVectorView operator()(ItemLocalId lid) const
  {
    ItemLocalIdViewT<Item> x = this->items(lid);
    return const_cast<IItemFamily*>(m_target_item_family)->view(x.toViewInt32());
  }

 private:

  IItemFamily* m_target_item_family = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
