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
    auto* ptr = reinterpret_cast<const Int32*>(&m_list_data[m_indexes[lid]]);
    Int32ConstArrayView v(m_nb_item[lid], ptr);
    return const_cast<IItemFamily*>(m_target_item_family)->view(v);
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
