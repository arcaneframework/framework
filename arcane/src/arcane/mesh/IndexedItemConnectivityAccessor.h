// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexedItemConnectivityAccessor.h                               (C) 2000-2021 */
/*                                                                           */
/* Connectivité incrémentale des entités.                                    */
/*---------------------------------------------------------------------------*/
#pragma once
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/IItemFamily.h"
#include "arcane/ItemVector.h"
#include "arcane/VariableTypes.h"
//#include "arcane/ItemInternal.h"
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
  IndexedItemConnectivityAccessor(IndexedItemConnectivityViewBase view, IItemFamily* target_item_family) ;

  IndexedItemConnectivityAccessor(IIncrementalItemConnectivity* connectivity) ;

  IndexedItemConnectivityAccessor() = default;

  void init(SmallSpan<const Int32> nb_item,
            SmallSpan<const Int32> indexes,
            SmallSpan<const Int32> list_data,
            IItemFamily* source_item_family,
            IItemFamily* target_item_family) ;

  ItemVectorView operator()(ItemLocalId lid) const
  {
    //assert(m_target_item_family) ;
    const Integer* ptr = & m_list_data[m_indexes[lid]];
    return const_cast<IItemFamily*>(m_target_item_family)->view(ConstArrayView<Integer>( m_nb_item[lid], ptr )) ;
  }
 private :
  IItemFamily* m_target_item_family = nullptr ;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

