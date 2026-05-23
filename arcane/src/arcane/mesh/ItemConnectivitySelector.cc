// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivitySelector.cc                                 (C) 2000-2021 */
/*                                                                           */
/* Selection between historical and on-demand connectivities.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/IMesh.h"
#include "arcane/IIncrementalItemConnectivity.h"

#include "arcane/mesh/ItemConnectivitySelector.h"
#include "arcane/mesh/ItemFamily.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemConnectivitySelector::
ItemConnectivitySelector(ItemFamily* source_family,IItemFamily* target_family,
                         const String& connectivity_name,Integer connectivity_index)
: TraceAccessor(source_family->traceMng())
, m_source_family(source_family)
, m_target_family(target_family)
, m_connectivity_name(connectivity_name)
, m_pre_allocated_size(0)
, m_item_connectivity_index(connectivity_index)
, m_item_connectivity_list(m_source_family->itemInternalConnectivityList())
, m_is_built(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivitySelector::
build()
{
  if (m_is_built)
    return;

  _createCustomConnectivity(m_connectivity_name);
  info(4) << "Family: " << m_source_family->fullName()
          << " create new connectivity: " << m_connectivity_name;

  _buildCustomConnectivity();
  m_is_built = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivitySelector::
setPreAllocatedSize(Integer size)
{
  m_pre_allocated_size = size;
  auto c = customConnectivity();
  // For new connectivities, the pre-allocation value is saved
  // during a protection and is not taken into account during recovery.
  if (c)
    c->setPreAllocatedSize(size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
