// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodesOfItemReorderer.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Classe utilitaire pour réordonner les noeuds d'une entité.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/NodesOfItemReorderer.h"

#include "arcane/core/ItemTypeId.h"
#include "arcane/core/MeshUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool NodesOfItemReorderer::
reorder([[maybe_unused]] ItemTypeId type_id, ConstArrayView<Int64> nodes_uids)
{
  Int32 nb_node = nodes_uids.size();
  m_work_sorted_nodes.resize(nb_node);
  return MeshUtils::reorderNodesOfFace(nodes_uids, m_work_sorted_nodes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
