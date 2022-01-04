// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTools.cc                                                (C) 2000-2020 */
/*                                                                           */
/* Utilitaires aidant à retrouver des items à partir d'autres                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/Item.h"
#include "arcane/mesh/ItemTools.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemTools::
isSameFace(ItemInternal* face, Int64ConstArrayView face_nodes_uid)
{
  Integer index = 0;
  for( ItemInternal* i_node : face->internalNodes() ){
    if (i_node->uniqueId()!=face_nodes_uid[index])
      return false;
    ++index;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* ItemTools::
findFaceInNode(ItemInternal* node,Integer face_type_id,
               Int64ConstArrayView face_nodes_uid)
{
  for( ItemInternal* current_face : node->internalFaces() ){
    if (current_face->typeId()!=face_type_id)
      continue;
    if (isSameFace(current_face,face_nodes_uid))
      return current_face;
  }
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* ItemTools::
findEdgeInNode(ItemInternal* inode,Int64 begin_node,Int64 end_node)
{
  Node node(inode);
  for( Edge edge : node.edges() )
    if (edge.node(0).uniqueId()==begin_node && edge.node(1).uniqueId()==end_node)
      return edge.internal();
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
