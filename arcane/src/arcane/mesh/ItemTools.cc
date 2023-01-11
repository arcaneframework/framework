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
isSameFace(Face face, Int64ConstArrayView face_nodes_uid)
{
  Integer index = 0;
  for( Node node : face.nodes() ){
    if (node.uniqueId()!=face_nodes_uid[index])
      return false;
    ++index;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Face ItemTools::
findFaceInNode2(Node node,Integer face_type_id,
               Int64ConstArrayView face_nodes_uid)
{
  for( Face current_face : node.faces() ){
    if (current_face.type()!=face_type_id)
      continue;
    if (isSameFace(current_face,face_nodes_uid))
      return current_face;
  }
  return Face();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* ItemTools::
findFaceInNode(Node node, Integer face_type_id,
               Int64ConstArrayView face_nodes_uid)
{
  Face face = findFaceInNode2(node, face_type_id, face_nodes_uid);
  if (face.null())
    return nullptr;
  return ItemCompatibility::_itemInternal(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Edge ItemTools::
findEdgeInNode2(Node node,Int64 begin_node,Int64 end_node)
{
  for( Edge edge : node.edges() )
    if (edge.node(0).uniqueId()==begin_node && edge.node(1).uniqueId()==end_node)
      return edge;
  return Edge();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* ItemTools::
findEdgeInNode(Node node, Int64 begin_node, Int64 end_node)
{
  Edge edge = findEdgeInNode2(node, begin_node, end_node);
  if (edge.null())
    return nullptr;
  return ItemCompatibility::_itemInternal(edge);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
