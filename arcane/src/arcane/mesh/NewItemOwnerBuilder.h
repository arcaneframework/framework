// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*                                                                           */
/* Owner management tool for new items                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_NEWITEMOWNERBUILDER_H
#define ARCANE_MESH_NEWITEMOWNERBUILDER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// While waiting for an algorithm that better balances
// messages, we apply the following:
// - each sub-domain is responsible for determining the new
// owner of nodes, faces, edges, dual nodes, and links belonging to it.
// - for nodes and edges, the new owner is the new owner of the mesh connected to this node whose uniqueId() is the smallest.
// - for faces, the new owner is the new owner of the mesh behind this face if it is an internal face, and the connected mesh if it is a boundary face.
// - for dual nodes, the new owner is the new owner of the mesh connected to the dual element
// - for links, the new owner is the new owner of the mesh connected to the first dual node, that is, the owner of the link's first dual node.

class NewItemOwnerBuilder
{
public:

  NewItemOwnerBuilder() {}
 
  // Determines the mesh connected to the item
  template<typename T>
  inline Cell connectedCellOfItem(const T& item) const;

  // Determines the owner of the item, that is,
  // the owner of the mesh connected to the item
  template<typename T>
  inline Integer ownerOfItem(const T& item) const 
  {
    return connectedCellOfItem(item).owner();
  }

private:
  
  // Finds the mesh with the smallest uniqueId() of an item
  // Static polymorphism: only item types with a
  // cell() method are accepted
  template<typename T>
  inline Cell _minimumUniqueIdCellOfItem(const T& item) const 
  {
    Cell cell = item.cell(0);
    for( Cell item_cell : item.cells() ){
      if (item_cell.uniqueId() < cell.uniqueId())
        cell = item_cell;
    }
    return cell;
  }
};

// For nodes, the connected mesh is the one with the smallest uniqueId().
template<>
inline Cell NewItemOwnerBuilder::connectedCellOfItem<Node>(const Node& node) const 
{
  return _minimumUniqueIdCellOfItem(node);
}

// For edges, the connected mesh is the one with the smallest uniqueId().
template<>
inline Cell NewItemOwnerBuilder::connectedCellOfItem<Edge>(const Edge& edge) const 
{
  return _minimumUniqueIdCellOfItem(edge);
}

// For faces, the connected mesh is backCell() if it exists, otherwise frontCell()
template<>
inline Cell NewItemOwnerBuilder::connectedCellOfItem<Face>(const Face& face) const 
{
  Cell cell = face.backCell();
  if (cell.null())
    cell = face.frontCell();
  return cell;
}
template<>
inline Cell NewItemOwnerBuilder::connectedCellOfItem<Particle>(const Particle& particle) const
{
  return particle.cell();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_MESH_NEWITEMOWNERBUILDER_H */
