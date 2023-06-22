// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*                                                                           */
/* Outil de gestion des propriétaires des nouveaux items                     */
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

// En attendant d'avoir un algorithme qui équilibre mieux
// les messages, on applique le suivant:
// - chaque sous-domaine est responsable pour déterminer le nouveau
// propriétaire des noeuds, faces, arêtes, noeuds duaux et liaisons qui lui appartiennent.
// - pour les noeuds et les arêtes, le nouveau propriétaire est le nouveau 
// propriétaire de la maille connectée à ce noeud dont le uniqueId() est le plus petit.
// - pour les faces, le nouveau propriétaire est le nouveau propriétaire
// de la maille qui est derrière cette face s'il s'agit d'une face
// interne et de la maille connectée s'il s'agit d'une face frontière.
// - pour les noeuds duaux, le nouveau propriétaire est le nouveau propriétaire
// de la maille connectée à l'élément dual
// - pour les liaisons, le nouveau propriétaire est le nouveau propriétaire
// de la maille connectée au premier noeud dual, c'est-à-dire le propriétaire
// du premier noeud dual de la liaison

class NewItemOwnerBuilder
{
public:

  NewItemOwnerBuilder() {}
 
  // Détermine la maille connectée à l'item
  template<typename T>
  inline Cell connectedCellOfItem(const T& item) const;

  // Détermine le propriétaire de l'item, c'est-à-dire
  // le propriétaire de la maille connectée à l'item
  template<typename T>
  inline Integer ownerOfItem(const T& item) const 
  {
    return connectedCellOfItem(item).owner();
  }

private:
  
  // Trouve la maille de plus petit uniqueId() d'un item
  // Polymorphisme statique : seul les types d'items avec une
  // méthode cell() sont accéptés
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

// Pour les noeuds, la maille connectée est celle le uniqueId() est le plus petit.
template<>
inline Cell NewItemOwnerBuilder::connectedCellOfItem<Node>(const Node& node) const 
{
  return _minimumUniqueIdCellOfItem(node);
}

// Pour les arêtes, la maille connectée est celle le uniqueId() est le plus petit.
template<>
inline Cell NewItemOwnerBuilder::connectedCellOfItem<Edge>(const Edge& edge) const 
{
  return _minimumUniqueIdCellOfItem(edge);
}

// Pour les faces, la maille connectée est backCell() si elle existe sinon la frontCell()
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
