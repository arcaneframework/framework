﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExternalPartitionConstraint.cc                              (C) 2000-2014 */
/*                                                                           */
/* Informations sur les contraintes pour le partitionnement.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMesh.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/IMeshPartitionConstraint.h"
#include "arcane/mesh/ExternalPartitionConstraint.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


void ExternalPartitionConstraint::
addLinkedCells(Int64Array& linked_cells,Int32Array& linked_owners)
{
  m_mesh->traceMng()->info()<<"ExternalPartitionConstraint::addLinkedCells(...)";
  
  // mise en place d'un filtre sur les cell pour éviter de les mettre plusieurs fois dans le tableau cells
  UniqueArray<Integer> filtre_cell;
  filtre_cell.resize(m_mesh->allCells().size());
  Integer marque = 0;
  filtre_cell.fill(marque);

  for( ItemGroup& group : m_constraints.range() ){
    // tableau contenant la liste des mailles à maintenir ensemble
    // cette liste est distribuée sur les processeurs 
    // et comporte comme éléments communs les mailles fantômes
    UniqueArray<Cell> cells;
    if (group.itemKind() == IK_Cell){
      ENUMERATE_CELL(icell,group.cellGroup()){
        cells.add(*icell);
      }
    }
    // cette méthode permet de récupérer les mailles dans le cas d'une semi-conformité
    // on filtre les cell pour ne les sélectionner qu'une fois
    else if (group.itemKind() == IK_Face){
      marque++;
      ENUMERATE_FACE(iface,group.faceGroup()){
	ENUMERATE_NODE(inode,iface->nodes()){
	  ENUMERATE_CELL(icell, inode->cells()){
	    if (filtre_cell[icell.localId()]!=marque){
	      cells.add(*icell);
	      filtre_cell[icell.localId()]=marque;
	    }
	  }
	}
      }
    }
    else if (group.itemKind() == IK_Node){
      ENUMERATE_NODE(inode,group.nodeGroup()){
        ENUMERATE_CELL(icell, inode->cells()){
          cells.add(*icell);
        }
      }
    }
    
    if (cells.size()==0)
      continue;

    // on renseigne les contraintes sous forme de couples de mailles
    Cell cell0 = cells[0];
    Int32 owner0 = cell0.owner();
    ItemUniqueId uid0 = cell0.uniqueId();

    for (Integer i=1; i<cells.size(); ++i){
      ItemUniqueId uidi = cells[i].uniqueId();
      if (uid0<uidi){
        linked_cells.add(uid0);
        linked_cells.add(uidi);
      }
      else {
        linked_cells.add(uidi);
        linked_cells.add(uid0);
      }
      linked_owners.add(owner0);
    } // end for i=1; i<cells.size()

    m_mesh->traceMng()->info()<<"cells.size() = "<<cells.size();


  } // end for iter=m_constraints.begin()
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
