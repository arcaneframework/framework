// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshToMeshTransposer.cc                                     (C) 2000-2009 */
/*                                                                           */
/* Operator for transposition between sub-meshes.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/IMesh.h"
#include "arcane/Item.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/ItemTypes.h"
#include "arcane/IItemFamily.h"
#include "arcane/MeshToMeshTransposer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eItemKind
MeshToMeshTransposer::
kindTranspose(eItemKind kindA, IMesh * meshA, IMesh * meshB) {
  // Case of nodes which are always of fixed dimension
  if (kindA == IK_Node) return IK_Node;

  // We keep the hierarchy Node <= Edge <= Face <= Cell
  // even if sometimes in low dimensions certain notions collapse.
  Integer dimA = meshA->dimension();
  Integer dimB = meshB->dimension();

  Integer iKind;
  switch (kindA) {
  case IK_Node:
    iKind = 0; break;
  case IK_Edge:
    iKind = 1; break;
  case IK_Face:
    iKind = 2; break;
  case IK_Cell:
    iKind = 3; break;
  default:
    throw FatalErrorException(A_FUNCINFO,"Cannot transpose unknown kind");
  }

  iKind = iKind - dimB + dimA; // numerical transposition (kindA+dimA==kindB+dimB)
  
  eItemKind i2k_mapper[4] = { IK_Node, IK_Edge, IK_Face, IK_Cell };
  if (iKind < 0 || iKind > 3)
    throw FatalErrorException(A_FUNCINFO,"Cannot transpose unknown dimension");
  eItemKind kindB = i2k_mapper[iKind];

  // Handling of dimension condensation
  if (kindB == IK_Edge && dimB < 3) kindB = IK_Node;
  else if (kindB == IK_Face && dimB < 2) kindB = IK_Node;

  return kindB;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVector MeshToMeshTransposer::
transpose(IMesh * meshA, IMesh * meshB, ItemVectorView itemsA, bool do_fatal)
{
  ARCANE_ASSERT((meshA!=NULL && meshB!=NULL),("Bad NULL mesh"));

  // Empty itemsA => empty return
  if (itemsA.size() == 0)
    return ItemVector();

  // We must calculate the transition on the kind
  eItemKind kindA = itemsA[0].kind(); // since itemsA is not empty
  eItemKind kindB = kindTranspose(kindA,meshA,meshB);
  IItemFamily * familyA = meshA->itemFamily(kindA);
  IItemFamily * familyB = meshB->itemFamily(kindB);
  return _transpose(familyA,familyB,itemsA,do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVector MeshToMeshTransposer::
transpose(IItemFamily * familyA, IItemFamily * familyB, ItemVectorView itemsA, bool do_fatal)
{
  ARCANE_ASSERT((familyA!=NULL && familyB!=NULL),("Bad NULL mesh"));

  // Empty itemsA => empty return
  if (itemsA.size() == 0)
    return ItemVector();

  return _transpose(familyA,familyB,itemsA,do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVector MeshToMeshTransposer::
_transpose(IItemFamily * familyA, IItemFamily * familyB, const ItemVectorView & itemsA, bool do_fatal)
{
  eItemKind kindB = familyB->itemKind();
  IItemFamily * parent_familyA = familyA->parentFamily();
  IItemFamily * parent_familyB = familyB->parentFamily();

  // Only processes transposition on a single level
  if (parent_familyA == familyB) {
    // meshA is a sub-mesh of meshB
    UniqueArray<Int32> lidsB(itemsA.size(),NULL_ITEM_LOCAL_ID);
    ENUMERATE_ITEM(iitem,itemsA) {
      const Item & item = *iitem;
      lidsB[iitem.index()] = item.parent().localId();
    }
    return ItemVector(familyB,lidsB);
  }
  else if (parent_familyB == familyA) {
    // meshB is a sub-mesh of meshA
    if (kindB==IK_Node || kindB==IK_Face || kindB==IK_Edge || kindB==IK_Cell ) {
      // Currently the uids are the same between sub-meshes and parent mesh 
      // By transitivity, this comes down to searching for the uids of itemsA in meshB
      UniqueArray<Int64> uidsA(itemsA.size());
      ENUMERATE_ITEM(iitem,itemsA) {
        uidsA[iitem.index()] = iitem->uniqueId();
      }
      UniqueArray<Int32> lidsB(uidsA.size());
      familyB->itemsUniqueIdToLocalId(lidsB,uidsA,do_fatal);
      return ItemVector(familyB,lidsB);
    }
    else {
      throw NotImplementedException(A_FUNCINFO,"Cannot only transpose item to cell or node");
    }
  } else if (familyA == familyB) {
    // same mesh
    return ItemVector(familyB,itemsA.localIds());
  } else {
    throw NotImplementedException(A_FUNCINFO,String::format("Cannot transpose between families {0}::{1} and {2}::{3}",familyA->mesh()->name(),familyA->name(),familyA->mesh()->name(),familyB->name()));
  }

//   // The implementation assumes and verifies that meshA and meshB have the same parent mesh
//   {
//     IMesh * masterSupportA = meshA;
//     while (masterSupportA->parentMesh()) {
//       masterSupportA = masterSupportA->parentMesh();
//     }
//     IMesh * masterSupportB = meshB;
//     while (masterSupportB->parentMesh()) {
//       masterSupportB = masterSupportB->parentMesh();
//     }
//     if (masterSupportA != masterSupportB)
//       throw FatalErrorException(A_FUNCINFO,"Non common parent for transposition between meshes");
//   }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
