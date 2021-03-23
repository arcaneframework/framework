// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshToMeshTransposer.cc                                     (C) 2000-2009 */
/*                                                                           */
/* Opérateur de transposition entre sous-maillages.                          */
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
  // Cas des noeuds qui sont toujours de dimension fixe
  if (kindA == IK_Node) return IK_Node;

  // On garde la hiérarchie Node <= Edge <= Face <= Cell
  // même si parfois en faible dimension certaines notions collapsent.
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

  iKind = iKind - dimB + dimA; // transposition numérique (kindA+dimA==kindB+dimB)
  
  eItemKind i2k_mapper[4] = { IK_Node, IK_Edge, IK_Face, IK_Cell };
  if (iKind < 0 || iKind > 3)
    throw FatalErrorException(A_FUNCINFO,"Cannot transpose unknown dimension");
  eItemKind kindB = i2k_mapper[iKind];

  // Gestion de la condensation de dimension
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

  // On doit calculer la transition sur le kind
  eItemKind kindA = itemsA[0].kind(); // vu que itemsA n'est pas vide
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

  // Ne traite que la transposition sur un seul niveau
  if (parent_familyA == familyB) {
    // meshA est sous-maillage de meshB
    SharedArray<Int32> lidsB(itemsA.size(),NULL_ITEM_LOCAL_ID);
    ENUMERATE_ITEM(iitem,itemsA) {
      const Item & item = *iitem;
      lidsB[iitem.index()] = item.parent().localId();
    }
    return ItemVector(familyB,lidsB,false);
  } else if (parent_familyB == familyA) {
    // meshB est sous-maillage de meshA
    if (kindB==IK_Node || kindB==IK_Face || kindB==IK_Edge || kindB==IK_Cell || kindB==IK_DualNode) {
      // Actuellement les uids sont les mêmes entre sous-maillages et maillage parent 
      // Par transitivité, cela revient à chercher les uids de itemsA dans meshB
      UniqueArray<Int64> uidsA(itemsA.size());
      ENUMERATE_ITEM(iitem,itemsA) {
        uidsA[iitem.index()] = iitem->uniqueId();
      }
      SharedArray<Int32> lidsB(uidsA.size());
      familyB->itemsUniqueIdToLocalId(lidsB,uidsA,do_fatal);
      return ItemVector(familyB,lidsB,false);
    } else {
      throw NotImplementedException(A_FUNCINFO,"Cannot only transpose item to cell or node");
    }
  } else if (familyA == familyB) {
    // même maillage
    return ItemVector(familyB,itemsA.localIds());
  } else {
    throw NotImplementedException(A_FUNCINFO,String::format("Cannot transpose between families {0}::{1} and {2}::{3}",familyA->mesh()->name(),familyA->name(),familyA->mesh()->name(),familyB->name()));
  }

//   // L'implémentation suppose et vérifie que meshA et meshB ont le même maillage parent
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
