// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPrinter.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Ecriture d'Item sur flux.                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/ItemPrinter.h"
#include "arcane/ItemInternalEnumerator.h"
#include "arcane/IItemFamily.h"
#include "arcane/IMesh.h"
#include "arcane/MeshPartInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * TODO: Il faudrait gérer l'affichage de l'entité via IItemFamily.
 *
 * Cela permettrait de spécialiser l'affichage par famille.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using impl::ItemBase;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct ARCANE_CORE_EXPORT ItemPrinter::Internal
{
  //! Ecrit les informations basiques de l'item
  static void _printBasics(std::ostream& o, ItemBase item);
  //! Ecrit les flags de l'item de manière explicite
  static void _printFlags(std::ostream& o, Integer flags);
  //! Ecrit les infos sur les parents
  static void _printParents(std::ostream& o, ItemBase item);
  //! Ecrit les infos sur les parents
  static void _printErrors(std::ostream& o, ItemBase item);
  //! Ecrit les informations d'une énumération d'items
  static void _printItemSubItems(std::ostream& ostr, String name, const ItemVectorView& enumerator);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPrinter::
print(std::ostream& o) const
{
  if (m_item.null()) {
    if (m_has_item_kind) {
      o << "(null " << itemKindName(m_item_kind) << ")";
    }
    else {
      o << "(null Item)";
    }
  }
  else {
    o << "(";
    ItemPrinter::Internal::_printBasics(o, m_item);
    ItemPrinter::Internal::_printFlags(o, m_item.flags());
    ItemPrinter::Internal::_printParents(o, m_item);
    ItemPrinter::Internal::_printErrors(o, m_item);
    o << ")";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FullItemPrinter::
print(std::ostream& ostr) const
{
  if (m_item.null()) {
    ostr << "(null Item)";
  }
  else {
    eItemKind ik = m_item.kind();
    ostr << "(";
    ItemPrinter::Internal::_printBasics(ostr, m_item);
    ItemPrinter::Internal::_printFlags(ostr, m_item.flags());
    ItemPrinter::Internal::_printParents(ostr, m_item);
    ItemPrinter::Internal::_printErrors(ostr, m_item);
    ostr << ")";
    ostr << "\n\t";
    if (ik != IK_Node)
      if (m_item.nbNode() != 0)
        ItemPrinter::Internal::_printItemSubItems(ostr, "Nodes", m_item.nodeList());
    ostr << "\n\t";
    if (ik != IK_Edge)
      if (m_item.nbEdge() != 0)
        ItemPrinter::Internal::_printItemSubItems(ostr, "Edges", m_item.edgeList());
    ostr << "\n\t";
    if (m_item.nbFace() != 0)
      ItemPrinter::Internal::_printItemSubItems(ostr, "Faces", m_item.faceList());
    ostr << "\n\t";
    if (ik != IK_Cell)
      if (m_item.nbCell() != 0)
        ItemPrinter::Internal::_printItemSubItems(ostr, "Cells", m_item.cellList());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NeighborItemPrinter::
_printSubItems(std::ostream& ostr, Integer level, Integer levelmax,
               ItemVectorView sub_items, const char* name)
{
  indent(ostr, levelmax - level) << String::plural(sub_items.size(), name) << ":\n";
  for (Item sub_item : sub_items) {
    indent(ostr, levelmax - level) << "\t" << name;
    print(ostr, sub_item, level - 1, levelmax);
    ostr << "\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NeighborItemPrinter::
print(std::ostream& ostr, Item xitem, Integer level, Integer levelmax)
{
  impl::ItemBase item = xitem.itemBase();
  if (xitem.null()) {
    ostr << "(null Item)";
  }
  else {
    eItemKind ik = item.kind();
    ostr << "(";
    ItemPrinter::Internal::_printBasics(ostr, item);
    ItemPrinter::Internal::_printFlags(ostr, item.flags());
    ItemPrinter::Internal::_printParents(ostr, item);
    ItemPrinter::Internal::_printErrors(ostr, item);
    ostr << ")";
    if (level > 0) {
      ostr << "\n";
      if (ik != IK_Node)
        if (item.nbNode() != 0)
          _printSubItems(ostr, level, levelmax, item.nodeList(), "node");
      if (item.nbEdge() != 0)
        _printSubItems(ostr, level, levelmax, item.edgeList(), "edge");
      if (item.nbFace() != 0)
        _printSubItems(ostr, level, levelmax, item.faceList(), "face");
      if (ik != IK_Cell)
        if (item.nbCell() != 0)
          _printSubItems(ostr, level, levelmax, item.cellList(), "cell");
    }
  }
}

std::ostream& NeighborItemPrinter::
indent(std::ostream& ostr, Integer level)
{
  for (Integer l = 0; l < level; ++l)
    ostr << "\t";
  return ostr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPrinter::Internal::
_printBasics(std::ostream& o, impl::ItemBase item)
{
  o << "uid=" << item.uniqueId()
    << ",lid=" << item.localId()
    << ",owner=" << item.owner()
    << ",type=" << item.typeInfo()->typeName()
    << ",kind=" << itemKindName(item.kind());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPrinter::Internal::
_printFlags(std::ostream& o, Integer flags)
{
  Integer position = 0;
  o << ",flags=";
  if (flags & ItemFlags::II_Boundary)
    o << ((position++) ? "|" : "") << "Boundary";
  if (flags & ItemFlags::II_HasFrontCell)
    o << ((position++) ? "|" : "") << "HasFrontCell";
  if (flags & ItemFlags::II_HasBackCell)
    o << ((position++) ? "|" : "") << "HasBackCell";
  if (flags & ItemFlags::II_FrontCellIsFirst)
    o << ((position++) ? "|" : "") << "FrontCellIsFirst";
  if (flags & ItemFlags::II_BackCellIsFirst)
    o << ((position++) ? "|" : "") << "BackCellIsFirst";
  if (flags & ItemFlags::II_Own)
    o << ((position++) ? "|" : "") << "Own";
  if (flags & ItemFlags::II_Added)
    o << ((position++) ? "|" : "") << "Added";
  if (flags & ItemFlags::II_Suppressed)
    o << ((position++) ? "|" : "") << "Suppressed";
  if (flags & ItemFlags::II_Shared)
    o << ((position++) ? "|" : "") << "Shared";
  if (flags & ItemFlags::II_SubDomainBoundary)
    o << ((position++) ? "|" : "") << "SubDomainBoundary";
  //   if (flags & ItemFlags::II_JustRemoved)
  //     o << ((position++)?"|":"") << "JustRemoved";
  if (flags & ItemFlags::II_JustAdded)
    o << ((position++) ? "|" : "") << "JustAdded";
  if (flags & ItemFlags::II_NeedRemove)
    o << ((position++) ? "|" : "") << "NeedRemove";
  if (flags & ItemFlags::II_SlaveFace)
    o << ((position++) ? "|" : "") << "SlaveFace";
  if (flags & ItemFlags::II_MasterFace)
    o << ((position++) ? "|" : "") << "MasterFace";
  if (flags & ItemFlags::II_Detached)
    o << ((position++) ? "|" : "") << "Detached";
  if (flags & ItemFlags::II_HasEdgeFor1DItems)
    o << ((position++) ? "|" : "") << "HasEdgeFor1DItems";
  if (flags & ItemFlags::II_Coarsen)
    o << ((position++) ? "|" : "") << "Coarsen";
  if (flags & ItemFlags::II_DoNothing)
    o << ((position++) ? "|" : "") << "DoNothing";
  if (flags & ItemFlags::II_Refine)
    o << ((position++) ? "|" : "") << "Refine";
  if (flags & ItemFlags::II_JustRefined)
    o << ((position++) ? "|" : "") << "JustRefined";
  if (flags & ItemFlags::II_JustCoarsened)
    o << ((position++) ? "|" : "") << "JustCoarsened";
  if (flags & ItemFlags::II_Inactive)
    o << ((position++) ? "|" : "") << "Inactive";
  if (flags & ItemFlags::II_CoarsenInactive)
    o << ((position++) ? "|" : "") << "CoarsenInactive";
  if (flags & ItemFlags::II_UserMark1)
    o << ((position++) ? "|" : "") << "UserMark1";
  if (flags & ItemFlags::II_UserMark2)
    o << ((position++) ? "|" : "") << "UserMark2";
  if (position == 0)
    o << "0";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPrinter::Internal::
_printParents(std::ostream& o, ItemBase item)
{
  if (item.nbParent() > 0) {
    ItemBase parent = item.parentBase(0);
    o << ",parent uid/lid="
      << parent.uniqueId()
      << "/"
      << parent.localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPrinter::Internal::
_printErrors(std::ostream& o, ItemBase item)
{
  if (item.isSuppressed())
    return;
  const char* str = ", error=";
  Integer nerror = 0;
  Int32 mesh_rank = item.family()->mesh()->meshPartInfo().partRank();
  if (item.isOwn() != (mesh_rank == item.owner()))
    o << ((nerror++ == 0) ? str : "|") << "WRONG ISOWN";
  if (item.nbParent() > 0) {
    // Les erreurs sont ici exclusives car invalidantes pour l'item
    ItemBase parent = item.parentBase(0);
    if (parent.uniqueId() != item.uniqueId())
      o << ((nerror++ == 0) ? str : "|") << "PARENT UID MISMATCH";
    else if (parent.isSuppressed())
      o << ((nerror++ == 0) ? str : "|") << "SUPPRESSED PARENT";
    else if (parent.owner() != item.owner())
      o << ((nerror++ == 0) ? str : "|") << "PARENT OWNER MISMATCH";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPrinter::Internal::
_printItemSubItems(std::ostream& ostr, String name, const ItemVectorView& enumerator)
{
  ostr << " " << name << " count=" << enumerator.size();
  ostr << " (uids=";
  for (Item item : enumerator) {
    if (item.localId() != NULL_ITEM_ID) {
      if (!item.null())
        ostr << " " << item.uniqueId();
      else
        ostr << " (null)";
    }
    else
      ostr << " (null)";
  }
  ostr << ", lids=";
  for (Item item : enumerator) {
    if (item.localId() != NULL_ITEM_ID) {
      if (!item.null())
        ostr << " " << item.localId();
      else
        ostr << " (null)";
    }
    else
      ostr << " (null)";
  }
  ostr << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
