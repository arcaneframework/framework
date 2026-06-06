// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemCompatibility.h                                         (C) 2000-2023 */
/*                                                                           */
/* Methods ensuring compatibility between Item versions.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMCOMPATIBILITY_H
#define ARCANE_ITEMCOMPATIBILITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// NOTE: This file is included directly by 'Item.h' and should not
// be included directly by other files.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace Materials
{
  class ComponentItemInternal;
}
namespace geometric
{
  class GeomShapeView;
}
namespace mesh
{
  class CellMerger;
  class ItemFamily;
  class ItemTools;
  class FaceFamily;
  class OneMeshItemAdder;
  class ParallelAMRConsistency;
  class MeshRefinement;
}
namespace AnyItem
{
  class Group;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Methods for conversions between different entity management classes
 * of entities
 *
 * This class is temporary and internal to Arcane. Only 'friend' classes
 * can use it.
 */
class ItemCompatibility
{
  // For accessing _internal()
  friend Materials::ComponentItemInternal;
  friend class ItemSharedInfo;
  friend class IItemFamilyModifier;
  friend geometric::GeomShapeView;
  friend class TotalviewAdapter;
  friend mesh::CellMerger;
  friend mesh::ItemFamily;
  friend mesh::ItemTools;
  friend mesh::FaceFamily;
  friend mesh::ParallelAMRConsistency;
  friend mesh::OneMeshItemAdder;
  friend mesh::MeshRefinement;
  friend AnyItem::Group;

 private:

  static ItemInternal* _itemInternal(const Item& item)
  {
    return item._internal();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
