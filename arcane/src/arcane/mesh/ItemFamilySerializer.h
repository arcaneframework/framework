// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilySerializer.h                                      (C) 2000-2018 */
/*                                                                           */
/* Unique Serializer valid for any item family.                              */
/* Requires the use of the family graph: ItemFamilyNetwork                   */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_SRC_ARCANE_MESH_ITEMFAMILYSERIALIZER_H_
#define ARCANE_SRC_ARCANE_MESH_ITEMFAMILYSERIALIZER_H_

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/core/IItemFamilySerializer.h"
#include "arcane/core/IItemFamilyModifier.h"

#include "arcane/mesh/ItemData.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h" // Todo replace this by IMeshModifier

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Serialization/Deserialization of item families.
 *
 * This serializer implementation uses the IItemFamilyNetwork family graph
 * to function. This graph allows exchanging each item family independently,
 * without using the concept of direct or indirect serialization. The connectivity information
 * for each family is stored in the ItemData class, which will then be serialized/deserialized.
 */
class ARCANE_MESH_EXPORT ItemFamilySerializer : public IItemFamilySerializer
{
 public:

  ItemFamilySerializer(IItemFamily* family, IItemFamilyModifier* family_modifier, DynamicMeshIncrementalBuilder* mesh_builder)
  : m_family(family)
  , m_family_modifier(family_modifier)
  , m_mesh_builder(mesh_builder)
  {
    if (!family->mesh()->itemFamilyNetwork())
      throw FatalErrorException("Cannot create ItemFamilySerializer if IItemFamilyNetwork is not defined. Exiting.");
  }
  ~ItemFamilySerializer() {}

 public:

  void serializeItems(ISerializer* buf, Int32ConstArrayView local_ids) override;
  void deserializeItems(ISerializer* buf, Int32Array* local_ids) override;

  void serializeItemRelations(ISerializer* buf, Int32ConstArrayView cells_local_id) override;
  void deserializeItemRelations(ISerializer* buf, Int32Array* cells_local_id) override;

  IItemFamily* family() const override;

 private:

  IItemFamily* m_family;
  IItemFamilyModifier* m_family_modifier;
  DynamicMeshIncrementalBuilder* m_mesh_builder;

 private:

  void _fillItemDependenciesData(ItemData& item_data, Int32ConstArrayView local_ids);
  void _fillItemRelationsData(ItemData& item_data, Int32ConstArrayView local_ids);

  void _deserializeItemsOrRelations(ISerializer* buf, Int32Array* local_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
