// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupsSerializer2.h                                     (C) 2000-2016 */
/*                                                                           */
/* Serialization of entity groups.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMGROUPSSERIALIZER2_H
#define ARCANE_MESH_ITEMGROUPSSERIALIZER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemGroup.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IMesh;
class SerializeBuffer;
class IParallelExchanger;
class ItemFamilySerializeArgs;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Serializes the entities of the groups.
 */
class ItemGroupsSerializer2
: public TraceAccessor
{
 public:

  ItemGroupsSerializer2(IItemFamily* item_family, IParallelExchanger* exchanger);
  virtual ~ItemGroupsSerializer2();

 public:

  void prepareData(ConstArrayView<SharedArray<Int32>> items_exchange);
  void serialize(const ItemFamilySerializeArgs& args);
  void get(ISerializer* sbuf, Int64Array& items_in_groups_uid);

  ItemGroupList groups() { return m_groups_to_exchange; }
  IMesh* mesh() const { return m_mesh; }
  //eItemKind itemKind() const { return m_item_kind; }
  IItemFamily* itemFamily() const { return m_item_family; }

 protected:
 private:

  IParallelExchanger* m_exchanger;
  IMesh* m_mesh;
  IItemFamily* m_item_family;
  /*! \brief List of groups to exchange.
    
    IMPORTANT: This list must be identical for all sub-domains
    otherwise the deserializations will give incorrect results.
  */
  ItemGroupList m_groups_to_exchange;
  //! List of entities to exchange per processor
  UniqueArray<SharedArray<Int64>> m_items_to_send;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
