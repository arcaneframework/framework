﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshSchedule2.cpp                                           (C) 2000-2023 */
/*                                                                           */
/* Asynchronous Mesh structure based on Neo kernel                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <stdexcept>

#include "Mesh.h"
#include "Neo.h"
#include "MeshKernel.h"

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleRemoveItems(Neo::Family& family, std::vector<Neo::utils::Int64> const& removed_item_uids) {
  const std::string removed_item_property_name = _removeItemPropertyName(family);
  // Add an algo to clear removed_items property at the beginning of a mesh update
  // This algo will be executed before remove item algo
  const std::string ok_to_start_remove_property_name = "ok_to_start_remove_property";
  family.addMeshScalarProperty<Neo::utils::Int32>(ok_to_start_remove_property_name);
  family.addMeshScalarProperty<Neo::utils::Int32>(removed_item_property_name);
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::OutProperty{ family, removed_item_property_name },
  Neo::MeshKernel::OutProperty{ family, ok_to_start_remove_property_name },
  [&family](Neo::MeshScalarPropertyT<Neo::utils::Int32>& removed_item_property,
            Neo::MeshScalarPropertyT<Neo::utils::Int32>& ok_to_start_remove_property) {
    Neo::print() << "Algorithm : clear remove item property for family " << family.name() << std::endl;
    removed_item_property.init(family.all(), 0);
    ok_to_start_remove_property.init(family.all(), 1);
  });
  // Remove item algo
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ family, ok_to_start_remove_property_name },
  Neo::MeshKernel::OutProperty{ family, family.lidPropName() },
  Neo::MeshKernel::OutProperty{ family, removed_item_property_name },
  [removed_item_uids, &family](
  Neo::MeshScalarPropertyT<Neo::utils::Int32> const&, // ok_to_start_remove_property
  Neo::ItemLidsProperty& item_lids_property,
  Neo::MeshScalarPropertyT<Neo::utils::Int32>& removed_item_property) {
    Neo::print() << "Algorithm: remove items in " << family.name() << std::endl;
    auto removed_items = item_lids_property.remove(removed_item_uids);
    item_lids_property.debugPrint();
    Neo::print() << "removed item range : " << removed_items;
    // Store removed items in internal_end_of_remove_tag
    removed_item_property.init(family.all(), 0);
    for (auto removed_item : removed_items) {
      removed_item_property[removed_item] = 1;
    }
  });
}


/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleRemoveItems(Neo::Family& family, Neo::ItemRange const& removed_items) {
  auto unique_ids = uniqueIds(family, removed_items.localIds());
  scheduleRemoveItems(family, unique_ids);
}

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
