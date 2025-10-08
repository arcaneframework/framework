// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshSchedule2.cpp                                           (C) 2000-2025 */
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

void Neo::Mesh::scheduleMoveItems(Neo::Family& item_family, std::vector<Neo::utils::Int64> moved_item_uids, std::vector<Neo::utils::Real3> moved_item_new_coords) {
  auto coord_prop_name = _itemCoordPropertyName(item_family);
  item_family.addMeshScalarProperty<Neo::utils::Real3>(coord_prop_name);
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ item_family, item_family.lidPropName(), Neo::PropertyStatus::ExistingProperty },
  Neo::MeshKernel::OutProperty{ item_family, coord_prop_name },
  [moved_item_uids = std::move(moved_item_uids),
        moved_item_new_coords = std::move(moved_item_new_coords),rank(m_rank)]
        (Neo::ItemLidsProperty const& item_lids_property,
        Neo::MeshScalarPropertyT<Neo::utils::Real3>& item_coords_property) {
    Neo::print(rank) << "== Algorithm: move items" << std::endl;
    // get range from uids and append
    auto moved_item_range = Neo::ItemRange{ Neo::ItemLocalIds::getIndexes(item_lids_property[moved_item_uids]) };
    item_coords_property.append(moved_item_range, moved_item_new_coords);
    item_coords_property.debugPrint(rank);
  });
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleAddMeshOperation(Neo::Family& input_property_family,
                                         std::string const& input_property_name,
                                         Neo::Family& output_property_family,
                                         std::string const& output_property_name,
                                         Neo::Mesh::MeshOperation mesh_operation) {
  std::visit([&](auto& algorithm) {
    m_mesh_graph->addAlgorithm(Neo::MeshKernel::InProperty{ input_property_family, input_property_name },
                               Neo::MeshKernel::OutProperty{ output_property_family, output_property_name },
                               algorithm);
  },
             mesh_operation);
}

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
