// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Mesh.cpp                                        (C) 2000-2026             */
/*                                                                           */
/* Asynchronous Mesh structure based on Neo kernel                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <stdexcept>
#include <fstream>

#include "Mesh.h"
#include "Neo.h"
#include "MeshKernel.h"

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/

Neo::Mesh::Mesh(const std::string& mesh_name)
: m_mesh_graph(std::make_unique<Neo::MeshKernel::AlgorithmPropertyGraph>(Neo::MeshKernel::AlgorithmPropertyGraph{ mesh_name })) {
}

/*-----------------------------------------------------------------------------*/

Neo::Mesh::Mesh(const std::string& mesh_name, int mesh_rank)
: m_mesh_graph(std::make_unique<Neo::MeshKernel::AlgorithmPropertyGraph>(Neo::MeshKernel::AlgorithmPropertyGraph{ mesh_name, mesh_rank }))
, m_rank(mesh_rank){
  // Remove existing output file if exists
  NeoOutputStream output_stream{Trace::VerboseInFile,m_rank};
  std::ofstream new_file{output_stream.fileName()};
  new_file.close();
}

/*-----------------------------------------------------------------------------*/

Neo::Mesh::~Mesh() = default;

/*-----------------------------------------------------------------------------*/

std::string const& Neo::Mesh::name() const noexcept {
  return m_mesh_graph->m_name;
}

/*-----------------------------------------------------------------------------*/

Neo::Family& Neo::Mesh::findFamily(Neo::ItemKind family_kind,
                                   std::string const& family_name) const noexcept(ndebug) {
  return m_families.operator()(family_kind, family_name);
}

/*-----------------------------------------------------------------------------*/

std::string Neo::Mesh::uniqueIdPropertyName(std::string const& family_name) noexcept {
  return family_name + "_uids";
}

std::string Neo::Mesh::_connectivityOrientationPropertyName(std::string const& source_family_name, std::string const& target_family_name) {
  return source_family_name + "to" + target_family_name + "_connectivity_orientation";
}

/*-----------------------------------------------------------------------------*/

Neo::Family& Neo::Mesh::addFamily(Neo::ItemKind item_kind, std::string family_name) noexcept {
  Neo::printer(m_rank) << "= Add Family " << family_name << " in mesh " << name() << Neo::endline;
  auto& item_family = m_families.push_back(item_kind, family_name);
  item_family.addMeshScalarProperty<Neo::utils::Int64>(uniqueIdPropertyName(family_name));
  return item_family;
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleAddItems(Neo::Family& family, std::vector<Neo::utils::Int64> uids, Neo::FutureItemRange& added_item_range) noexcept {
  ItemRange& added_items = added_item_range;
  // Add items
  m_mesh_graph->addAlgorithm(Neo::MeshKernel::OutProperty{ family, family.lidPropName() },
                             [&family, uids, &added_items, rank = m_rank](Neo::ItemLidsProperty& lids_property) {
                               Neo::printer(rank) << "== Algorithm: create items in family " << family.name() << Neo::endline;
                               added_items = lids_property.append(uids);
                               lids_property.debugPrint(rank);
                               Neo::printer(rank) << "Inserted item range : " << added_items;
                             });
  // register their uids
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ family, family.lidPropName() },
  Neo::MeshKernel::OutProperty{ family, uniqueIdPropertyName(family.name()) },
  [&family, uids{ std::move(uids) }, &added_items, rank = m_rank]([[maybe_unused]] Neo::ItemLidsProperty const& item_lids_property,
                                                   Neo::MeshScalarPropertyT<Neo::utils::Int64>& item_uids_property) {
    Neo::printer(rank) << "== Algorithm: register item uids for family " << family.name() << Neo::endline;
    if (item_uids_property.isInitializableFrom(added_items)) {
      item_uids_property.init(added_items, std::move(uids)); // init can steal the input values
    }
    else {
      item_uids_property.append(added_items, uids);
    }
    item_uids_property.debugPrint(rank);
  }); // need to add a property check for existing uid
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::_addConnectivityOrientationCheck(Neo::Family& source_family, Neo::Family const& target_family) {
  std::string orientation_property_name = _connectivityOrientationPropertyName(source_family.name(), target_family.name());
  std::string orientation_check_property_name = orientation_property_name + "_check";
  source_family.addMeshScalarProperty<int>(orientation_check_property_name);
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ source_family, orientation_property_name },
  Neo::MeshKernel::InProperty{ source_family, source_family.m_name + "_uids" },
  Neo::MeshKernel::OutProperty{ source_family, orientation_check_property_name },
  [&source_family, &target_family, rank=m_rank](
  Neo::MeshArrayPropertyT<int> const& item_orientation,
  Neo::MeshScalarPropertyT<Neo::utils::Int64> const& item_uids,
  Neo::MeshScalarPropertyT<int>& item_orientation_check) {
    Neo::printer(rank) << "== Algorithm: check orientation in connectivity between "
                 << source_family.name() << "  and  " << target_family.name()
                 << Neo::endline;
    item_orientation_check.init(source_family.all(), 1);
    std::ostringstream exception_info;
    exception_info << "Connectivity orientation false for items\n";
    auto has_error = false;
    for (auto item : source_family.all()) {
      auto orientation = item_orientation[item];
      if (std::abs(std::accumulate(orientation.begin(), orientation.end(), 0)) > 1) {
        item_orientation_check[item] = 0;
        exception_info << item_uids[item] << " ";
        has_error = true;
      }
    }
    if (has_error)
      throw std::runtime_error(exception_info.str());
  });
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleSetItemCoords(Neo::Family& item_family, Neo::FutureItemRange& future_added_item_range, std::vector<Neo::utils::Real3> item_coords) noexcept {
  auto coord_prop_name = _itemCoordPropertyName(item_family);
  item_family.addMeshScalarProperty<Neo::utils::Real3>(coord_prop_name);
  ItemRange& added_items = future_added_item_range;
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ item_family, item_family.lidPropName(), Neo::PropertyStatus::ExistingProperty }, // TODO handle property status in Property Holder constructor
  Neo::MeshKernel::OutProperty{ item_family, coord_prop_name },
  [item_coords{ std::move(item_coords) }, &added_items, rank=m_rank](Neo::ItemLidsProperty const& item_lids_property,
                                                        Neo::MeshScalarPropertyT<Neo::utils::Real3>& item_coords_property) {
    Neo::printer(rank) << "== Algorithm: register item coords" << Neo::endline;
    if (item_coords_property.isInitializableFrom(added_items)) {
      item_coords_property.init(
      added_items,
      std::move(item_coords)); // init can steal the input values
    }
    else {
      item_coords_property.append(added_items, item_coords);
    }
    item_coords_property.debugPrint(rank);
  });
}

/*-----------------------------------------------------------------------------*/

Neo::EndOfMeshUpdate Neo::Mesh::applyScheduledOperations() {
  return m_mesh_graph->applyAlgorithms();
}

/*-----------------------------------------------------------------------------*/

Neo::Mesh::CoordPropertyType& Neo::Mesh::getItemCoordProperty(Neo::Family& family) {
  return family.getConcreteProperty<CoordPropertyType>(_itemCoordPropertyName(family));
}

/*-----------------------------------------------------------------------------*/

Neo::Mesh::Connectivity Neo::Mesh::getConnectivity(Neo::Family const& source_family, Neo::Family const& target_family, std::string const& connectivity_name) const {
  auto connectivity_iter = m_connectivities.find(connectivity_name);
  if (connectivity_iter == m_connectivities.end())
    throw std::invalid_argument("Cannot find Connectivity " + connectivity_name);
  else if (connectivity_iter->second.source_family != source_family || connectivity_iter->second.target_family != target_family) {
    throw std::invalid_argument("Error in getConnectivity. The Connectivity " + connectivity_name + " does not connect the family " + source_family.name() + " to " + target_family.name() + " it connects family " + connectivity_iter->second.source_family.name() + " to family " + connectivity_iter->second.target_family.name());
  }
  return connectivity_iter->second;
  // todo check source and target family type...(add operator== on family)
}

/*-----------------------------------------------------------------------------*/

Neo::utils::ConstSpan<Neo::Mesh::Connectivity> Neo::Mesh::getConnectivities(Neo::Family const& source_family) const {
  auto connectivity_iter = m_connectivities_per_family.find({source_family.itemKind(),source_family.name()});
  if (connectivity_iter != m_connectivities_per_family.end()) {
    return utils::ConstSpan<Neo::Mesh::Connectivity>{connectivity_iter->second.data(), connectivity_iter->second.size()};
  }
  else return utils::ConstSpan<Connectivity>{};
}

/*-----------------------------------------------------------------------------*/

Neo::Mesh::CoordPropertyType const& Neo::Mesh::getItemCoordProperty(Neo::Family const& family) const {
  return family.getConcreteProperty<CoordPropertyType>(_itemCoordPropertyName(family));
}

/*-----------------------------------------------------------------------------*/

std::vector<Neo::Mesh::Connectivity> Neo::Mesh::items(Neo::Family const& source_family, Neo::ItemKind item_kind) const noexcept {
  std::vector<Connectivity> connectivities_vector{};
  std::transform(m_connectivities.begin(), m_connectivities.end(), std::back_inserter(connectivities_vector),
                 [](auto& name_connectivity_pair) { return name_connectivity_pair.second; });
  std::vector<Connectivity> item_connectivities_vector{};
  std::copy_if(connectivities_vector.begin(), connectivities_vector.end(), std::back_inserter(item_connectivities_vector),
               [&source_family, item_kind](auto& connectivity) {
                 return (connectivity.source_family == source_family &&
                         connectivity.target_family.itemKind() == item_kind);
               });
  return item_connectivities_vector;
}

/*-----------------------------------------------------------------------------*/

std::vector<Neo::Mesh::Connectivity> Neo::Mesh::edges(Neo::Family const& source_family) const noexcept {
  return items(source_family, Neo::ItemKind::IK_Edge);
}

/*-----------------------------------------------------------------------------*/

std::vector<Neo::Mesh::Connectivity> Neo::Mesh::nodes(Neo::Family const& source_family) const noexcept {
  return items(source_family, Neo::ItemKind::IK_Node);
}
/*-----------------------------------------------------------------------------*/

std::vector<Neo::Mesh::Connectivity> Neo::Mesh::faces(Neo::Family const& source_family) const noexcept {
  return items(source_family, Neo::ItemKind::IK_Face);
}

/*-----------------------------------------------------------------------------*/

std::vector<Neo::Mesh::Connectivity> Neo::Mesh::cells(Neo::Family const& source_family) const noexcept {
  return items(source_family, Neo::ItemKind::IK_Cell);
}

/*-----------------------------------------------------------------------------*/

std::vector<Neo::Mesh::Connectivity> Neo::Mesh::dofs(Neo::Family const& source_family) const noexcept {
  return items(source_family, Neo::ItemKind::IK_Dof);
}

/*-----------------------------------------------------------------------------*/

std::vector<Neo::utils::Int64> Neo::Mesh::uniqueIds(Neo::Family const& item_family,
                                                    std::vector<Neo::utils::Int32> const& item_lids) const noexcept {
  auto const& uid_property = getItemUidsProperty(item_family);
  return uid_property[item_lids];
}

/*-----------------------------------------------------------------------------*/
std::vector<Neo::utils::Int64> Neo::Mesh::uniqueIds(Neo::Family const& item_family,
                                                    Neo::utils::Int32ConstSpan item_lids) const noexcept {
  auto const& uid_property = getItemUidsProperty(item_family);
  return uid_property[item_lids];
}

/*-----------------------------------------------------------------------------*/

const Neo::Mesh::UniqueIdPropertyType& Neo::Mesh::getItemUidsProperty(const Neo::Family& item_family) const noexcept {
  return item_family.getConcreteProperty<UniqueIdPropertyType>(uniqueIdPropertyName(item_family.name()));
}

/*-----------------------------------------------------------------------------*/

std::vector<Neo::utils::Int32> Neo::Mesh::localIds(Neo::Family const& item_family,
                                                   std::vector<Neo::utils::Int64> const& item_uids) const noexcept {
  return item_family.itemUniqueIdsToLocalids(item_uids);
}

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
