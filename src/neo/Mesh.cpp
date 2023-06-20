// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Mesh.cpp                                        (C) 2000-2023             */
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

Neo::Mesh::Mesh(const std::string& mesh_name)
: m_mesh_graph(std::make_unique<Neo::MeshKernel::MeshBase>(Neo::MeshKernel::MeshBase{ mesh_name })) {
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
  Neo::print() << "Add Family " << family_name << " in mesh " << name() << std::endl;
  auto& item_family = m_families.push_back(item_kind, std::move(family_name));
  item_family.addProperty<Neo::utils::Int64>(uniqueIdPropertyName(family_name));
  return item_family;
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleAddItems(Neo::Family& family, std::vector<Neo::utils::Int64> uids, Neo::FutureItemRange& added_item_range) noexcept {
  ItemRange& added_items = added_item_range;
  // Add items
  m_mesh_graph->addAlgorithm(Neo::OutProperty{ family, family.lidPropName() },
                             [&family, uids, &added_items](Neo::ItemLidsProperty& lids_property) {
                               Neo::print() << "Algorithm: create items in family " << family.name() << std::endl;
                               added_items = lids_property.append(uids);
                               lids_property.debugPrint();
                               Neo::print() << "Inserted item range : " << added_items;
                             });
  // register their uids
  m_mesh_graph->addAlgorithm(
  Neo::InProperty{ family, family.lidPropName() },
  Neo::OutProperty{ family, uniqueIdPropertyName(family.name()) },
  [&family, uids{ std::move(uids) }, &added_items]([[maybe_unused]] Neo::ItemLidsProperty const& item_lids_property,
                                                   Neo::PropertyT<Neo::utils::Int64>& item_uids_property) {
    Neo::print() << "Algorithm: register item uids for family " << family.name() << std::endl;
    if (item_uids_property.isInitializableFrom(added_items)) {
      item_uids_property.init(added_items, std::move(uids)); // init can steal the input values
    }
    else {
      item_uids_property.append(added_items, uids);
    }
    item_uids_property.debugPrint();
  }); // need to add a property check for existing uid
}

/*-----------------------------------------------------------------------------*/

template <typename ItemRangeT>
void Neo::Mesh::_scheduleAddConnectivity(Neo::Family& source_family, Neo::ItemRangeWrapper<ItemRangeT> source_items_wrapper,
                                         Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                         std::vector<Neo::utils::Int64> connected_item_uids,
                                         std::string const& connectivity_unique_name,
                                         ConnectivityOperation add_or_modify) {
  // add connectivity property if doesn't exist
  source_family.addArrayProperty<Neo::utils::Int32>(connectivity_unique_name);
  // add orientation property (only used for oriented connectivity)
  std::string orientation_name = _connectivityOrientationPropertyName(source_family.name(), target_family.name());
  source_family.addArrayProperty<Neo::utils::Int32>(orientation_name);
  // Create connectivity wrapper and add it to mesh
  auto& connectivity_property = source_family.getConcreteProperty<Mesh::ConnectivityPropertyType>(connectivity_unique_name);
  auto& connectivity_orientation = source_family.getConcreteProperty<Mesh::ConnectivityPropertyType>(orientation_name);
  auto [iterator, is_inserted] = m_connectivities.insert(std::make_pair(connectivity_unique_name,
                                                                        Connectivity{
                                                                        source_family,
                                                                        target_family,
                                                                        connectivity_unique_name,
                                                                        connectivity_property,
                                                                        connectivity_orientation }));
  if (!is_inserted && add_or_modify == ConnectivityOperation::Add) {
    throw std::invalid_argument("Cannot include already inserted connectivity " + connectivity_unique_name + ". Choose ConnectivityOperation::Modify");
  }
  m_mesh_graph->addAlgorithm(
  Neo::InProperty{ source_family, source_family.lidPropName(), PropertyStatus::ExistingProperty },
  Neo::InProperty{ target_family, target_family.lidPropName(), PropertyStatus::ExistingProperty },
  Neo::OutProperty{ source_family, connectivity_unique_name },
  [connected_item_uids{ std::move(connected_item_uids) },
   nb_connected_item_per_item{ std::move(nb_connected_item_per_item) },
   source_items_wrapper, &source_family, &target_family](Neo::ItemLidsProperty const& source_family_lids_property,
                                                         Neo::ItemLidsProperty const& target_family_lids_property,
                                                         Neo::ArrayProperty<Neo::utils::Int32>& source2target) {
    Neo::print() << "Algorithm: register connectivity between " << source_family.m_name << "  and  " << target_family.m_name << std::endl;
    ItemRange const& source_items = source_items_wrapper.get();
    auto connected_item_lids = target_family_lids_property[connected_item_uids];
    if (source2target.isInitializableFrom(source_items)) {
      source2target.resize(std::move(nb_connected_item_per_item));
      source2target.init(source_items, std::move(connected_item_lids));
    }
    else {
      source2target.append(source_items, connected_item_lids,
                           nb_connected_item_per_item);
    }
    source2target.debugPrint();
  });
  // update connectivity of removed items : this algo must be permanent (not removed by a call to applyScheduledOperations)
  //  const std::string removed_item_property_name{ "removed_" + target_family.m_name + "_items" };
  const std::string removed_item_property_name = _removeItemPropertyName(target_family);
  source_family.addProperty<Neo::utils::Int32>(removed_item_property_name);
  m_mesh_graph->addAlgorithm(
  Neo::InProperty{ target_family, removed_item_property_name },
  Neo::OutProperty{ source_family, connectivity_unique_name },
  [&source_family, &target_family](
  Neo::PropertyT<Neo::utils::Int32> const& target_family_removed_items,
  Neo::ArrayProperty<Neo::utils::Int32>& connectivity) {
    Neo::print() << "Algorithm update connectivity after remove " << connectivity.m_name << std::endl;
    for (auto item : source_family.all()) {
      auto connected_items = connectivity[item];
      for (auto& connected_item : connected_items) {
        if (connected_item != Neo::utils::NULL_ITEM_LID && target_family_removed_items[connected_item] == 1) {
          Neo::print() << "modify connected item : " << connected_item << " in family " << target_family.m_name << std::endl;
          connected_item = Neo::utils::NULL_ITEM_LID;
        }
      }
    }
  },
  Neo::MeshKernel::MeshBase::AlgorithmPersistence::KeepAfterExecution);
}

/*-----------------------------------------------------------------------------*/

template <typename ItemRangeT>
void Neo::Mesh::_scheduleAddConnectivityOrientation(Neo::Family& source_family, Neo::ItemRangeWrapper<ItemRangeT> source_items_wrapper,
                                                    Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                                    std::vector<int> source_item_orientation_in_target_item,
                                                    bool do_check_orientation) {
  // add orientation property if doesn't exist
  std::string orientation_property_name = _connectivityOrientationPropertyName(source_family.name(), target_family.name());
  source_family.addArrayProperty<Neo::utils::Int32>(orientation_property_name);
  m_mesh_graph->addAlgorithm(
  Neo::InProperty{ source_family, source_family.lidPropName() },
  Neo::OutProperty{ source_family, orientation_property_name },
  [source_item_orientation_in_target_item{ std::move(source_item_orientation_in_target_item) },
   nb_connected_item_per_item{ std::move(nb_connected_item_per_item) }, source_items_wrapper,
   &source_family, &target_family](Neo::ItemLidsProperty const& source_family_lids_property,
                                   Neo::ArrayProperty<int>& item_orientation) {
    Neo::print() << "Algorithm: add orientation in connectivity between "
                 << source_family.m_name << "  and  " << target_family.m_name
                 << std::endl;
    ItemRange const& source_items = source_items_wrapper.get();
    if (item_orientation.isInitializableFrom(source_items)) {
      item_orientation.resize(std::move(nb_connected_item_per_item));
      item_orientation.init(source_items, std::move(source_item_orientation_in_target_item));
    }
    else {
      item_orientation.append(source_items, source_item_orientation_in_target_item,
                              nb_connected_item_per_item);
    }
    item_orientation.debugPrint();
  });
  if (do_check_orientation) {
    _addConnectivityOrientationCheck(source_family, target_family);
  }
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::_addConnectivityOrientationCheck(Neo::Family& source_family, Neo::Family const& target_family) {
  std::string orientation_property_name = _connectivityOrientationPropertyName(source_family.name(), target_family.name());
  std::string orientation_check_property_name = orientation_property_name + "_check";
  source_family.addProperty<int>(orientation_check_property_name);
  m_mesh_graph->addAlgorithm(
  Neo::InProperty{ source_family, orientation_property_name },
  Neo::InProperty{ source_family, source_family.m_name + "_uids" },
  Neo::OutProperty{ source_family, orientation_check_property_name },
  [&source_family, &target_family](
  Neo::ArrayProperty<int> const& item_orientation,
  Neo::PropertyT<Neo::utils::Int64> const& item_uids,
  Neo::PropertyT<int>& item_orientation_check) {
    Neo::print() << "Algorithm: check orientation in connectivity between "
                 << source_family.name() << "  and  " << target_family.name()
                 << std::endl;
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

// Todo change the order to match .h declarations

void Neo::Mesh::scheduleAddConnectivity(Neo::Family& source_family, Neo::ItemRange const& source_items,
                                        Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                        std::vector<Neo::utils::Int64> connected_item_uids,
                                        std::string const& connectivity_unique_name,
                                        ConnectivityOperation add_or_modify) {
  _scheduleAddConnectivity(source_family, Neo::ItemRangeWrapper<const ItemRange>{ source_items }, target_family,
                           std::move(nb_connected_item_per_item), std::move(connected_item_uids), connectivity_unique_name, add_or_modify);
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleAddConnectivity(Neo::Family& source_family, Neo::FutureItemRange& source_items,
                                        Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                        std::vector<Neo::utils::Int64> connected_item_uids,
                                        std::string const& connectivity_unique_name,
                                        ConnectivityOperation add_or_modify) {
  _scheduleAddConnectivity(source_family, Neo::ItemRangeWrapper<FutureItemRange>{ source_items },
                           target_family, std::move(nb_connected_item_per_item),
                           std::move(connected_item_uids),
                           connectivity_unique_name, add_or_modify);
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleAddConnectivity(Neo::Family& source_family, Neo::ItemRange const& source_items,
                                        Neo::Family& target_family, int nb_connected_item_per_item,
                                        std::vector<Neo::utils::Int64> connected_item_uids,
                                        std::string const& connectivity_unique_name,
                                        ConnectivityOperation add_or_modify) {
  assert(("source items and connected item uids sizes are not coherent with nb_connected_item_per_item",
          source_items.size() * nb_connected_item_per_item == connected_item_uids.size()));
  std::vector<int> nb_connected_item_per_item_array(source_items.size(), nb_connected_item_per_item);
  scheduleAddConnectivity(source_family, source_items, target_family,
                          std::move(nb_connected_item_per_item_array),
                          std::move(connected_item_uids),
                          connectivity_unique_name, add_or_modify);
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleAddConnectivity(Neo::Family& source_family, Neo::FutureItemRange& source_items,
                                        Neo::Family& target_family, int nb_connected_item_per_item,
                                        std::vector<Neo::utils::Int64> connected_item_uids,
                                        std::string const& connectivity_unique_name,
                                        ConnectivityOperation add_or_modify) {
  assert(("Connected item uids size is not compatible with nb_connected_item_per_item",
          connected_item_uids.size() % nb_connected_item_per_item == 0));
  auto source_item_size = connected_item_uids.size() / nb_connected_item_per_item;
  std::vector<int> nb_connected_item_per_item_array(source_item_size, nb_connected_item_per_item);
  _scheduleAddConnectivity(source_family, Neo::ItemRangeWrapper<FutureItemRange>{ source_items }, target_family,
                           std::move(nb_connected_item_per_item_array),
                           std::move(connected_item_uids),
                           connectivity_unique_name, add_or_modify);
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleAddOrientedConnectivity(Neo::Family& source_family, Neo::FutureItemRange& source_items,
                                                Neo::Family& target_family, int nb_connected_item_per_item,
                                                std::vector<Neo::utils::Int64> connected_item_uids,
                                                const std::string& connectivity_unique_name,
                                                std::vector<Neo::utils::Int32> source_item_orientation_in_target_item,
                                                Neo::Mesh::ConnectivityOperation add_or_modify, bool do_check_orientation) {
  auto source_item_size = connected_item_uids.size() / nb_connected_item_per_item;
  std::vector<Neo::utils::Int32> nb_connected_item_per_item_array(source_item_size, nb_connected_item_per_item);
  scheduleAddOrientedConnectivity(source_family, source_items,
                                  target_family, std::move(nb_connected_item_per_item_array),
                                  std::move(connected_item_uids), connectivity_unique_name,
                                  std::move(source_item_orientation_in_target_item), add_or_modify, do_check_orientation);
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleAddOrientedConnectivity(Neo::Family& source_family, Neo::ItemRange const& source_items,
                                                Neo::Family& target_family, int nb_connected_item_per_item,
                                                std::vector<Neo::utils::Int64> connected_item_uids,
                                                std::string const& connectivity_unique_name,
                                                std::vector<Neo::utils::Int32> source_item_orientation_in_target_item,
                                                Neo::Mesh::ConnectivityOperation add_or_modify,
                                                bool do_check_orientation) {
  auto source_item_size = connected_item_uids.size() / nb_connected_item_per_item;
  std::vector<Neo::utils::Int32> nb_connected_item_per_item_array(source_item_size, nb_connected_item_per_item);
  scheduleAddOrientedConnectivity(source_family, source_items,
                                  target_family, std::move(nb_connected_item_per_item_array),
                                  std::move(connected_item_uids), connectivity_unique_name,
                                  std::move(source_item_orientation_in_target_item), add_or_modify, do_check_orientation);
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleAddOrientedConnectivity(Neo::Family& source_family, Neo::FutureItemRange& source_items,
                                                Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                                std::vector<Neo::utils::Int64> connected_item_uids, std::string const& connectivity_unique_name,
                                                std::vector<Neo::utils::Int32> source_item_orientation_in_target_item,
                                                ConnectivityOperation add_or_modify, bool do_check_orientation) {
  scheduleAddConnectivity(source_family, source_items, target_family,
                          nb_connected_item_per_item, // cannot move nb_connected_item_per_item, is needed for connectivityOrientation
                          std::move(connected_item_uids), connectivity_unique_name, add_or_modify);
  _scheduleAddConnectivityOrientation(source_family, Neo::ItemRangeWrapper<FutureItemRange>{ source_items },
                                      target_family, std::move(nb_connected_item_per_item),
                                      std::move(source_item_orientation_in_target_item),
                                      do_check_orientation);
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleAddOrientedConnectivity(Neo::Family& source_family, Neo::ItemRange const& source_items,
                                                Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                                std::vector<Neo::utils::Int64> connected_item_uids,
                                                std::string const& connectivity_unique_name,
                                                std::vector<int> source_item_orientation_in_target_item,
                                                ConnectivityOperation add_or_modify,
                                                bool do_check_orientation) {
  scheduleAddConnectivity(source_family, source_items, target_family,
                          std::move(nb_connected_item_per_item),
                          std::move(connected_item_uids), connectivity_unique_name, add_or_modify);
  _scheduleAddConnectivityOrientation(source_family, Neo::ItemRangeWrapper<const ItemRange>{ source_items },
                                      target_family, std::move(nb_connected_item_per_item),
                                      std::move(source_item_orientation_in_target_item),
                                      do_check_orientation);
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleSetItemCoords(Neo::Family& item_family, Neo::FutureItemRange& future_added_item_range, std::vector<Neo::utils::Real3> item_coords) noexcept {
  auto coord_prop_name = _itemCoordPropertyName(item_family);
  item_family.addProperty<Neo::utils::Real3>(coord_prop_name);
  ItemRange& added_items = future_added_item_range;
  m_mesh_graph->addAlgorithm(
  Neo::InProperty{ item_family, item_family.lidPropName(), Neo::PropertyStatus::ExistingProperty }, // TODO handle property status in Property Holder constructor
  Neo::OutProperty{ item_family, coord_prop_name },
  [item_coords{ std::move(item_coords) }, &added_items](Neo::ItemLidsProperty const& item_lids_property,
                                                        Neo::PropertyT<Neo::utils::Real3>& item_coords_property) {
    Neo::print() << "Algorithm: register item coords" << std::endl;
    if (item_coords_property.isInitializableFrom(added_items)) {
      item_coords_property.init(
      added_items,
      std::move(item_coords)); // init can steal the input values
    }
    else {
      item_coords_property.append(added_items, item_coords);
    }
    item_coords_property.debugPrint();
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

std::vector<Neo::utils::Int64> Neo::Mesh::uniqueIds(Neo::Family const& item_family,
                                                    std::vector<Neo::utils::Int32> const& item_lids) const noexcept {
  auto const& uid_property = getItemUidsProperty(item_family);
  return uid_property[item_lids];
}

/*-----------------------------------------------------------------------------*/

const Neo::Mesh::UidPropertyType& Neo::Mesh::getItemUidsProperty(const Neo::Family& item_family) const noexcept {
  return item_family.getConcreteProperty<UidPropertyType>(uniqueIdPropertyName(item_family.name()));
}

/*-----------------------------------------------------------------------------*/

std::vector<Neo::utils::Int32> Neo::Mesh::localIds(Neo::Family const& item_family,
                                                   std::vector<Neo::utils::Int64> const& item_uids) const noexcept {
  return item_family.itemUniqueIdsToLocalids(item_uids);
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleMoveItems(Neo::Family& item_family, std::vector<Neo::utils::Int64> const& moved_item_uids, std::vector<Neo::utils::Real3> const& moved_item_new_coords) {
  auto coord_prop_name = _itemCoordPropertyName(item_family);
  item_family.addProperty<Neo::utils::Real3>(coord_prop_name);
  m_mesh_graph->addAlgorithm(
  Neo::InProperty{ item_family, item_family.lidPropName(), Neo::PropertyStatus::ExistingProperty },
  Neo::OutProperty{ item_family, coord_prop_name },
  [&moved_item_uids, moved_item_new_coords](Neo::ItemLidsProperty const& item_lids_property,
                                            Neo::PropertyT<Neo::utils::Real3>& item_coords_property) {
    Neo::print() << "Algorithm: move items" << std::endl;
    // get range from uids and append
    auto moved_item_range = Neo::ItemRange{ Neo::ItemLocalIds::getIndexes(item_lids_property[moved_item_uids]) };
    item_coords_property.append(moved_item_range, moved_item_new_coords);
    item_coords_property.debugPrint();
  });
}

/*-----------------------------------------------------------------------------*/

void Neo::Mesh::scheduleRemoveItems(Neo::Family& family, std::vector<Neo::utils::Int64> const& removed_item_uids) {
  const std::string removed_item_property_name = _removeItemPropertyName(family);
  // Add an algo to clear removed_items property at the beginning of a mesh update
  // This algo will be executed before remove item algo
  const std::string ok_to_start_remove_property_name = "ok_to_start_remove_property";
  family.addProperty<Neo::utils::Int32>(ok_to_start_remove_property_name);
  family.addProperty<Neo::utils::Int32>(removed_item_property_name);
  m_mesh_graph->addAlgorithm(
  Neo::OutProperty{ family, removed_item_property_name },
  Neo::OutProperty{ family, ok_to_start_remove_property_name },
  [&family](Neo::PropertyT<Neo::utils::Int32>& removed_item_property,
            Neo::PropertyT<Neo::utils::Int32>& ok_to_start_remove_property) {
    Neo::print() << "Algorithm : clear remove item property for family " << family.name() << std::endl;
    removed_item_property.init(family.all(), 0);
    ok_to_start_remove_property.init(family.all(), 1);
  });
  // Remove item algo
  m_mesh_graph->addAlgorithm(
  Neo::InProperty{ family, ok_to_start_remove_property_name },
  Neo::OutProperty{ family, family.lidPropName() },
  Neo::OutProperty{ family, removed_item_property_name },
  [removed_item_uids, &family](
  Neo::PropertyT<Neo::utils::Int32> const&, // ok_to_start_remove_property
  Neo::ItemLidsProperty& item_lids_property,
  Neo::PropertyT<Neo::utils::Int32>& removed_item_property) {
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
// design meshOperation
//void Neo::Mesh::addOperation(Operation::StartingPoint when, Operation::Input input, Operation::Output output, IAlgorithm1Input1OutputPtr algo_ptr, Family& item_family) {
//  m_mesh_graph->addAlgorithm(Neo::InProperty{ item_family, item_family.lidPropName() },
//                             Neo::OutProperty{ item_family, "node_coords" },
//                             [algo_ptr](Neo::ItemLidsProperty const& node_lids_property,
//                                        Neo::PropertyT<Neo::utils::Real3>& node_coords_property) {
//                               algo_ptr();
//                             });
//}

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/