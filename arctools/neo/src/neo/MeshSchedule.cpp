// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshSchedule.cpp                                            (C) 2000-2025 */
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

void Neo::Mesh::_filterNullItems(std::vector<Neo::utils::Int32>& connected_item_lids, std::vector<int>& nb_connected_item_per_item) {
  auto index = 0;
  bool has_null_item = false;
  for (auto& nb_item : nb_connected_item_per_item) {
    auto nb_item_copy = nb_item;
    for (auto i = 0 ; i < nb_item_copy ; ++i){
      if (connected_item_lids[index] == Neo::utils::NULL_ITEM_LID) {
        --nb_item;
        has_null_item = true;
      }
      ++index;
    }
  }
  if (!has_null_item) return;
  std::vector<Neo::utils::Int32> non_null_connected_items;
  non_null_connected_items.reserve(connected_item_lids.size());
  std::copy_if(connected_item_lids.begin(),connected_item_lids.end(),
      std::back_inserter(non_null_connected_items),
      [](int value){return value != Neo::utils::NULL_ITEM_LID;});

  connected_item_lids = std::move(non_null_connected_items);
  connected_item_lids.shrink_to_fit();
}

/*-----------------------------------------------------------------------------*/


template <typename ItemRangeT>
void Neo::Mesh::_scheduleAddConnectivity(Neo::Family& source_family, Neo::ItemRangeWrapper<ItemRangeT> source_items_wrapper,
                                         Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                         std::vector<Neo::utils::Int64> connected_item_uids,
                                         std::string const& connectivity_unique_name,
                                         ConnectivityOperation add_or_modify) {
  // add connectivity property if doesn't exist
  source_family.addMeshArrayProperty<Neo::utils::Int32>(connectivity_unique_name);
  // add orientation property (only used for oriented connectivity)
  std::string orientation_name = _connectivityOrientationPropertyName(source_family.name(), target_family.name());
  source_family.addMeshArrayProperty<Neo::utils::Int32>(orientation_name);
  // Create connectivity wrapper and add it to mesh
  auto& connectivity_property = source_family.getConcreteProperty<Mesh::ConnectivityPropertyType>(connectivity_unique_name);
  auto& connectivity_orientation = source_family.getConcreteProperty<Mesh::ConnectivityPropertyType>(orientation_name);
  Connectivity current_connectivity = Connectivity{ source_family, target_family,
                                                    connectivity_unique_name, connectivity_property,
                                                    connectivity_orientation };
  auto [inserted_connectivity_iterator, is_inserted] = m_connectivities.insert(std::make_pair(connectivity_unique_name, current_connectivity));
  auto& source_family_con = m_connectivities_per_family[{ source_family.itemKind(), source_family.name() }];
  if (std::find(source_family_con.begin(),
                source_family_con.end(),
                current_connectivity) == source_family_con.end()) {
    source_family_con.push_back(current_connectivity);
  }
  if (!is_inserted && add_or_modify == ConnectivityOperation::Add && !inserted_connectivity_iterator->second.isEmpty()) {
    throw std::invalid_argument("Cannot include already inserted connectivity " + connectivity_unique_name + ". Choose ConnectivityOperation::Modify");
  }
  auto nb_connected_item_sum = std::accumulate(nb_connected_item_per_item.begin(),nb_connected_item_per_item.end(),0);
  if (connected_item_uids.empty() || nb_connected_item_per_item.empty() || nb_connected_item_sum == 0) {
    Neo::print(m_rank) << "== Algorithm: register empty connectivity between " << source_family.m_name << "  and  " << target_family.m_name << std::endl;
    return;
  }
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ source_family, source_family.lidPropName(), PropertyStatus::ExistingProperty },
  Neo::MeshKernel::InProperty{ target_family, target_family.lidPropName(), PropertyStatus::ExistingProperty },
  Neo::MeshKernel::OutProperty{ source_family, connectivity_unique_name },
  [connected_item_uids{ std::move(connected_item_uids) },
   nb_connected_item_per_item{ std::move(nb_connected_item_per_item) },
   source_items_wrapper, &source_family, &target_family, rank(m_rank),this](Neo::ItemLidsProperty const& source_family_lids_property,
                                                         Neo::ItemLidsProperty const& target_family_lids_property,
                                                         Neo::MeshArrayPropertyT<Neo::utils::Int32>& source2target) mutable {
    Neo::print(rank) << "== Algorithm: register connectivity between " << source_family.m_name << "  and  " << target_family.m_name << std::endl;
    ItemRange const& source_items = source_items_wrapper.get();
    auto connected_item_lids = target_family_lids_property[connected_item_uids];
    _filterNullItems(connected_item_lids, nb_connected_item_per_item);
    auto nb_connected_item_sum = std::accumulate(nb_connected_item_per_item.begin(),nb_connected_item_per_item.end(),0);
    if (nb_connected_item_sum == 0) {
      Neo::print(rank) << "== Algorithm: register empty connectivity between " << source_family.m_name << "  and  " << target_family.m_name << std::endl;
      return;
    }
    if (connected_item_lids.empty() || nb_connected_item_per_item.empty()) return;
    if (source2target.isInitializableFrom(source_items)) {
      source2target.resize(std::move(nb_connected_item_per_item));
      source2target.init(source_items, std::move(connected_item_lids));
    }
    else {
      source2target.append(source_items, connected_item_lids,
                           nb_connected_item_per_item);
    }
    source2target.debugPrint(rank);
  });

  // update connectivity of target family removed items: this algorithm must be persistant (not removed by a call to applyScheduledOperations)
  auto removed_item_property_name = _removeItemPropertyName(target_family);
  source_family.addMeshScalarProperty<Neo::utils::Int32>(removed_item_property_name);
  auto isolated_items_property_name = _isolatedItemLidsPropertyName(source_family, target_family);
  source_family.addMeshScalarProperty<Neo::utils::Int32>(isolated_items_property_name);
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ target_family, removed_item_property_name },
  Neo::MeshKernel::OutProperty{ source_family, connectivity_unique_name },
  Neo::MeshKernel::OutProperty{ source_family, isolated_items_property_name },
  [&source_family, &target_family,rank(m_rank)](
  Neo::MeshScalarPropertyT<Neo::utils::Int32> const& target_family_removed_items,
  Neo::MeshArrayPropertyT<Neo::utils::Int32>& connectivity,
  Neo::MeshScalarPropertyT<Neo::utils::Int32>& isolated_items) {
    Neo::print(rank) << "== Algorithm: update connectivity after target family remove items " << connectivity.m_name << std::endl;
    isolated_items.init(source_family.all(), 0);
    for (auto item : source_family.all()) {
      auto connected_items = connectivity[item];
      auto nb_disconnected_items = 0;
      for (auto& connected_item : connected_items) {
        if (connected_item == Neo::utils::NULL_ITEM_LID) {
          ++nb_disconnected_items;
        }
        else if (target_family_removed_items[connected_item] == 1) {
          connected_item = Neo::utils::NULL_ITEM_LID;
          ++nb_disconnected_items;
        }
      }
      if (nb_disconnected_items == connected_items.size()) {// all connected items are removed ; item is isolated for this connectivity
        // add to isolated items removal property
        isolated_items[item] = 1;
      }
    }
  },
  Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmPersistence::KeepAfterExecution);

  // update connectivity of source family removed items: this algorithm must be persistant
  removed_item_property_name = _removeItemPropertyName(source_family);
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ source_family, removed_item_property_name },
  Neo::MeshKernel::OutProperty{ source_family, connectivity_unique_name },
  [&source_family,rank(m_rank)](
  Neo::MeshScalarPropertyT<Neo::utils::Int32> const& source_family_removed_items,
  Neo::MeshArrayPropertyT<Neo::utils::Int32>& connectivity) {
    Neo::print(rank) << "== Algorithm: update connectivity after source family remove items " << connectivity.m_name << std::endl;
    std::vector<Neo::utils::Int32> removed_source_items{};
    removed_source_items.reserve(source_family.nbElements());
    auto index = 0;
    for (auto item_status : source_family_removed_items) {
      if (item_status == 1) {
        removed_source_items.push_back(index);
        }
      ++index;
    }
    ItemRange removed_source_item_range{ ItemLocalIds{ removed_source_items } };
    auto zero_sizes = std::vector<int>(removed_source_items.size(), 0);
    connectivity.append(removed_source_item_range,{},zero_sizes);
    },
    MeshKernel::AlgorithmPropertyGraph::AlgorithmPersistence::KeepAfterExecution
  );

  // Add an algorithm to remove isolated items
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ source_family, isolated_items_property_name },
  Neo::MeshKernel::OutProperty{ source_family, source_family.lidPropName() },
  [&source_family,rank(m_rank)](
  Neo::MeshScalarPropertyT<Neo::utils::Int32> const& isolated_items,
  Neo::ItemLidsProperty& item_lids_property){
    Neo::print(rank) << "== Algorithm: remove isolated items in " << source_family.name() << std::endl;
    std::vector<utils::Int32> removed_item_lids{};
    removed_item_lids.reserve(source_family.nbElements());
    for (auto item : source_family.all()) {
      if (isolated_items[item] == 1) {
        removed_item_lids.push_back(item);
      }
    }
    auto& item_uids = source_family.getConcreteProperty<MeshScalarPropertyT<utils::Int64>>(uniqueIdPropertyName(source_family.name()));
    auto removed_item_uids = item_uids[removed_item_lids];
    auto removed_items = item_lids_property.remove(removed_item_uids);
    item_lids_property.debugPrint(rank);
  }, Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmPersistence::KeepAfterExecution);
}

/*-----------------------------------------------------------------------------*/

template <typename ItemRangeT>
void Neo::Mesh::_scheduleAddConnectivityOrientation(Neo::Family& source_family, Neo::ItemRangeWrapper<ItemRangeT> source_items_wrapper,
                                                    Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                                    std::vector<int> source_item_orientation_in_target_item,
                                                    bool do_check_orientation) {
  // add orientation property if doesn't exist
  std::string orientation_property_name = _connectivityOrientationPropertyName(source_family.name(), target_family.name());
  source_family.addMeshArrayProperty<Neo::utils::Int32>(orientation_property_name);
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ source_family, source_family.lidPropName() },
  Neo::MeshKernel::OutProperty{ source_family, orientation_property_name },
  [source_item_orientation_in_target_item{ std::move(source_item_orientation_in_target_item) },
   nb_connected_item_per_item{ std::move(nb_connected_item_per_item) }, source_items_wrapper,
   &source_family, &target_family,rank(m_rank)](Neo::ItemLidsProperty const& source_family_lids_property,
                                   Neo::MeshArrayPropertyT<int>& item_orientation) {
    Neo::print(rank) << "== Algorithm: add orientation in connectivity between "
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
    item_orientation.debugPrint(rank);
  });
  if (do_check_orientation) {
    _addConnectivityOrientationCheck(source_family, target_family);
  }
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
          source_items.size() * nb_connected_item_per_item == (int)connected_item_uids.size()));
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
                                      target_family, nb_connected_item_per_item,
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
                          nb_connected_item_per_item,
                          std::move(connected_item_uids), connectivity_unique_name, add_or_modify);
  _scheduleAddConnectivityOrientation(source_family, Neo::ItemRangeWrapper<const ItemRange>{ source_items },
                                      target_family, nb_connected_item_per_item,
                                      std::move(source_item_orientation_in_target_item),
                                      do_check_orientation);
}

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
