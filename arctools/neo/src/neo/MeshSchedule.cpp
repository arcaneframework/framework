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
  Connectivity current_connectivity = Connectivity {source_family,target_family,
                                                    connectivity_unique_name,connectivity_property,
                                                    connectivity_orientation};
  auto [iterator, is_inserted] = m_connectivities.insert(std::make_pair(connectivity_unique_name,current_connectivity));
  m_connectivities_per_family[{source_family.itemKind(),source_family.name()}].push_back(current_connectivity);
  if (!is_inserted && add_or_modify == ConnectivityOperation::Add) {
    throw std::invalid_argument("Cannot include already inserted connectivity " + connectivity_unique_name + ". Choose ConnectivityOperation::Modify");
  }
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ source_family, source_family.lidPropName(), PropertyStatus::ExistingProperty },
  Neo::MeshKernel::InProperty{ target_family, target_family.lidPropName(), PropertyStatus::ExistingProperty },
  Neo::MeshKernel::OutProperty{ source_family, connectivity_unique_name },
  [connected_item_uids{ std::move(connected_item_uids) },
   nb_connected_item_per_item{ std::move(nb_connected_item_per_item) },
   source_items_wrapper, &source_family, &target_family](Neo::ItemLidsProperty const& source_family_lids_property,
                                                         Neo::ItemLidsProperty const& target_family_lids_property,
                                                         Neo::MeshArrayPropertyT<Neo::utils::Int32>& source2target) {
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
  source_family.addMeshScalarProperty<Neo::utils::Int32>(removed_item_property_name);
  m_mesh_graph->addAlgorithm(
  Neo::MeshKernel::InProperty{ target_family, removed_item_property_name },
  Neo::MeshKernel::OutProperty{ source_family, connectivity_unique_name },
  [&source_family, &target_family](
  Neo::MeshScalarPropertyT<Neo::utils::Int32> const& target_family_removed_items,
  Neo::MeshArrayPropertyT<Neo::utils::Int32>& connectivity) {
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
  Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmPersistence::KeepAfterExecution);
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
   &source_family, &target_family](Neo::ItemLidsProperty const& source_family_lids_property,
                                   Neo::MeshArrayPropertyT<int>& item_orientation) {
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
