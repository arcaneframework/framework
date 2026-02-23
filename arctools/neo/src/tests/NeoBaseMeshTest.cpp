// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NeoBaseTest.cpp                                 (C) 2000-2026             */
/*                                                                           */
/* Base tests for Neo kernel                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <vector>
#include <array>
#include <algorithm>

#include "gtest/gtest.h"
#include "neo/Neo.h"
#include "neo/MeshKernel.h"


/*-----------------------------------------------------------------------------*/

namespace
{
void add_properties(Neo::Family& cell_family, Neo::Family& node_family) {

  // Adding node family and properties
  node_family.addMeshScalarProperty<Neo::utils::Real3>(std::string("node_coords"));
  node_family.addMeshScalarProperty<Neo::utils::Int64>("node_uids");
  node_family.addMeshArrayProperty<Neo::utils::Int32>("node2cells");
  node_family.addMeshScalarProperty<Neo::utils::Int32>("internal_end_of_remove_tag"); // not a user-defined property // todo use byte ?

  // Test adds
  EXPECT_NO_THROW(node_family.getProperty("node_uids"));

  // Adding cell family and properties
  cell_family.addMeshScalarProperty<Neo::utils::Int64>("cell_uids");
  cell_family.addMeshArrayProperty<Neo::utils::Int32>("cell2nodes");
}
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestBaseMesh, base_mesh_creation_test) {

  std::cout << "*------------------------------------*" << std::endl;
  std::cout << "* Test framework Neo thoughts " << std::endl;
  std::cout << "*------------------------------------*" << std::endl;

  // creating mesh as a graph of algorithm and properties
  auto mesh = Neo::MeshKernel::AlgorithmPropertyGraph{ "my_neo_mesh" };
  Neo::Family node_family{ Neo::ItemKind::IK_Node, "NodeFamily" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "CellFamily" };

  add_properties(cell_family, node_family);
  // return;

  // given data to create mesh. After mesh creation data is no longer available
  std::vector<Neo::utils::Int64> node_uids{ 0, 1, 2 };
  std::vector<Neo::utils::Real3> node_coords{ { 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
  std::vector<Neo::utils::Int64> cell_uids{ 0, 2, 7, 9 };

  // add algos:
  // create nodes
  auto added_nodes = Neo::ItemRange{};
  mesh.addAlgorithm(
  Neo::MeshKernel::OutProperty{ node_family, node_family.lidPropName() },
  [&node_uids, &added_nodes](Neo::ItemLidsProperty& node_lids_property) {
    std::cout << "Algorithm: create nodes" << std::endl;
    added_nodes = node_lids_property.append(node_uids);
    node_lids_property.debugPrint();
    Neo::printer() << "Inserted item range : " << added_nodes;
  });

  // register node uids
  mesh.addAlgorithm(
  Neo::MeshKernel::InProperty{ node_family, node_family.lidPropName() },
  Neo::MeshKernel::OutProperty{ node_family, "node_uids" },
  [&node_uids, &added_nodes]([[maybe_unused]] Neo::ItemLidsProperty const& node_lids_property,
                             Neo::MeshScalarPropertyT<Neo::utils::Int64>& node_uids_property) {
    std::cout << "Algorithm: register node uids" << std::endl;
    if (node_uids_property.isInitializableFrom(added_nodes))
      node_uids_property.init(added_nodes, std::move(node_uids)); // init can steal the input values
    else
      node_uids_property.append(added_nodes, node_uids);
    node_uids_property.debugPrint();
  }); // need to add a property check for existing uid

  // register node coords
  mesh.addAlgorithm(
  Neo::MeshKernel::InProperty{ node_family, node_family.lidPropName() },
  Neo::MeshKernel::OutProperty{ node_family, "node_coords" },
  [&node_coords, &added_nodes]([[maybe_unused]] Neo::ItemLidsProperty const& node_lids_property,
                               Neo::MeshScalarPropertyT<Neo::utils::Real3>& node_coords_property) {
    std::cout << "Algorithm: register node coords" << std::endl;
    if (node_coords_property.isInitializableFrom(added_nodes))
      node_coords_property.init(added_nodes, std::move(node_coords)); // init can steal the input values
    else
      node_coords_property.append(added_nodes, node_coords);
    node_coords_property.debugPrint();
  });
  //
  // Add cells and connectivity

  // create cells
  auto added_cells = Neo::ItemRange{};
  mesh.addAlgorithm(
  Neo::MeshKernel::OutProperty{ cell_family, cell_family.lidPropName() },
  [&cell_uids, &added_cells](Neo::ItemLidsProperty& cell_lids_property) {
    std::cout << "Algorithm: create cells" << std::endl;
    added_cells = cell_lids_property.append(cell_uids);
    cell_lids_property.debugPrint();
    Neo::printer() << "Inserted item range : " << added_cells;
  });

  // register cell uids
  mesh.addAlgorithm(
  Neo::MeshKernel::InProperty{ cell_family, cell_family.lidPropName() },
  Neo::MeshKernel::OutProperty{ cell_family, "cell_uids" },
  [&cell_uids, &added_cells](
  [[maybe_unused]] Neo::ItemLidsProperty const& cell_lids_property,
  Neo::MeshScalarPropertyT<Neo::utils::Int64>& cell_uids_property) {
    std::cout << "Algorithm: register cell uids" << std::endl;
    if (cell_uids_property.isInitializableFrom(added_cells))
      cell_uids_property.init(added_cells, std::move(cell_uids)); // init can steal the input values
    else
      cell_uids_property.append(added_cells, cell_uids);
    cell_uids_property.debugPrint();
  });

  // register connectivity
  // node to cell
  std::vector<Neo::utils::Int64> connected_cell_uids{ 0, 0, 2, 2, 7, 9 };
  std::vector<int> nb_cell_per_node{ 1, 2, 3 };
  mesh.addAlgorithm(
  Neo::MeshKernel::InProperty{ node_family, node_family.lidPropName() },
  Neo::MeshKernel::InProperty{ cell_family, cell_family.lidPropName() },
  Neo::MeshKernel::OutProperty{ node_family, "node2cells" },
  [&connected_cell_uids, &nb_cell_per_node, &added_nodes]([[maybe_unused]] Neo::ItemLidsProperty const& node_lids_property,
                                                          Neo::ItemLidsProperty const& cell_lids_property,
                                                          Neo::MeshArrayPropertyT<Neo::utils::Int32>& node2cells) {
    std::cout << "Algorithm: register node-cell connectivity" << std::endl;
    auto connected_cell_lids = cell_lids_property[connected_cell_uids];
    if (node2cells.isInitializableFrom(added_nodes)) {
      node2cells.resize(std::move(nb_cell_per_node));
      node2cells.init(added_nodes, std::move(connected_cell_lids));
    }
    else {
      node2cells.append(added_nodes, connected_cell_lids, nb_cell_per_node);
    }
    node2cells.debugPrint();
  });

  // cell to node
  std::vector<Neo::utils::Int64> connected_node_uids{ 0, 1, 2, 1, 2, 0, 2, 1, 0 }; // on ne connecte volontairement pas toutes les mailles pour vérifier initialisation ok sur la famille
  std::vector nb_node_per_cell = { 3, 0, 3, 3 };
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ node_family, node_family.lidPropName() },
                    Neo::MeshKernel::InProperty{ cell_family, cell_family.lidPropName() },
                    Neo::MeshKernel::OutProperty{ cell_family, "cell2nodes" },
                    [&connected_node_uids, &nb_node_per_cell, &added_cells](
                    Neo::ItemLidsProperty const& node_lids_property,
                    [[maybe_unused]] Neo::ItemLidsProperty const& cell_lids_property,
                    Neo::MeshArrayPropertyT<Neo::utils::Int32>& cells2nodes) {
                      std::cout << "Algorithm: register cell-node connectivity" << std::endl;
                      auto connected_node_lids = node_lids_property[connected_node_uids];
                      if (cells2nodes.isInitializableFrom(added_cells)) {
                        cells2nodes.resize(std::move(nb_node_per_cell));
                        cells2nodes.init(added_cells, std::move(connected_node_lids));
                      }
                      else
                        cells2nodes.append(added_cells, connected_node_lids, nb_node_per_cell);
                      cells2nodes.debugPrint();
                    });

  // try to modify an existing property
  // add new cells
  std::vector<Neo::utils::Int64> new_cell_uids{ 10, 11, 12 }; // elles seront toutes rouges
  auto new_cell_added = Neo::ItemRange{};
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ cell_family, cell_family.lidPropName() },
                    [&new_cell_uids, &new_cell_added](Neo::ItemLidsProperty& cell_lids_property) {
                      std::cout << "Algorithm: add new cells" << std::endl;
                      new_cell_added = cell_lids_property.append(new_cell_uids);
                      cell_lids_property.debugPrint();
                      Neo::printer() << "Inserted item range : " << new_cell_added;
                    });

  // register new cell uids
  mesh.addAlgorithm(
  Neo::MeshKernel::InProperty{ cell_family, cell_family.lidPropName() },
  Neo::MeshKernel::OutProperty{ cell_family, "cell_uids" },
  [&new_cell_uids, &new_cell_added](
  [[maybe_unused]] Neo::ItemLidsProperty const& cell_lids_property,
  Neo::MeshScalarPropertyT<Neo::utils::Int64>& cell_uids_property) {
    std::cout << "Algorithm: register new cell uids" << std::endl;
    // must append and not initialize
    if (cell_uids_property.isInitializableFrom(new_cell_added))
      cell_uids_property.init(new_cell_added, std::move(new_cell_uids)); // init can steal the input values
    else
      cell_uids_property.append(new_cell_added, new_cell_uids);
    cell_uids_property.debugPrint();
  });

  // add connectivity to new cells
  std::vector<Neo::utils::Int64> new_cell_connected_node_uids{ 0, 1, 2, 1, 2 }; // on ne connecte volontairement pas toutes les mailles pour vérifier initialisation ok sur la famille
  std::vector<int> nb_node_per_new_cell{ 0, 3, 2 };
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ node_family, node_family.lidPropName() },
                    Neo::MeshKernel::InProperty{ cell_family, cell_family.lidPropName() },
                    Neo::MeshKernel::OutProperty{ cell_family, "cell2nodes" },
                    [&new_cell_connected_node_uids, &nb_node_per_new_cell, &new_cell_added](
                    Neo::ItemLidsProperty const& node_lids_property,
                    [[maybe_unused]] Neo::ItemLidsProperty const& cell_lids_property,
                    Neo::MeshArrayPropertyT<Neo::utils::Int32>& cells2nodes) {
                      std::cout << "Algorithm: register new cell-node connectivity" << std::endl;
                      auto connected_node_lids = node_lids_property[new_cell_connected_node_uids];
                      if (cells2nodes.isInitializableFrom(new_cell_added)) {
                        cells2nodes.resize(std::move(nb_node_per_new_cell));
                        cells2nodes.init(new_cell_added, std::move(connected_node_lids));
                      }
                      else
                        cells2nodes.append(new_cell_added, connected_node_lids, nb_node_per_new_cell);
                      cells2nodes.debugPrint();
                    });

  // remove nodes
  std::vector<Neo::utils::Int64> removed_node_uids{ 1, 2 };
  auto removed_nodes = Neo::ItemRange{};
  mesh.addAlgorithm(
  Neo::MeshKernel::OutProperty{ node_family, node_family.lidPropName() },
  Neo::MeshKernel::OutProperty{ node_family, "internal_end_of_remove_tag" },
  [&removed_node_uids, &removed_nodes, &node_family](
  Neo::ItemLidsProperty& node_lids_property,
  Neo::MeshScalarPropertyT<Neo::utils::Int32>& internal_end_of_remove_tag) {
    // Store removed items in internal_end_of_remove_tag
    internal_end_of_remove_tag.init(node_family.all(), 0);
    for (auto removed_item : removed_nodes) {
      internal_end_of_remove_tag[removed_item] = 1;
    }
    std::cout << "Algorithm: remove nodes" << std::endl;
    removed_nodes = node_lids_property.remove(removed_node_uids);
    node_lids_property.debugPrint();
    Neo::printer() << "removed item range : " << removed_nodes;
  });

  // handle node removal in connectivity with node family = target family
  mesh.addAlgorithm(
  Neo::MeshKernel::InProperty{ node_family, "internal_end_of_remove_tag" },
  Neo::MeshKernel::OutProperty{ cell_family, "cell2nodes" },
  [&cell_family](
  Neo::MeshScalarPropertyT<Neo::utils::Int32> const& internal_end_of_remove_tag,
  Neo::MeshArrayPropertyT<Neo::utils::Int32>& cells2nodes) {
    //                    std::transform()
    //                    Neo::ItemRange node_range {Neo::ItemLocalIds{{},0,node_family.size()}};
    for (auto cell : cell_family.all()) {
      auto connected_nodes = cells2nodes[cell];
      for (auto& connected_node : connected_nodes) {
        if (connected_node != Neo::utils::NULL_ITEM_LID && internal_end_of_remove_tag[connected_node] == 1) {
          std::cout << "modify node : " << connected_node << std::endl;
          connected_node = Neo::utils::NULL_ITEM_LID;
        }
      }
    }
  });

  // launch algos
  mesh.applyAlgorithms();
}


/*-----------------------------------------------------------------------------*/

TEST(NeoTestPartialMeshModification, partial_mesh_modif_test) {

  // WIP: test in construction
  // modify node coords
  // input data
  //std::array<int, 3> node_uids{ 0, 1, 3 };
  // Neo::utils::Real3 r = { 0, 0, 0 };
  //std::array<Neo::utils::Real3, 3> node_coords = { r, r, r }; // don't get why I can't write {{0,0,0},{0,0,0},{0,0,0}}; ...??

  // creating mesh as a graph of Algorithms and Properties
  auto mesh = Neo::MeshKernel::AlgorithmPropertyGraph{ "my_neo_mesh" };
  Neo::Family node_family{ Neo::ItemKind::IK_Node, "NodeFamily" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "CellFamily" };

  add_properties(cell_family, node_family);

  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ node_family, node_family.lidPropName() },
                    Neo::MeshKernel::OutProperty{ node_family, "node_coords" },
                    //[&node_coords, &node_uids]( // todo
                    [](
                    [[maybe_unused]] Neo::ItemLidsProperty const& node_lids_property,
                    [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Real3>& node_coords_property) {
                      std::cout << "Algorithm: register node coords" << std::endl;
                      //auto& lids = node_lids_property[node_uids];//todo
                      //node_coords_property.appendAt(lids, node_coords);// steal node_coords memory//todo
                    });

  mesh.applyAlgorithms();
}

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
