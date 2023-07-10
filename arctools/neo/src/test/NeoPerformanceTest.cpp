// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NeoPerformanceTest.cpp                          (C) 2000-2023             */
/*                                                                           */
/* First performance check on Neo                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include <numeric>

#include "neo/Mesh.h"
#include "neo/Utils.h"

TEST(PerformanceTests, AddItemsByCopy) {
  auto nb_cells = 1e3; // 1e6
  std::vector<Neo::utils::Int64> cell_uids(nb_cells);
  std::iota(cell_uids.begin(), cell_uids.end(), 0);
  Neo::Mesh mesh{ "mesh" };
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  Neo::FutureItemRange cell_added_items;
  Neo::utils::Profiler profiler{ "AddItemsByCopy" };
  profiler.start();
  mesh.scheduleAddItems(cell_family, cell_uids, cell_added_items);
  profiler.stop();
  profiler.print("add mesh algorithms ");
  profiler.start();
  mesh.applyScheduledOperations();
  profiler.stop();
  profiler.print("apply mesh algorithms ");
}

/*-----------------------------------------------------------------------------*/

TEST(PerformanceTests, AddItemsByMove) {
  auto nb_cells = 1e3; // 1e6
  std::vector<Neo::utils::Int64> cell_uids(nb_cells);
  std::iota(cell_uids.begin(), cell_uids.end(), 0);
  Neo::Mesh mesh{ "mesh" };
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  Neo::FutureItemRange cell_added_items;
  Neo::utils::Profiler profiler{ "AddItemsByMove" };
  profiler.start();
  mesh.scheduleAddItems(cell_family, std::move(cell_uids), cell_added_items);
  profiler.stop();
  profiler.print("add mesh algorithms ");
  profiler.start();
  mesh.applyScheduledOperations();
  profiler.stop();
  profiler.print("apply mesh algorithms ");
}

/*-----------------------------------------------------------------------------*/

TEST(PerformanceTests, AppendItems) {
  auto nb_cells = 1e3; //1e6
  std::vector<Neo::utils::Int64> cell_uids(nb_cells);
  std::iota(cell_uids.begin(), cell_uids.end(), 0);
  Neo::Mesh mesh{ "mesh" };
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  // Add a first block of items
  Neo::FutureItemRange cell_added_items;
  mesh.scheduleAddItems(cell_family, std::move(cell_uids), cell_added_items);
  mesh.applyScheduledOperations();
  // Append new items
  cell_uids.resize(nb_cells);
  std::iota(cell_uids.begin(), cell_uids.end(), nb_cells);
  Neo::utils::Profiler profiler{ "AppendItems" };
  profiler.start();
  profiler.stop();
  profiler.print("add mesh algorithms ");
  mesh.scheduleAddItems(cell_family, std::move(cell_uids), cell_added_items); // should be useless to move...to check
  profiler.start();
  mesh.applyScheduledOperations();
  profiler.stop();
  profiler.print("apply mesh algorithms ");
}

/*-----------------------------------------------------------------------------*/

TEST(PerformanceTests, AddConnectivity) {
  int nb_cells = 1e3; // 1e5
  auto nb_node_per_cells = 4;
  std::vector<Neo::utils::Int64> cell_uids(nb_cells);
  std::iota(cell_uids.begin(), cell_uids.end(), 0);
  std::vector<Neo::utils::Int64> node_uids(nb_cells * nb_node_per_cells);
  std::iota(node_uids.begin(), node_uids.end(), 0);
  Neo::Mesh mesh{ "mesh" };
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "node_family");
  Neo::FutureItemRange cell_added_items;
  Neo::FutureItemRange node_added_items;
  Neo::utils::Profiler profiler{ "AddCell&NodesByCopy" };
  profiler.start();
  mesh.scheduleAddItems(cell_family, cell_uids, cell_added_items);
  mesh.scheduleAddItems(node_family, node_uids, node_added_items);
  profiler.stop();
  profiler.print("add mesh algorithms - AddItems");
  profiler.start();
  mesh.applyScheduledOperations();
  profiler.stop();
  profiler.print("apply mesh algorithms - AddItems");
  profiler.start("add mesh algorithms - AddConnectivities");
  //  mesh.scheduleAddConnectivity(cell_family, cell_family.all(), node_family, nb_node_per_cells, node_uids, "cell_to_nodes");
  mesh.scheduleAddConnectivity(cell_family, cell_added_items, node_family, nb_node_per_cells, node_uids, "cell_to_nodes");
  profiler.stop_and_print();
  profiler.start("apply mesh algorithms - AddConnectivities");
  mesh.applyScheduledOperations();
  profiler.stop_and_print();
  // we reset the same connectivity to check time difference between creation and modification
  profiler.start("add mesh algorithms - AppendConnectivities");
  mesh.scheduleAddConnectivity(cell_family, cell_family.all(), node_family, nb_node_per_cells, node_uids, "cell_to_nodes",
                               Neo::Mesh::ConnectivityOperation::Modify);
  profiler.stop_and_print();
  profiler.start("apply mesh algorithms - AppendConnectivities");
  mesh.applyScheduledOperations();
  profiler.stop_and_print();
}

/*-----------------------------------------------------------------------------*/

// todo :  set coordinates, methode items(), localids
// + voir le scaling en fonction du nombre d'items