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

#include "Neo/Mesh.h"

TEST(PerformanceTests,MeshReadingPerformanceTest){
  auto nb_cells = 100;
  std::vector<Neo::utils::Int64> cell_uids(nb_cells);
  std::iota(cell_uids.begin(), cell_uids.end(), 0);
  auto nb_node_per_cell = 4;
  auto nb_nodes = nb_node_per_cell * nb_cells;
  Neo::Mesh mesh{"mesh"};
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "node_family");
  Neo::FutureItemRange cell_added_items;
  mesh.scheduleAddItems(cell_family, cell_uids, cell_added_items);
  mesh.applyScheduledOperations();
}