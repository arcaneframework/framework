// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NeoMeshAPITest.h                                (C) 2000-2020             */
/*                                                                           */
/* First tests for mesh class using Neo                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "gtest/gtest.h"
#include "neo/Mesh.h"
#include "neo/Neo.h"

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, MeshApiCreationTest) {
  auto mesh_name = "MeshTest";
  auto mesh = Neo::Mesh{ mesh_name };
  std::cout << "Creating mesh " << mesh.name();
  EXPECT_EQ(mesh_name, mesh.name());
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, AddFamilyTest) {
  auto mesh = Neo::Mesh{ "AddFamilyTestMesh" };
  auto cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
  std::cout << "Create family " << cell_family.name() << " with item kind " << Neo::utils::itemKindName(cell_family.itemKind()) << std::endl;
  EXPECT_EQ(cell_family.name(), "CellFamily");
  EXPECT_EQ(cell_family.itemKind(), Neo::ItemKind::IK_Cell);
  auto node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "NodeFamily");
  std::cout << "Create family " << node_family.name() << " with item kind " << Neo::utils::itemKindName(node_family.itemKind()) << std::endl;
  EXPECT_EQ(node_family.name(), "NodeFamily");
  EXPECT_EQ(node_family.itemKind(), Neo::ItemKind::IK_Node);
  auto dof_family = mesh.addFamily(Neo::ItemKind::IK_Dof, "DoFFamily");
  std::cout << "Create family " << dof_family.name() << " with item kind " << Neo::utils::itemKindName(dof_family.itemKind()) << std::endl;
  EXPECT_EQ(dof_family.name(), "DoFFamily");
  EXPECT_EQ(dof_family.itemKind(), Neo::ItemKind::IK_Dof);
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, AddItemTest) {
  auto mesh = Neo::Mesh{ "AddItemsTestMesh" };
  auto cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
  auto added_cells = Neo::FutureItemRange{};
  auto added_cells2 = Neo::FutureItemRange{};
  auto added_cells3 = Neo::FutureItemRange{};
  {
    // check lifetime
    std::vector<Neo::utils::Int64> cell_uids{ 1, 10, 100 };
    mesh.scheduleAddItems(cell_family, cell_uids, added_cells);
  }
  std::vector<Neo::utils::Int64> cell_uids_ref{ 1, 10, 100 };
  std::vector<Neo::utils::Int64> cell_uids2_ref{ 2, 3, 4 };
  std::vector<Neo::utils::Int64> cell_uids3_ref{ 1, 10, 100 };
  mesh.scheduleAddItems(cell_family, { 2, 3, 4 }, added_cells2); // memory stealing API
  mesh.scheduleAddItems(cell_family, std::move(cell_uids3_ref), added_cells3); // memory stealing API
  cell_uids3_ref = { 1, 10, 100 }; // for test checks (it was emtpy after the move)
  auto end_mesh_update = mesh.applyScheduledOperations();
  auto new_cells = added_cells.get(end_mesh_update);
  for (auto item : new_cells) {
    std::cout << "Added local id " << item << std::endl;
  }
  auto new_cells2 = added_cells2.get(end_mesh_update);
  for (auto item : new_cells2) {
    std::cout << "Added local id " << item << std::endl;
  }
  auto new_cells3 = added_cells3.get(end_mesh_update);
  for (auto item : new_cells3) {
    std::cout << "Added local id " << item << std::endl;
  }
  // API for uids
  // get uid property
  auto const& cell_uid_property = mesh.getItemUidsProperty(cell_family);
  EXPECT_EQ(&cell_uid_property,
            &cell_family.getConcreteProperty<Neo::Mesh::UidPropertyType>(mesh.uniqueIdPropertyName(cell_family.name())));
  // or get directly uids
  auto cell_uids = mesh.uniqueIds(cell_family, new_cells.localIds());
  auto cell_uids2 = mesh.uniqueIds(cell_family, new_cells2.localIds());
  auto new_cells3_local_ids = new_cells3.localIds();
  auto cell_uids3 = mesh.uniqueIds(cell_family, Neo::utils::Int32ConstSpan{ new_cells3_local_ids.size(), new_cells3_local_ids.data() }); // to test span API
  auto i = 0;
  for (auto item : new_cells) {
    std::cout << "Added unique id " << cell_uid_property[item] << std::endl;
    EXPECT_EQ(cell_uids[i++], cell_uid_property[item]);
  }
  i = 0;
  for (auto item : new_cells2) {
    std::cout << "Added unique id " << cell_uid_property[item] << std::endl;
    EXPECT_EQ(cell_uids2[i++], cell_uid_property[item]);
  }
  i = 0;
  for (auto item : new_cells3) {
    std::cout << "Added unique id " << cell_uid_property[item] << std::endl;
    EXPECT_EQ(cell_uids3[i++], cell_uid_property[item]);
  }
  // API for lids
  auto cell_lids = mesh.localIds(cell_family, cell_uids);
  auto cell_lids2 = mesh.localIds(cell_family, cell_uids2);
  auto cell_lids3 = mesh.localIds(cell_family, cell_uids3);
  auto cell_lids_ref = new_cells.localIds();
  auto cell_lids_ref2 = new_cells2.localIds();
  auto cell_lids_ref3 = new_cells3.localIds();
  EXPECT_TRUE(std::equal(cell_lids_ref.begin(), cell_lids_ref.end(), cell_lids.begin()));
  EXPECT_TRUE(std::equal(cell_lids_ref2.begin(), cell_lids_ref2.end(), cell_lids2.begin()));
  EXPECT_TRUE(std::equal(cell_lids_ref3.begin(), cell_lids_ref3.end(), cell_lids3.begin()));
  // Get uids view
  auto uid_view = cell_uid_property.constView(new_cells);
  // Print uids
  for (auto i = 0; i < new_cells.size(); ++i) {
    std::cout << "uid view index " << i << " = " << uid_view[i]<< std::endl;
  }
 }

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, MeshApiInfoTest) {
  auto mesh = Neo::Mesh{ "MeshTest" };
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  auto& face_family = mesh.addFamily(Neo::ItemKind::IK_Face, "face_family");
  auto& edge_family = mesh.addFamily(Neo::ItemKind::IK_Edge, "edge_family");
  auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "node_family");
  auto& dof_family = mesh.addFamily(Neo::ItemKind::IK_Dof, "dof_family");
  auto added_cells = Neo::FutureItemRange{};
  mesh.scheduleAddItems(cell_family, { 0, 1 }, added_cells);
  auto added_faces = Neo::FutureItemRange{};
  mesh.scheduleAddItems(face_family, { 0, 1, 2, 3, 4 }, added_faces);
  auto added_edges = Neo::FutureItemRange{};
  mesh.scheduleAddItems(edge_family, { 0, 1, 2, 3, 4 }, added_edges);
  auto added_nodes = Neo::FutureItemRange{};
  mesh.scheduleAddItems(node_family, { 0, 1, 2, 3 }, added_nodes);
  auto added_dofs = Neo::FutureItemRange{};
  mesh.scheduleAddItems(dof_family, { 0, 1 }, added_dofs);
  mesh.applyScheduledOperations();
  EXPECT_EQ(mesh.nbCells(), 2);
  EXPECT_EQ(mesh.nbFaces(), 5);
  EXPECT_EQ(mesh.nbFaces(), 5);
  EXPECT_EQ(mesh.nbNodes(), 4);
  EXPECT_EQ(mesh.nbDoFs(), 2);
  EXPECT_EQ(mesh.dimension(), 3);
  auto& found_cell_family =
  mesh.findFamily(Neo::ItemKind::IK_Cell, "cell_family");
  EXPECT_EQ(&cell_family, &found_cell_family);
  auto& found_face_family =
  mesh.findFamily(Neo::ItemKind::IK_Face, "face_family");
  EXPECT_EQ(&face_family, &found_face_family);
  auto& found_edge_family =
  mesh.findFamily(Neo::ItemKind::IK_Edge, "edge_family");
  EXPECT_EQ(&edge_family, &found_edge_family);
  auto& found_node_family =
  mesh.findFamily(Neo::ItemKind::IK_Node, "node_family");
  EXPECT_EQ(&node_family, &found_node_family);
  auto& found_dof_family =
  mesh.findFamily(Neo::ItemKind::IK_Dof, "dof_family");
  EXPECT_EQ(&dof_family, &found_dof_family);
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, SetNodeCoordsTest) {
  auto mesh = Neo::Mesh{ "SetNodeCoordsTestMesh" };
  auto node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "NodeFamily");
  auto added_nodes = Neo::FutureItemRange{};
  auto added_nodes2 = Neo::FutureItemRange{};
  std::vector<Neo::utils::Int64> node_uids{ 1, 10, 100 };
  mesh.scheduleAddItems(node_family, node_uids, added_nodes);
  mesh.scheduleAddItems(node_family, { 0, 5 }, added_nodes2);
  {
    std::vector<Neo::utils::Real3> node_coords{ { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } };
    mesh.scheduleSetItemCoords(node_family, added_nodes, node_coords);
  } // Check memory
  mesh.scheduleSetItemCoords(node_family, added_nodes2, { { 1, 0, 0 }, { 1, 1, 1 } }); // memory stealing API
  auto item_range_unlocker = mesh.applyScheduledOperations();
  auto added_node_range = added_nodes.get(item_range_unlocker);
  auto added_node_range2 = added_nodes2.get(item_range_unlocker);
  auto& node_coord_property = mesh.getItemCoordProperty(node_family);
  auto const& node_coord_property_const = mesh.getItemCoordProperty(node_family);
  auto i = 0;
  std::vector<Neo::utils::Real3> node_coords{ { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } };
  for (auto item : added_node_range) {
    std::cout << "Node coord for item " << item << " = " << node_coord_property_const[item] << std::endl;
    EXPECT_EQ(node_coord_property_const[item].x, node_coords[i].x);
    EXPECT_EQ(node_coord_property_const[item].y, node_coords[i].y);
    EXPECT_EQ(node_coord_property_const[item].z, node_coords[i++].z);
  }
  i = 0;
  node_coords = { { 1, 0, 0 }, { 1, 1, 1 } };
  for (auto item : added_node_range2) {
    std::cout << "Node coord for item " << item << " = " << node_coord_property_const[item] << std::endl;
    EXPECT_EQ(node_coord_property_const[item].x, node_coords[i].x);
    EXPECT_EQ(node_coord_property_const[item].y, node_coords[i].y);
    EXPECT_EQ(node_coord_property_const[item].z, node_coords[i++].z);
  }
  // Change coords : can also use moveItems API, cf NeoEvolutiveMeshTest.cpp
  node_coords = { { 0, 0, 0 }, { 0, 0, -1 }, { 0, -1, 0 } };
  i = 0;
  for (auto item : added_node_range) {
    node_coord_property[item] = node_coords[i];
    EXPECT_EQ(node_coord_property_const[item].x, node_coords[i].x);
    EXPECT_EQ(node_coord_property_const[item].y, node_coords[i].y);
    EXPECT_EQ(node_coord_property_const[item].z, node_coords[i++].z);
  }
  // Check throw for non-existing coord property
  auto cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
  EXPECT_THROW(mesh.getItemCoordProperty(cell_family), std::invalid_argument);
}

TEST(NeoMeshApiTest,AddItemConnectivity) {
  auto mesh = Neo::Mesh{"AddItemConnectivityTestMesh"};
  auto node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "NodeFamily");
  auto cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
  auto dof_family = mesh.addFamily(Neo::ItemKind::IK_Dof, "DoFFamily");
  std::vector<Neo::utils::Int64> node_uids{ 0, 1, 2, 3, 4, 5 };
  std::vector<Neo::utils::Int64> cell_uids{ 0, 1 };
  std::vector<Neo::utils::Int64> dof_uids{ 0, 1, 2, 3, 4 };
  auto future_nodes = Neo::FutureItemRange{};
  auto future_cells = Neo::FutureItemRange{};
  auto future_dofs = Neo::FutureItemRange{};
  mesh.scheduleAddItems(node_family, node_uids, future_nodes);
  mesh.scheduleAddItems(cell_family, cell_uids, future_cells);
  mesh.scheduleAddItems(dof_family, dof_uids, future_dofs);
  // Create connectivity (fictive mesh) cells with 4 nodes
  std::string cell_to_nodes_connectivity_name{ "cell_to_nodes" };
  std::string cell_to_dofs_connectivity_name{ "cell_to_dofs" };
  std::string node_to_cells_connectivity_name{ "node_to_cell" };
  std::string node_to_dofs_connectivity_name{ "node_to_dofs" };

  // Connectivity cell to nodes
  auto nb_node_per_cell = 4;
  {
    std::vector<Neo::utils::Int64> cell_nodes{ 0, 1, 2, 3, 5, 0, 3, 4 };
    mesh.scheduleAddConnectivity(cell_family, future_cells, node_family,
                                 nb_node_per_cell, cell_nodes,
                                 cell_to_nodes_connectivity_name);
  } // check memory
  // Connectivity cell to dof
  std::vector<int> nb_dof_per_cell{ 3, 2 };
  std::vector<Neo::utils::Int64> cell_dofs{ 0, 3, 4, 2, 1 };
  std::vector<Neo::utils::Int64> cell_dofs_ref{ cell_dofs };
  mesh.scheduleAddConnectivity(cell_family, future_cells, dof_family,
                               std::move(nb_dof_per_cell), std::move(cell_dofs),
                               cell_to_dofs_connectivity_name);
  // apply
  auto end_mesh_update = mesh.applyScheduledOperations();
  // Add further connectivity
  auto added_nodes = future_nodes.get(end_mesh_update);
  mesh.scheduleAddConnectivity(node_family, added_nodes, cell_family,
                               { 2, 1, 1, 2, 1, 1 }, { 0, 1, 0, 0, 0, 1, 1, 1 },
                               node_to_cells_connectivity_name);
  auto nb_dof_per_node = 1;
  mesh.scheduleAddConnectivity(node_family, added_nodes, dof_family,
                               nb_dof_per_node, { 0, 1, 2, 3, 4, 0 },
                               node_to_dofs_connectivity_name);
  end_mesh_update = mesh.applyScheduledOperations();
  auto added_cells = future_cells.get(end_mesh_update);
  // check connectivities
  // check cell_to_nodes
  auto cell_to_nodes = mesh.getConnectivity(cell_family, node_family, cell_to_nodes_connectivity_name);
  EXPECT_EQ(cell_to_nodes_connectivity_name, cell_to_nodes.name);
  EXPECT_EQ(&cell_family, &cell_to_nodes.source_family);
  EXPECT_EQ(&node_family, &cell_to_nodes.target_family);
  for (auto const cell : added_cells) {
    std::cout << "cell lid " << cell << " nodes lids " << cell_to_nodes[cell]
              << std::endl;
  }
  auto const cell_to_dofs = mesh.getConnectivity(
      cell_family, dof_family, cell_to_dofs_connectivity_name);
  EXPECT_EQ(cell_to_dofs_connectivity_name, cell_to_dofs.name);
  EXPECT_EQ(&cell_family, &cell_to_dofs.source_family);
  EXPECT_EQ(&dof_family, &cell_to_dofs.target_family);
  std::vector<Neo::utils::Int32> cell_dofs_lids =
      dof_family.itemUniqueIdsToLocalids(cell_dofs_ref);
  for (auto const cell : added_cells) {
    auto current_cell_dofs = cell_to_dofs[cell];
    std::cout << "cell lid " << cell << " dofs lids " << current_cell_dofs
              << std::endl;
    auto i = 0;
    for (auto dof_lid : current_cell_dofs) {
      EXPECT_EQ(dof_lid, current_cell_dofs[i++]);
    }
  }
  // Check another connectivity getter
  auto cell_to_nodes_connectivities = mesh.nodes(
      cell_family); // returns all IK_Node families connected wih cell_family
  for (auto connectivity : cell_to_nodes_connectivities) {
    std::cout << "Connectivity name " << connectivity.name;
  }
  auto cell_to_dofs_connectivities = mesh.dofs(
      cell_family); // returns all IK_Node families connected wih cell_family
  for (auto connectivity : cell_to_dofs_connectivities) {
    std::cout << "Connectivity name " << connectivity.name;
  }
  // Add new connectivities
  std::vector<Neo::utils::Int64> cell_nodes2{
      0, 3, 4, 5, 0, 1, 2, 3,
      0, 1, 2, 3, 0, 3, 4, 5}; // node + neighbour cell nodes
  std::vector<Neo::utils::Int64> face_uids{0, 1, 2, 3, 4, 5, 6};
  std::vector<Neo::utils::Int64> cell_faces{0, 1, 2, 3, 4, 5, 6, 1};
  std::vector<Neo::utils::Int64> edge_uids{0, 1, 2, 3, 4, 5, 6};
  std::vector<Neo::utils::Int64> cell_edges{0, 1, 2, 3, 4, 5, 6, 1};
  std::vector<Neo::utils::Int64> cell_dofs2{0, 1, 2, 3, 0, 4};
  auto &face_family = mesh.addFamily(Neo::ItemKind::IK_Face, "face_family");
  auto &edge_family = mesh.addFamily(Neo::ItemKind::IK_Edge, "edge_family");
  auto promised_faces = Neo::FutureItemRange{};
  auto promised_edges = Neo::FutureItemRange{};
  mesh.scheduleAddItems(face_family, face_uids, promised_faces);
  mesh.scheduleAddItems(edge_family, edge_uids, promised_edges);
  mesh.scheduleAddConnectivity(cell_family, added_cells, node_family, 8,
                               cell_nodes2, "cell_to_nodes_2");
  mesh.scheduleAddConnectivity(cell_family, added_cells, face_family, 4,
                               cell_faces, "cell_to_faces");
  mesh.scheduleAddConnectivity(cell_family, added_cells, edge_family, 4,
                               cell_edges, "cell_to_edges");
  mesh.scheduleAddConnectivity(cell_family, added_cells, dof_family, 3,
                               cell_dofs2, "cell_to_dofs_2");
  mesh.applyScheduledOperations();
  // connectivity access
  cell_to_nodes_connectivities = mesh.nodes(
      cell_family); // returns all IK_Node families connected wih cell_family
  for (auto connectivity : cell_to_nodes_connectivities) {
    std::cout << "Connectivity name " << connectivity.name;
  }
  cell_to_dofs_connectivities = mesh.dofs(
      cell_family); // returns all IK_Node families connected wih cell_family
  for (auto connectivity : cell_to_dofs_connectivities) {
    std::cout << "Connectivity name " << connectivity.name;
  }
  // check asking non existing connectivity
  EXPECT_THROW(
      mesh.getConnectivity(cell_family, node_family, "unexisting_connectivity"),
      std::invalid_argument);
  // Change an existing connectivity : cell 0 now points to dofs uids {3,4}
  auto cell_lids = cell_family.itemUniqueIdsToLocalids({0});
  Neo::ItemRange cell_range{cell_lids};
  auto connected_dofs = std::vector<Neo::utils::Int64>{3, 4};
  mesh.scheduleAddConnectivity(cell_family, cell_range, dof_family, 2,
                               connected_dofs, cell_to_dofs_connectivity_name,
                               Neo::Mesh::ConnectivityOperation::Modify);
  mesh.applyScheduledOperations();
  // Try to connect a subpart of added items by index
  {
    cell_uids = {2, 3, 4};
    dof_uids = {5, 6};
    auto added_cells_future_new = Neo::FutureItemRange{};
    auto added_dofs_future_new = Neo::FutureItemRange{};
    mesh.scheduleAddItems(cell_family, cell_uids, added_cells_future_new);
    mesh.scheduleAddItems(dof_family, dof_uids, added_dofs_future_new);
    // Create a filtered ItemRange containing elements with indexes 0 & 1
    auto filtered_future_cell_range =
        Neo::make_future_range(added_cells_future, {0, 1});
    mesh.scheduleAddConnectivity(cell_family, filtered_future_cell_range,
                                 dof_family, {1, 2}, {5, 5, 6},
                                 "cell_to_dofs_new");
    auto end_update = mesh.applyScheduledOperations();
    auto added_cells_filtered = filtered_future_cell_range.get(end_update);
    auto added_cells2 = added_cells_future_new.get(end_update);
  }
  // Try to connect a subpart of added items by an uids vector
  {
    cell_uids = {5, 6, 7};
    dof_uids = {7, 8};
    auto added_cells_future_new = Neo::FutureItemRange{};
    auto added_dofs_future_new = Neo::FutureItemRange{};
    mesh.scheduleAddItems(cell_family, cell_uids, added_cells_future_new);
    mesh.scheduleAddItems(dof_family, dof_uids, added_dofs_future_new);
    // Create a filtered ItemRange containing cells with uids 5 & 7
    auto filtered_future_cell_range =
    Neo::make_future_range(added_cells_future_new, cell_uids, { 5, 7 });
    mesh.scheduleAddConnectivity(cell_family, filtered_future_cell_range,
                                 dof_family, { 2, 1 }, { 7, 8, 7 },
                                 "cell_to_dofs_new2");
    mesh.scheduleAddConnectivity(cell_family, added_cells_future_new,
                                 dof_family, { 2, 1, 2 }, { 7, 8, 7, 8, 7 },
                                 "cell_to_dofs_new3");
    auto end_update = mesh.applyScheduledOperations();
    auto cell_to_dofs_new = mesh.dofs(cell_family).back(); // get the last cell to dof connectivity
    auto cell_dofs_cons = mesh.dofs(cell_family);
    auto added_cell_new = added_cells_future_new.get(end_update);
    for (auto cell : added_cell_new) {
      for (auto dof : cell_to_dofs_new[cell]) {
        std::cout << "cell " << cell << " connected with dof " << dof << std::endl;
      }
    }
    //    auto source_cell_lids = mesh.localIds(cell_family,{5,7});
//    std::vector<Neo::utils::Int32> connected_dofs_lids;
//    connected_dofs_lids.reserve(2);
//    for (const auto cell : source_cell_lids) {
//      for (auto dof : cell_to_dofs_new[cell]) {
//        connected_dofs_lids.push_back(dof);
//      }
//    }
//    std::vector<Neo::utils::Int64> connected_dofs_uids = mesh.uniqueIds(dof_family,connected_dofs_lids);
//    std::vector<Neo::utils::Int64> connected_dofs_uids_ref {7,8,7};
//    EXPECT_TRUE(std::equal(connected_dofs_uids_ref.begin(),
//                           connected_dofs_uids_ref.end(),
//                           connected_dofs_uids.begin()));
  }
}