// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NeoMeshAPITest.h                                (C) 2000-2025             */
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
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
  std::cout << "Create family " << cell_family.name() << " with item kind " << Neo::utils::itemKindName(cell_family.itemKind()) << std::endl;
  EXPECT_EQ(cell_family.name(), "CellFamily");
  EXPECT_EQ(cell_family.itemKind(), Neo::ItemKind::IK_Cell);
  auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "NodeFamily");
  std::cout << "Create family " << node_family.name() << " with item kind " << Neo::utils::itemKindName(node_family.itemKind()) << std::endl;
  EXPECT_EQ(node_family.name(), "NodeFamily");
  EXPECT_EQ(node_family.itemKind(), Neo::ItemKind::IK_Node);
  auto& dof_family = mesh.addFamily(Neo::ItemKind::IK_Dof, "DoFFamily");
  std::cout << "Create family " << dof_family.name() << " with item kind " << Neo::utils::itemKindName(dof_family.itemKind()) << std::endl;
  EXPECT_EQ(dof_family.name(), "DoFFamily");
  EXPECT_EQ(dof_family.itemKind(), Neo::ItemKind::IK_Dof);
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, AddItemTest) {
  auto mesh = Neo::Mesh{ "AddItemsTestMesh" };
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
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
  // Check range are equals for added cells 1 and 3 : uids are the same
  EXPECT_TRUE(std::equal(new_cells.begin(), new_cells.end(), new_cells3.begin()));
  // API for uids
  // get uid property
  auto const& cell_uid_property = mesh.getItemUidsProperty(cell_family);
  EXPECT_EQ(&cell_uid_property,
            &cell_family.getConcreteProperty<Neo::Mesh::UniqueIdPropertyType>(mesh.uniqueIdPropertyName(cell_family.name())));
  // or get directly uids
  auto cell_uids = mesh.uniqueIds(cell_family, new_cells.localIds());
  auto cell_uids2 = mesh.uniqueIds(cell_family, new_cells2.localIds());
  auto new_cells3_local_ids = new_cells3.localIds();
  auto cell_uids3 = mesh.uniqueIds(cell_family, Neo::utils::Int32ConstSpan{ new_cells3_local_ids.data(), new_cells3_local_ids.size()  }); // to test span API
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
  EXPECT_TRUE(std::equal(cell_uids.begin(), cell_uids.end(), cell_uids_ref.begin()));
  EXPECT_TRUE(std::equal(cell_uids2.begin(), cell_uids2.end(), cell_uids2_ref.begin()));
  EXPECT_TRUE(std::equal(cell_uids3.begin(), cell_uids3.end(), cell_uids3_ref.begin()));

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
  // Check uids1
  auto uid1_view = cell_uid_property.constView(new_cells);
  for (auto i = 0; i < new_cells.size(); ++i) {
    std::cout << "uid1 view index " << i << " = " << uid1_view[i] << std::endl;
  }
  EXPECT_TRUE(std::equal(cell_uids_ref.begin(), cell_uids_ref.end(), uid1_view.begin()));
  // Check uids2
  auto uid2_view = cell_uid_property.constView(new_cells2);
  for (auto i = 0; i < new_cells.size(); ++i) {
    std::cout << "uid2 view index " << i << " = " << uid2_view[i] << std::endl;
  }
  EXPECT_TRUE(std::equal(cell_uids2_ref.begin(), cell_uids2_ref.end(), uid2_view.begin()));
  // Check uids3
  auto uid3_view = cell_uid_property.constView(new_cells3);
  for (auto i = 0; i < new_cells.size(); ++i) {
    std::cout << "uid3 view index " << i << " = " << uid3_view[i] << std::endl;
  }
  EXPECT_TRUE(std::equal(cell_uids3_ref.begin(), cell_uids3_ref.end(), uid3_view.begin()));
  // Check uids complete
  auto uid_complete_view = cell_uid_property.constView();
  for (auto i = 0; i < new_cells.size(); ++i) {
    std::cout << "uid_complete view index " << i << " = " << uid_complete_view[i] << std::endl;
  }
  std::vector<Neo::utils::Int64> cell_uids_complete_ref = { 1, 10, 100, 2, 3, 4 };
  EXPECT_TRUE(std::equal(cell_uids_complete_ref.begin(), cell_uids_complete_ref.end(), uid_complete_view.begin()));
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
  auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "NodeFamily");
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
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
  EXPECT_THROW(mesh.getItemCoordProperty(cell_family), std::invalid_argument);
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest,EmptyMeshConnectivity) {
  // Check empty connectivity
  Neo::Family empty_family{Neo::ItemKind::IK_None, "EmptyFamily"};
  Neo::Mesh::ConnectivityPropertyType empty_property{};
  Neo::Mesh::Connectivity empty_connectivity{empty_family,empty_family,"empty_connectivity",
    empty_property,empty_property};
  EXPECT_TRUE(empty_connectivity.isEmpty());
}

/*---------------------------------------------------------------------------*/

bool areEqual(Neo::Mesh::Connectivity const con1, Neo::Mesh::Connectivity const con2) {
  bool are_equal = con1.name == con2.name;
  are_equal &= &con1.source_family == &con2.source_family;
  are_equal &= &con1.target_family == &con2.target_family;
  are_equal &= &con1.connectivity_value == &con2.connectivity_value;
  are_equal &= &con1.connectivity_orientation == &con2.connectivity_orientation;
  return are_equal;
}

TEST(NeoMeshApiTest, AddItemConnectivity) {
  auto mesh = Neo::Mesh{ "AddItemConnectivityTestMesh" };
  auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "NodeFamily");
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
  auto& dof_family = mesh.addFamily(Neo::ItemKind::IK_Dof, "DoFFamily");
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

  auto empty_connectivities = mesh.getConnectivities(cell_family);
  EXPECT_EQ(empty_connectivities.size(), 0);

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
                               nb_dof_per_cell, std::move(cell_dofs),
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
  EXPECT_FALSE(cell_to_nodes.isEmpty());
  // check operator ==
  auto cell_to_nodes_copy{cell_to_nodes};
  EXPECT_EQ(cell_to_nodes,cell_to_nodes_copy);
  EXPECT_EQ(cell_to_nodes_connectivity_name, cell_to_nodes.name);
  EXPECT_EQ(&cell_family, &cell_to_nodes.source_family);
  EXPECT_EQ(&node_family, &cell_to_nodes.target_family);
  EXPECT_EQ(cell_to_nodes.maxNbConnectedItems(),nb_node_per_cell);
  auto connected_nodes = mesh.uniqueIds(node_family, cell_to_nodes.connectivity_value.constView());
  std::vector<Neo::utils::Int64> cell_nodes_ref{ 0, 1, 2, 3, 5, 0, 3, 4 };
  EXPECT_TRUE(std::equal(connected_nodes.begin(), connected_nodes.end(), cell_nodes_ref.begin()));
  std::vector<Neo::utils::Int32> cell_nodes_lids_ref = node_family.itemUniqueIdsToLocalids(cell_nodes_ref);
  auto i = 0;
  for (auto const cell : added_cells) {
    auto current_cell_nodes = cell_to_nodes[cell];
    std::cout << "cell lid " << cell << " connected nodes lids " << current_cell_nodes
              << std::endl;
    for (auto const& node_lid : current_cell_nodes) {
      EXPECT_EQ(node_lid, cell_nodes_lids_ref[i++]);
    }
  }
  // check cell_to_dofs
  auto cell_to_dofs = mesh.getConnectivity(
  cell_family, dof_family, cell_to_dofs_connectivity_name);
  EXPECT_EQ(cell_to_dofs_connectivity_name, cell_to_dofs.name);
  EXPECT_EQ(&cell_family, &cell_to_dofs.source_family);
  EXPECT_EQ(&dof_family, &cell_to_dofs.target_family);
  EXPECT_EQ(cell_to_dofs.maxNbConnectedItems(), *std::max_element(nb_dof_per_cell.begin(), nb_dof_per_cell.end()));
  auto connected_dofs = cell_to_dofs.connectivity_value.constView();
  std::vector<Neo::utils::Int32> cell_dofs_lids_ref = dof_family.itemUniqueIdsToLocalids(cell_dofs_ref);
  EXPECT_TRUE(std::equal(connected_dofs.begin(), connected_dofs.end(), cell_dofs_lids_ref.begin()));
  i = 0;
  for (auto const cell : added_cells) {
    auto current_cell_dofs = cell_to_dofs[cell];
    std::cout << "cell lid " << cell << " connected dofs lids " << current_cell_dofs
              << std::endl;
    for (auto const& dof_lid : current_cell_dofs) {
      EXPECT_EQ(dof_lid, cell_dofs_lids_ref[i++]);
    }
  }
  // check node_to_cells
  auto node_to_cells = mesh.getConnectivity(node_family, cell_family, node_to_cells_connectivity_name);
  EXPECT_EQ(node_to_cells_connectivity_name, node_to_cells.name);
  EXPECT_EQ(&node_family, &node_to_cells.source_family);
  EXPECT_EQ(&cell_family, &node_to_cells.target_family);
  EXPECT_EQ(node_to_cells.maxNbConnectedItems(), 2);
  auto connected_cells = mesh.uniqueIds(cell_family, node_to_cells.connectivity_value.constView());
  std::vector<Neo::utils::Int64> node_cells_ref{ 0, 1, 0, 0, 0, 1, 1, 1 };
  std::vector<Neo::utils::Int32> node_cells_lids_ref = node_family.itemUniqueIdsToLocalids(node_cells_ref);
  EXPECT_TRUE(std::equal(connected_cells.begin(), connected_cells.end(), node_cells_lids_ref.begin()));
  i = 0;
  for (auto const node : added_nodes) {
    auto current_node_cells = node_to_cells[node];
    std::cout << "node lid " << node << " connected cell lids " << current_node_cells
              << std::endl;
    for (auto const& cell_lid : current_node_cells) {
      EXPECT_EQ(cell_lid, node_cells_lids_ref[i++]);
    }
  }
  // check node_to_dofs
  auto node_to_dofs = mesh.getConnectivity(node_family, dof_family, node_to_dofs_connectivity_name);
  EXPECT_EQ(node_to_dofs_connectivity_name, node_to_dofs.name);
  EXPECT_EQ(&node_family, &node_to_dofs.source_family);
  EXPECT_EQ(&dof_family, &node_to_dofs.target_family);
  EXPECT_EQ(node_to_dofs.maxNbConnectedItems(),nb_dof_per_node);
  connected_dofs = node_to_dofs.connectivity_value.constView();
  std::vector<Neo::utils::Int64> node_dofs_ref{ 0, 1, 2, 3, 4, 0 };
  std::vector<Neo::utils::Int32> node_dofs_lids_ref = dof_family.itemUniqueIdsToLocalids(node_dofs_ref);
  EXPECT_TRUE(std::equal(connected_dofs.begin(), connected_dofs.end(), node_dofs_lids_ref.begin()));
  i = 0;
  for (auto const node : added_nodes) {
    auto current_node_dofs = node_to_dofs[node];
    std::cout << "node lid " << node << " connected dof lids " << current_node_dofs
              << std::endl;
    for (auto const& dof_lid : current_node_dofs) {
      EXPECT_EQ(dof_lid, node_dofs_lids_ref[i++]);
    }
  }
  // Add a second connectivity cell -> dofs
  auto& dof_family2 = mesh.addFamily(Neo::ItemKind::IK_Dof, "dof_family2");
  mesh.scheduleAddConnectivity(cell_family, added_cells, dof_family2, { 2, 2 }, { 10, 11, 10, 11 }, "cell_to_dofs2");
  mesh.applyScheduledOperations();
  auto cell_to_dofs2 = mesh.getConnectivity(cell_family, dof_family2, "cell_to_dofs2");
  // Check another connectivity getter
  // cell to nodes
  auto cell_to_nodes_connectivities = mesh.nodes(
  cell_family); // returns all IK_Node families connected wih cell_family
  for (auto connectivity : cell_to_nodes_connectivities) {
    std::cout << "Connectivity name " << connectivity.name << std::endl;
  }
  EXPECT_EQ(cell_to_nodes_connectivities.size(), 1);
  EXPECT_TRUE(areEqual(cell_to_nodes_connectivities[0], cell_to_nodes));
  // cell to dofs & cell to dofs 2
  auto cell_to_dofs_connectivities = mesh.dofs(cell_family); // returns all IK_Node families connected wih cell_family
  for (auto connectivity : cell_to_dofs_connectivities) {
    std::cout << "Connectivity name " << connectivity.name << std::endl;
  }
  EXPECT_EQ(cell_to_dofs_connectivities.size(), 2);
  EXPECT_TRUE(areEqual(cell_to_dofs_connectivities[0], cell_to_dofs));
  EXPECT_TRUE(areEqual(cell_to_dofs_connectivities[1], cell_to_dofs2));

  // node to cells
  auto node_to_cells_connectivities = mesh.cells(node_family); // returns all IK_Cell families connected wih, node_family
  for (auto connectivity : node_to_cells_connectivities) {
    std::cout << "Connectivity name " << connectivity.name << std::endl;
  }
  EXPECT_EQ(node_to_cells_connectivities.size(), 1);
  EXPECT_TRUE(areEqual(node_to_cells_connectivities[0], node_to_cells));
  // node to dofs
  auto node_to_dofs_connectivities = mesh.dofs(node_family); // returns all IK_Cell families connected wih, node_family
  for (auto connectivity : node_to_dofs_connectivities) {
    std::cout << "Connectivity name " << connectivity.name << std::endl;
  }
  EXPECT_EQ(node_to_dofs_connectivities.size(), 1);
  EXPECT_TRUE(areEqual(node_to_dofs_connectivities[0], node_to_dofs));
  // check asking non-existing connectivity
  EXPECT_THROW(mesh.getConnectivity(cell_family, node_family, "unexisting_connectivity"), std::invalid_argument);
  // check asking an existing connectivity with wrong families
  EXPECT_THROW(mesh.getConnectivity(cell_family, node_family, node_to_cells_connectivity_name), std::invalid_argument);
  EXPECT_THROW(mesh.getConnectivity(cell_family, dof_family, cell_to_nodes_connectivity_name), std::invalid_argument);
  EXPECT_THROW(mesh.getConnectivity(dof_family, node_family, cell_to_nodes_connectivity_name), std::invalid_argument);
  auto& cell_family2 = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family2");
  auto& node_family2 = mesh.addFamily(Neo::ItemKind::IK_Node, "node_family2");
  EXPECT_THROW(mesh.getConnectivity(node_family2, cell_family2, cell_to_nodes_connectivity_name), std::invalid_argument);
  // check getConnectivities: all the connectivity attached to a same source family
  auto connectivities = mesh.getConnectivities(cell_family);
  auto nb_cell_connectivities = 3;
  EXPECT_EQ(connectivities.size(), nb_cell_connectivities);
  std::vector<std::string> target_family_names{node_family.name(),dof_family.name(),dof_family2.name()};
  EXPECT_TRUE(areEqual(connectivities[0], cell_to_nodes));
  EXPECT_TRUE(areEqual(connectivities[1], cell_to_dofs));
  EXPECT_TRUE(areEqual(connectivities[2], cell_to_dofs2));
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, AddAndChangeItemConnectivity) {
  // Add new connectivities
  auto mesh = Neo::Mesh{ "AddAndChangeItemConnectivityTestMesh" };
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  auto& dof_family = mesh.addFamily(Neo::ItemKind::IK_Dof, "dof_family");
  std::vector<Neo::utils::Int64> cell_uids{ 0, 1 };
  std::vector<Neo::utils::Int64> dof_uids{ 0, 1, 2, 3, 4 };
  std::vector<Neo::utils::Int64> cell_dofs{ 0, 1, 2, 3, 0, 4 };
  auto future_cells = Neo::FutureItemRange{};
  auto future_dofs = Neo::FutureItemRange{};
  mesh.scheduleAddItems(cell_family, cell_uids, future_cells);
  mesh.scheduleAddItems(dof_family, dof_uids, future_dofs);
  // Add en empty connectivity
  Neo::ItemRange empty_range{};
  std::vector<Neo::utils::Int64> empty_dof_uids{};
  mesh.scheduleAddConnectivity(cell_family, empty_range, dof_family, 0,
                               empty_dof_uids, "cell_to_dofs");
  mesh.applyScheduledOperations();
  // It is possible to modify an empty connectivity using ConnectivityOperation::Add (the default)
  mesh.scheduleAddConnectivity(cell_family, future_cells, dof_family, 3,
                               cell_dofs, "cell_to_dofs");
  mesh.applyScheduledOperations();
  // Change an existing connectivity: cell 0 now points to dofs uids {3,4}
  auto cell_lids = cell_family.itemUniqueIdsToLocalids({ 0 });
  Neo::ItemRange cell_range{ cell_lids };
  auto connected_dofs = std::vector<Neo::utils::Int64>{ 3, 4 };
  // First try using ConnectivityOperation::Add an existing connectivity: it fails
  EXPECT_THROW(mesh.scheduleAddConnectivity(cell_family, cell_range, dof_family, 2,
                               connected_dofs, "cell_to_dofs",
                               Neo::Mesh::ConnectivityOperation::Add), std::invalid_argument);
  // Second try using ConnectivityOperation::Add an existing connectivity: it works
  mesh.scheduleAddConnectivity(cell_family, cell_range, dof_family, 2,
                               connected_dofs, "cell_to_dofs",
                               Neo::Mesh::ConnectivityOperation::Modify);
  mesh.applyScheduledOperations();
  // check modification
  cell_dofs = { 3, 4, 3, 0, 4 }; // cell 0 => dof 3&4 (modif), cell1 => dof 3,0,4 (unmodified)
  auto connected_dofs_new_lids = mesh.dofs(cell_family)[0].connectivity_value.constView();
  auto connected_dofs_new_uids = mesh.uniqueIds(dof_family, connected_dofs_new_lids);
  EXPECT_TRUE(std::equal(connected_dofs_new_uids.begin(), connected_dofs_new_uids.end(), cell_dofs.begin()));
  // Try to connect a subpart of added items by index
  {
    cell_uids = { 2, 3, 4 };
    dof_uids = { 5, 6 };
    auto added_cells_future_new = Neo::FutureItemRange{};
    auto added_dofs_future_new = Neo::FutureItemRange{};
    mesh.scheduleAddItems(cell_family, cell_uids, added_cells_future_new);
    mesh.scheduleAddItems(dof_family, dof_uids, added_dofs_future_new);
    // Create a filtered ItemRange containing elements with indexes 0 & 1
    std::vector<int> cell_indexes = { 0, 1 };
    auto filtered_future_cell_range = Neo::make_future_range(added_cells_future_new, cell_indexes);
    mesh.scheduleAddConnectivity(cell_family, filtered_future_cell_range,
                                 dof_family, { 1, 2 }, { 5, 5, 6 },
                                 "cell_to_dofs_new");
    auto end_update = mesh.applyScheduledOperations();
    std::vector<Neo::utils::Int64> connected_dof_uids_ref{ 5, 5, 6 };
    auto added_cells_filtered = filtered_future_cell_range.get(end_update);
    auto added_cells_filtered_uids = mesh.uniqueIds(cell_family, added_cells_filtered.localIds());
    auto added_cells2 = added_cells_future_new.get(end_update);
    auto added_cells2_uids = mesh.uniqueIds(cell_family, added_cells2.localIds());
    std::vector<Neo::utils::Int64> added_cells_filtered_ref = { cell_uids[cell_indexes[0]], cell_uids[cell_indexes[1]] }; // two first uids given
    EXPECT_TRUE(std::equal(added_cells_filtered_uids.begin(), added_cells_filtered_uids.end(), added_cells_filtered_ref.begin()));
    EXPECT_TRUE(std::equal(added_cells2_uids.begin(), added_cells2_uids.end(), cell_uids.begin()));
    auto connected_dof_lids = mesh.getConnectivity(cell_family, dof_family, "cell_to_dofs_new").connectivity_value.constView();
    auto connected_dof_uids = mesh.uniqueIds(dof_family, connected_dof_lids);
    EXPECT_TRUE(std::equal(connected_dof_uids.begin(), connected_dof_uids.end(), connected_dof_uids_ref.begin()));
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
    auto added_cell_new = added_cells_future_new.get(end_update);
    // Enumerate cell_to_dofs_new2
    auto cell_to_dofs_new2 = mesh.getConnectivity(cell_family, dof_family, "cell_to_dofs_new2");
    for (auto cell : added_cell_new) {
      for (auto dof : cell_to_dofs_new2[cell]) {
        std::cout << "cell " << cell << " connected with dof " << dof << std::endl;
      }
    }
    // Enumerate cell_to_dofs_new3
    auto cell_to_dofs_new3 = mesh.getConnectivity(cell_family, dof_family, "cell_to_dofs_new3");
    for (auto cell : added_cell_new) {
      for (auto dof : cell_to_dofs_new3[cell]) {
        std::cout << "cell " << cell << " connected with dof " << dof << std::endl;
      }
    }
    // Check cell_to_dofs_new2
    auto connected_dofs_new2_lids = cell_to_dofs_new2.connectivity_value.constView();
    auto connected_dofs_new2_uids = mesh.uniqueIds(dof_family, connected_dofs_new2_lids);
    std::vector<Neo::utils::Int64> connected_dofs_new2_uids_ref{ 7, 8, 7 };
    EXPECT_TRUE(std::equal(connected_dofs_new2_lids.begin(), connected_dofs_new2_lids.end(), connected_dofs_new2_uids_ref.begin()));
    // Check cell_to_dofs_new3
    auto connected_dofs_new3_lids = cell_to_dofs_new3.connectivity_value.constView();
    auto connected_dofs_new3_uids = mesh.uniqueIds(dof_family, connected_dofs_new3_lids);
    std::vector<Neo::utils::Int64> connected_dofs_new3_uids_ref{ 7, 8, 7, 8, 7 };
    EXPECT_TRUE(std::equal(connected_dofs_new3_lids.begin(), connected_dofs_new3_lids.end(), connected_dofs_new3_uids_ref.begin()));
  }
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, AddItemConnectivityWithSubsteps) {
  // Add new connectivities
  auto mesh = Neo::Mesh{ "AddAndChangeItemConnectivityTestMesh" };
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  auto& dof_family = mesh.addFamily(Neo::ItemKind::IK_Dof, "dof_family");
  std::vector<Neo::utils::Int64> cell_uids1{ 0 };
  std::vector<Neo::utils::Int64> cell_uids2{ 1 };
  std::vector<Neo::utils::Int64> dof_uids{ 0, 1, 2, 3, 4 };
  std::vector<Neo::utils::Int64> cell_dofs1{ 0, 1, 2 };
  std::vector<Neo::utils::Int64> cell_dofs2{ 3, 0, 4 };
  auto future_cells1 = Neo::FutureItemRange{};
  auto future_cells2 = Neo::FutureItemRange{};
  auto future_dofs = Neo::FutureItemRange{};
  mesh.scheduleAddItems(cell_family, cell_uids1, future_cells1);
  mesh.scheduleAddItems(cell_family, cell_uids2, future_cells2);
  mesh.scheduleAddItems(dof_family, dof_uids, future_dofs);
  mesh.scheduleAddConnectivity(cell_family, future_cells2, dof_family, 3,
                               cell_dofs2, "cell_to_dofs");
  mesh.scheduleAddConnectivity(cell_family, future_cells1, dof_family, 3,
                               cell_dofs1, "cell_to_dofs", Neo::Mesh::ConnectivityOperation::Modify);
  mesh.applyScheduledOperations();
  // Check
  auto cell_to_dofs = mesh.dofs(cell_family)[0]; // only one cell to dof connectivity
  auto connected_dofs_lids = cell_to_dofs.connectivity_value.constView();
  auto connected_dofs_uids = mesh.uniqueIds(dof_family, connected_dofs_lids);
  std::vector<Neo::utils::Int64> connected_dofs_uids_ref{ 0, 1, 2, 3, 0, 4 };
  EXPECT_TRUE(std::equal(connected_dofs_lids.begin(), connected_dofs_lids.end(), connected_dofs_uids_ref.begin()));
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, AddMeshOperationAfterAddingItem) {
  auto mesh = Neo::Mesh{ "AddMeshOperationTestMesh" };
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  std::vector<Neo::utils::Int64> cell_uids{ 0, 1, 2, 3, 4 };
  // Schedule items and connectivities add
  auto future_cells = Neo::FutureItemRange{};
  mesh.scheduleAddItems(cell_family, cell_uids, future_cells);
  // Schedule a user operation depending on dof lids : compute a property storing lids sum
  cell_family.addScalarProperty<Neo::utils::Int32>("lid_sum");
  mesh.scheduleAddMeshOperation(cell_family, cell_family.lidPropName(), cell_family, "lid_sum",
                                [](Neo::ItemLidsProperty const& cell_lids, Neo::ScalarPropertyT<Neo::utils::Int32>& lid_sum) {
                                  lid_sum.set(0);
                                  for (auto const& cell_lid : cell_lids.values()) {
                                    lid_sum() += cell_lid;
                                  }
                                });
  mesh.applyScheduledOperations();
  // get algo result
  auto& lid_sum_property = cell_family.getConcreteProperty<Neo::ScalarPropertyT<Neo::utils::Int32>>("lid_sum");
  std::vector<Neo::utils::Int32> cell_lids = cell_family.itemUniqueIdsToLocalids(cell_uids);
  auto lid_sum_ref = std::accumulate(cell_lids.begin(), cell_lids.end(), 0);
  Neo::print() << "--- lid sum = " << lid_sum_property() << " lid sum ref value " << lid_sum_ref << std::endl;
  EXPECT_EQ(lid_sum_property(), lid_sum_ref);
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, AddMeshOperationAfterAddingConnectivity) {
  auto mesh = Neo::Mesh{ "AddMeshOperationTestMesh" };
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  auto& dof_family = mesh.addFamily(Neo::ItemKind::IK_Dof, "dof_family");
  std::vector<Neo::utils::Int64> cell_uids{ 0, 1 };
  std::vector<Neo::utils::Int64> dof_uids{ 0, 1, 2, 3, 4 };
  std::vector<Neo::utils::Int64> cell_dofs{ 0, 1, 2, 3, 0, 4 };
  // Schedule items and connectivities add
  auto future_cells = Neo::FutureItemRange{};
  auto future_dofs = Neo::FutureItemRange{};
  mesh.scheduleAddItems(cell_family, cell_uids, future_cells);
  mesh.scheduleAddItems(dof_family, dof_uids, future_dofs);
  mesh.scheduleAddConnectivity(cell_family, future_cells, dof_family, 3,
                               cell_dofs, "cell_to_dofs");
  // Schedule a user operation depending on cell dof connectivity :
  // create a property containing the number of cells connected to a dof
  dof_family.addMeshScalarProperty<Neo::utils::Int32>("nb_connected_cells");
  mesh.scheduleAddMeshOperation(cell_family, mesh.getConnectivity(cell_family, dof_family, "cell_to_dofs").connectivity_value.name(), dof_family, "nb_connected_cells",
                                [&dof_family, &cell_family](Neo::Mesh::ConnectivityPropertyType const& cell_to_dofs, Neo::MeshScalarPropertyT<Neo::utils::Int32>& nb_connected_cells) {
                                  nb_connected_cells.init(dof_family.all(), 0);
                                  for (auto cell : cell_family.all()) {
                                    for (auto dof : cell_to_dofs[cell]) {
                                      nb_connected_cells[dof] += 1;
                                    }
                                  }
                                  nb_connected_cells.debugPrint();
                                });
  mesh.applyScheduledOperations();
  // get algo result
  auto& nb_connected_cell_property = dof_family.getConcreteProperty<Neo::MeshScalarPropertyT<Neo::utils::Int32>>("nb_connected_cells");
  std::vector<Neo::utils::Int32> nb_connected_cell_ref{ 2, 1, 1, 1, 1 };
  nb_connected_cell_property.debugPrint();
  EXPECT_TRUE(std::equal(nb_connected_cell_property.begin(), nb_connected_cell_property.end(), nb_connected_cell_ref.begin()));
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, AddMeshOperationAfterSettingCoordinates) {
  auto mesh = Neo::Mesh{ "SetNodeCoordsTestMesh" };
  auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "NodeFamily");
  auto added_nodes = Neo::FutureItemRange{};
  std::vector<Neo::utils::Int64> node_uids{ 1, 10, 100 };
  mesh.scheduleAddItems(node_family, node_uids, added_nodes);
  std::vector<Neo::utils::Real3> node_coords{ { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } };
  mesh.scheduleSetItemCoords(node_family, added_nodes, node_coords);
  // Schedule a user operation depending on node coordinates : sum the coordinates
  node_family.addScalarProperty<Neo::utils::Real3>("coord_sum");
  mesh.scheduleAddMeshOperation(node_family, mesh._itemCoordPropertyName(node_family), node_family, "coord_sum",
                                [](Neo::Mesh::CoordPropertyType const& coords,
                                   Neo::ScalarPropertyT<Neo::utils::Real3>& coord_sum) {
                                  std::for_each(coords.begin(), coords.end(), [&coord_sum](auto const& coord) {
                                    coord_sum() += coord;
                                  });
                                });
  mesh.applyScheduledOperations();
  auto& coord_sum = node_family.getConcreteProperty<Neo::ScalarPropertyT<Neo::utils::Real3>>("coord_sum");
  auto& computed_sum = coord_sum();
  auto ref_sum = Neo::utils::Real3{ 1, 1, 1 };
  EXPECT_EQ(computed_sum, ref_sum);
}

/*---------------------------------------------------------------------------*/

TEST(NeoMeshApiTest, UpdateConnectivityAndRemoveIsolatedItemsAfterSourceFamilyChange) {
  auto mesh = Neo::Mesh{ "RemoveIsolatedItemsMesh" };
  auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node, "NodeFamily");
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
  std::vector<Neo::utils::Int64> node_uids{ 0, 1, 2, 3 };
  std::vector<Neo::utils::Int64> cell_uids{ 0, 1 };
  auto future_nodes = Neo::FutureItemRange{};
  auto future_cells = Neo::FutureItemRange{};
  mesh.scheduleAddItems(node_family, node_uids, future_nodes);
  mesh.scheduleAddItems(cell_family, cell_uids, future_cells);
  // Create connectivity (fictive mesh) cells with 4 nodes
  std::string cell_to_nodes_connectivity_name{ "cell_to_nodes" };
  std::string node_to_cells_connectivity_name{ "node_to_cells" };

  // Connectivity cell to nodes
  {
    auto nb_node_per_cell = 3;
    std::vector<Neo::utils::Int64> cell_nodes{ 0, 1, 2, 1, 3, 2 };
    mesh.scheduleAddConnectivity(cell_family, future_cells, node_family,
                                 nb_node_per_cell, cell_nodes,
                                 cell_to_nodes_connectivity_name);
  }
  // Connectivity nodes to cells
  mesh.scheduleAddConnectivity(node_family, future_nodes, cell_family,
                               { 1, 2, 2, 1 }, { 0, 0, 1, 0, 1, 1 },
                               node_to_cells_connectivity_name);
  auto end_update = mesh.applyScheduledOperations();
  auto nodes = future_nodes.get(end_update);
  auto cells = future_cells.get(end_update);
  // Check connectivities
  auto cell_2_nodes = mesh.getConnectivity(cell_family, node_family, cell_to_nodes_connectivity_name);
  for (auto cell : cells) {
    std::cout << "cell " << cell << " has nodes " << cell_2_nodes[cell] << std::endl;
  }
  std::cout << "nb cell before remove " << cell_family.nbElements() << std::endl;
  EXPECT_EQ(cell_family.nbElements(), 2);
  EXPECT_EQ(cell_2_nodes[0].size(), 3);
  EXPECT_EQ(cell_2_nodes[1].size(), 3);
  // Remove cell 0
  std::vector<Neo::utils::Int64> removed_cell_uids{ 0 };
  mesh.scheduleRemoveItems(cell_family, removed_cell_uids);
  mesh.applyScheduledOperations();
  // Check connectivities
  auto node_2_cells = mesh.getConnectivity(node_family, cell_family, node_to_cells_connectivity_name);
  auto cells_new = cell_family.all();
  std::cout << "nb cell after remove " << cells_new.size() << std::endl;
  // Cell to nodes
  for (auto cell : cells_new) {
    std::cout << "cell " << cell << " has nodes " << cell_2_nodes[cell] << std::endl;
  }
  EXPECT_EQ(cells_new.size(), 1);
  EXPECT_EQ(cell_family.nbElements(), 1);
  EXPECT_EQ(cell_2_nodes[0].size(), 0);
  EXPECT_EQ(cell_2_nodes[1].size(), 3);

  // Isolated node uid(0) must be removed
  auto const& node_uids_prop = mesh.getItemUidsProperty(node_family);
  EXPECT_EQ(node_family.nbElements(), 3);
  for (auto node:node_family.all()) {
    EXPECT_FALSE(node_uids_prop[node] == 0);
  }

  // Get the 0th cell back
  Neo::FutureItemRange future_cells2;
  mesh.scheduleAddItems(cell_family, removed_cell_uids, future_cells2);
  auto end_update2 = mesh.applyScheduledOperations();
  // Check cell 0 has no connected items
  auto cells_new2 = cell_family.all();
  std::cout << "nb cell after re-add " << cells_new2.size() << std::endl;
  for (auto cell : cells_new2) {
    std::cout << "cell " << cell << " has nodes " << cell_2_nodes[cell] << std::endl;
  }
  EXPECT_EQ(cells_new2.size(), 2);
  EXPECT_EQ(cell_family.nbElements(), 2);
  EXPECT_EQ(cell_2_nodes[0].size(), 0);
  EXPECT_EQ(cell_2_nodes[1].size(), 3);

  // Finally Remove all cells: must remove connected nodes
  removed_cell_uids.push_back(1);
  mesh.scheduleRemoveItems(cell_family, removed_cell_uids);
  mesh.applyScheduledOperations();
  // Check nodes are removed
  EXPECT_EQ(node_family.nbElements(), 0);
}

/*---------------------------------------------------------------------------*/