//
// Created by dechaiss on 5/7/20.
//

/*-------------------------
 * Neo library
 * Polyhedral mesh test
 * sdc (C) 2020
 *
 *-------------------------
 */

#include "gtest/gtest.h"
#include "neo/Mesh.h"
#include "neo/Neo.h"

TEST(NeoMeshApiTest,MeshApiCreationTest)
{
    auto mesh_name = "MeshTest";
    auto mesh = Neo::Mesh{mesh_name};
    std::cout << "Creating mesh " << mesh.name();
    EXPECT_EQ(mesh_name, mesh.name());
}

TEST(NeoMeshApiTest,AddFamilyTest)
{
  auto mesh = Neo::Mesh{"AddFamilyTestMesh"};
  auto cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell,"CellFamily");
  std::cout << "Create family " << cell_family.name() << " with item kind " << Neo::utils::itemKindName(cell_family.itemKind()) << std::endl;
  EXPECT_EQ(cell_family.name(),"CellFamily");
  EXPECT_EQ(cell_family.itemKind(),Neo::ItemKind::IK_Cell);
  auto node_family = mesh.addFamily(Neo::ItemKind::IK_Node,"NodeFamily");
  std::cout << "Create family " << node_family.name() << " with item kind " << Neo::utils::itemKindName(node_family.itemKind()) << std::endl;
  EXPECT_EQ(node_family.name(),"NodeFamily");
  EXPECT_EQ(node_family.itemKind(),Neo::ItemKind::IK_Node);
  auto dof_family  = mesh.addFamily(Neo::ItemKind::IK_Dof,"DoFFamily");
  std::cout << "Create family " << dof_family.name() << " with item kind " << Neo::utils::itemKindName(dof_family.itemKind()) << std::endl;
  EXPECT_EQ(dof_family.name(),"DoFFamily");
  EXPECT_EQ(dof_family.itemKind(),Neo::ItemKind::IK_Dof);
}

TEST(NeoMeshApiTest,AddItemTest)
{
  auto mesh = Neo::Mesh{"AddItemsTestMesh"};
  auto cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell,"CellFamily");
  auto added_cells = Neo::FutureItemRange{};
  auto added_cells2 = Neo::FutureItemRange{};
  auto added_cells3 = Neo::FutureItemRange{};
  {
    // check lifetime
    std::vector<Neo::utils::Int64> cell_uids{1,10,100};
    mesh.scheduleAddItems(cell_family,cell_uids,added_cells2);
  }
  std::vector<Neo::utils::Int64> cell_uids2{1,10,100};
  mesh.scheduleAddItems(cell_family,{2,3,4},added_cells); // memory stealing API
  mesh.scheduleAddItems(cell_family,std::move(cell_uids2),added_cells3);// memory stealing API
  auto item_range_unlocker = mesh.applyScheduledOperations();
  auto new_cells = added_cells.get(item_range_unlocker);
  for (auto item : new_cells) {
    std::cout << "Added local id " << item << std::endl;
  }
  auto new_cells2 = added_cells2.get(item_range_unlocker);
  for (auto item : new_cells2) {
    std::cout << "Added local id " << item << std::endl;
  }
  auto new_cells3 = added_cells3.get(item_range_unlocker);
  for (auto item : new_cells3) {
    std::cout << "Added local id " << item << std::endl;
  }
  std::vector<Neo::utils::Int64> cell_uids{1,10,100};
  EXPECT_EQ(cell_uids.size()+3,new_cells.size()+new_cells2.size());
  auto& cell_uid_property = cell_family.getConcreteProperty<Neo::Mesh::UidPropertyType>
      (mesh.uniqueIdPropertyName(cell_family.name()));
  std::vector<Neo::utils::Int64> expected_uids{2,3,4};
  auto i = 0;
  for (auto item : new_cells) {
    std::cout << "Added unique id " << cell_uid_property[item] << std::endl;
    EXPECT_EQ(expected_uids[i++],cell_uid_property[item]);
  }
  i = 0;
  for (auto item : new_cells2) {
    std::cout << "Added unique id " << cell_uid_property[item] << std::endl;
    EXPECT_EQ(cell_uids[i++],cell_uid_property[item]);
  }
  i = 0;
  for (auto item : new_cells3) {
    std::cout << "Added unique id " << cell_uid_property[item] << std::endl;
    EXPECT_EQ(cell_uids[i++],cell_uid_property[item]);
  }
  // Get uids view
  Neo::PropertyView<Neo::utils::Int64> uid_view = cell_uid_property.view(new_cells);
  // Print uids
  for (auto i = 0; i < new_cells.size(); ++i) {
    std::cout << "uid view index " << i << " = " << uid_view[i]<< std::endl;
  }
}

TEST(NeoMeshApiTest,SetNodeCoordsTest)
{
  auto mesh = Neo::Mesh{"SetNodeCoordsTestMesh"};
  auto node_family = mesh.addFamily(Neo::ItemKind::IK_Node,"NodeFamily");
  auto added_nodes  = Neo::FutureItemRange{};
  auto added_nodes2 = Neo::FutureItemRange{};
  std::vector<Neo::utils::Int64> node_uids{1,10,100};
  mesh.scheduleAddItems(node_family,node_uids,added_nodes);
  mesh.scheduleAddItems(node_family,{0,5},added_nodes2);
  {
    std::vector<Neo::utils::Real3> node_coords{{0,0,0},{0,0,1},{0,1,0}};
    mesh.scheduleSetItemCoords(node_family,added_nodes,node_coords);
  } // Check memory
  mesh.scheduleSetItemCoords(node_family, added_nodes2,{{1,0,0},{1,1,1}});// memory stealing API
  auto item_range_unlocker  = mesh.applyScheduledOperations();
  auto& added_node_range  = added_nodes.get(item_range_unlocker);
  auto& added_node_range2 = added_nodes2.get(item_range_unlocker);
  auto& node_coord_property = mesh.getItemCoordProperty(node_family);
  auto const& node_coord_property_const = mesh.getItemCoordProperty(node_family);
  auto i = 0;
  std::vector<Neo::utils::Real3> node_coords{{0,0,0},{0,0,1},{0,1,0}};
  for (auto item : added_node_range) {
    std::cout << "Node coord for item " << item << " = " << node_coord_property_const[item]<< std::endl;
    EXPECT_EQ(node_coord_property_const[item].x,node_coords[i].x);
    EXPECT_EQ(node_coord_property_const[item].y,node_coords[i].y);
    EXPECT_EQ(node_coord_property_const[item].z,node_coords[i++].z);
  }
  i = 0;
  node_coords = {{1,0,0},{1,1,1}};
  for (auto item : added_node_range2) {
    std::cout << "Node coord for item " << item << " = " << node_coord_property_const[item]<< std::endl;
    EXPECT_EQ(node_coord_property_const[item].x,node_coords[i].x);
    EXPECT_EQ(node_coord_property_const[item].y,node_coords[i].y);
    EXPECT_EQ(node_coord_property_const[item].z,node_coords[i++].z);
  }
  // Change coords
  node_coords = {{0,0,0},{0,0,-1},{0,-1,0}};
  i = 0;
  for (auto item : added_node_range) {
    node_coord_property[item] = node_coords[i];
    EXPECT_EQ(node_coord_property_const[item].x,node_coords[i].x);
    EXPECT_EQ(node_coord_property_const[item].y,node_coords[i].y);
    EXPECT_EQ(node_coord_property_const[item].z,node_coords[i++].z);
  }
  // Check throw for non existing coord property
  auto cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell,"CellFamily");
  EXPECT_THROW(mesh.getItemCoordProperty(cell_family),std::invalid_argument);
}

TEST(NeoMeshApiTest,AddItemConnectivity)
{
  auto mesh = Neo::Mesh{"AddItemConnectivityTestMesh"};
  auto node_family = mesh.addFamily(Neo::ItemKind::IK_Node,"NodeFamily");
  auto cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell,"CellFamily");
  auto dof_family  = mesh.addFamily(Neo::ItemKind::IK_Dof,"DoFFamily");
  std::vector<Neo::utils::Int64> node_uids {0,1,2,3,4,5};
  std::vector<Neo::utils::Int64> cell_uids{0,1};
  std::vector<Neo::utils::Int64> dof_uids{0,1,2,3,4};
  auto added_nodes_future = Neo::FutureItemRange{};
  auto added_cells_future = Neo::FutureItemRange{};
  auto added_dofs_future = Neo::FutureItemRange{};
  mesh.scheduleAddItems(node_family, node_uids, added_nodes_future);
  mesh.scheduleAddItems(cell_family, cell_uids, added_cells_future);
  mesh.scheduleAddItems(dof_family, dof_uids, added_dofs_future);
  // Create connectivity (fictive mesh) cells with 4 nodes
  std::string cell_to_nodes_connectivity_name{"cell_to_nodes"};
  std::string cell_to_dofs_connectivity_name{"cell_to_dofs"};
  std::string node_to_cells_connectivity_name{"node_to_cell"};
  std::string node_to_dofs_connectivity_name{"node_to_dofs"};

  auto nb_node_per_cell = 4;
  {
    std::vector<Neo::utils::Int64> cell_nodes {0,1,2,3,5,0,3,4};
    mesh.scheduleAddConnectivity(cell_family, added_cells_future,node_family,nb_node_per_cell,cell_nodes,cell_to_nodes_connectivity_name);
  } // check memory
  // Connectivity cell to dof
  std::vector<int> nb_dof_per_cell{3,2};
  std::vector<Neo::utils::Int64> cell_dofs {0,3,4,2,1,};
  std::vector<Neo::utils::Int64> cell_dofs_ref{cell_dofs};
  mesh.scheduleAddConnectivity(cell_family, added_cells_future,dof_family,std::move(nb_dof_per_cell),std::move(cell_dofs),cell_to_dofs_connectivity_name);
  // apply
  auto added_range_unlocker = mesh.applyScheduledOperations();
  // Add further connectivity
  auto added_nodes = added_nodes_future.get(added_range_unlocker);
  mesh.scheduleAddConnectivity(node_family, added_nodes,cell_family,{2,1,1,2,1,1},{0,1,0,0,0,1,1,1},node_to_cells_connectivity_name);
  auto nb_dof_per_node =1;
  mesh.scheduleAddConnectivity(node_family, added_nodes,dof_family,nb_dof_per_node,{0,1,2,3,4,0},node_to_dofs_connectivity_name);
  auto range_getter = mesh.applyScheduledOperations();
  auto added_cells = added_cells_future.get(range_getter);
  // check connectivity
  auto const cell_to_nodes = mesh.getConnectivity(cell_family, node_family,cell_to_nodes_connectivity_name);
  EXPECT_EQ(cell_to_nodes_connectivity_name,cell_to_nodes.connectivity_name);
  EXPECT_EQ(&cell_family,&cell_to_nodes.source_family);
  EXPECT_EQ(&node_family,&cell_to_nodes.target_family);
  for (auto const cell : added_cells){
    std::cout << "cell lid " << cell << " nodes lids " << cell_to_nodes[cell] << std::endl;
  }
  auto const cell_to_dofs = mesh.getConnectivity(cell_family,dof_family,cell_to_dofs_connectivity_name);
  EXPECT_EQ(cell_to_dofs_connectivity_name,cell_to_dofs.connectivity_name);
  EXPECT_EQ(&cell_family,&cell_to_dofs.source_family);
  EXPECT_EQ(&dof_family,&cell_to_dofs.target_family);
  std::vector<Neo::utils::Int32> cell_dofs_lids = dof_family._lidProp().operator[](cell_dofs_ref);
  for (auto const cell : added_cells){
    auto current_cell_dofs = cell_to_dofs[cell];
    std::cout << "cell lid " << cell << " dofs lids " << current_cell_dofs << std::endl;
    auto i = 0;
    for (auto dof_lid : current_cell_dofs) {
      EXPECT_EQ(dof_lid,current_cell_dofs[i++]);
    }
  }
}