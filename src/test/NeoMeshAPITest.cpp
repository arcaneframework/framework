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
  auto added_cells = Neo::ScheduledItemRange{};
  auto added_cells2 = Neo::ScheduledItemRange{};
  std::vector<Neo::utils::Int64> cell_uids{1,10,100};
  mesh.scheduleAddItems(cell_family,{2,3,4},added_cells); // memory stealing API
  mesh.scheduleAddItems(cell_family,cell_uids,added_cells2);
  auto item_range_unlocker = mesh.applyScheduledOperations();
  auto new_cells = added_cells.get(item_range_unlocker);
  for (auto item : new_cells) {
    std::cout << "Added local id " << item << std::endl;
  }
  auto new_cells2 = added_cells2.get(item_range_unlocker);
  for (auto item : new_cells2) {
    std::cout << "Added local id " << item << std::endl;
  }
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
  auto added_nodes  = Neo::ScheduledItemRange{};
  auto added_nodes2 = Neo::ScheduledItemRange{};
  std::vector<Neo::utils::Int64> node_uids{1,10,100};
  mesh.scheduleAddItems(node_family,node_uids,added_nodes);
  mesh.scheduleAddItems(node_family,{0,5},added_nodes2);
  std::vector<Neo::utils::Real3> node_coords{{0,0,0},{0,0,1},{0,1,0}};
  mesh.scheduleSetNodeCoords(node_family,added_nodes,node_coords);
  mesh.scheduleSetNodeCoords(node_family,added_nodes2,{{1,0,0},{1,1,1}});// memory stealing API
}