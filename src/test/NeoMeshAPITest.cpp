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
  auto node_family = mesh.addFamily(Neo::ItemKind::IK_Node,"Node Family");
  std::cout << "Create family " << node_family.name() << " with item kind " << Neo::utils::itemKindName(node_family.itemKind()) << std::endl;
  auto dof_family  = mesh.addFamily(Neo::ItemKind::IK_Dof,"DoFFamily");
  std::cout << "Create family " << dof_family.name() << " with item kind " << Neo::utils::itemKindName(dof_family.itemKind()) << std::endl;
}