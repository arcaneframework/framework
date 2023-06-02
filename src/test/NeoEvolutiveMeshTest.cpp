// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NeoEvolutiveMeshTest.cpp                        (C) 2000-2023             */
/*                                                                           */
/* First very basic mesh evolution test                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "neo/Neo.h"
#include "neo/Mesh.h"
#include "gtest/gtest.h"

//----------------------------------------------------------------------------/

auto& addCellFamily(Neo::MeshBase & mesh,std::string family_name){
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell,std::move(family_name));
  cell_family.addProperty<Neo::utils::Int64>(family_name+"_uids");
  return cell_family;
}

auto& addNodeFamily(Neo::MeshBase & mesh,std::string family_name){
  auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node,std::move(family_name));
  node_family.addProperty<Neo::utils::Int64>(family_name+"_uids");
  return node_family;
}

auto& addFaceFamily(Neo::MeshBase & mesh,std::string family_name){
  auto& face_family = mesh.addFamily(Neo::ItemKind::IK_Face,std::move(family_name));
  face_family.addProperty<Neo::utils::Int64>(family_name+"_uids");
  return face_family;
}

//----------------------------------------------------------------------------/

// todo create another signature where uids are moved, or make clear that they potentially are...
void addItems(Neo::MeshBase & mesh, Neo::Family& family, std::vector<Neo::utils::Int64>& uids, Neo::FutureItemRange & added_item_range)
{
  auto& added_items = added_item_range.new_items;
  // Add items
  mesh.addAlgorithm(  Neo::OutProperty{family,family.lidPropName()},
  [&family,&uids,&added_items](Neo::ItemLidsProperty & lids_property){
    std::cout << "Algorithm: create items in family " << family.m_name << std::endl;
    added_items = lids_property.append(uids);
    lids_property.debugPrint();
    std::cout << "Inserted item range : " << added_items;
  });
  // register their uids
  auto uid_property_name = family.m_name+"_uids";
  mesh.addAlgorithm(
      Neo::InProperty{family,family.lidPropName()},
      Neo::OutProperty{family, uid_property_name},
      [&family,&uids,&added_items](Neo::ItemLidsProperty const& item_lids_property,
                                Neo::PropertyT<Neo::utils::Int64>& item_uids_property){
        std::cout << "Algorithm: register item uids for family " << family.m_name << std::endl;
        if (item_uids_property.isInitializableFrom(added_items))
          item_uids_property.init(added_items,std::move(uids)); // init can steal the input values
        else
          item_uids_property.append(added_items, uids);
        item_uids_property.debugPrint();
      });// need to add a property check for existing uid
}

//----------------------------------------------------------------------------/

// todo same interface with nb_connected_item_per_item as an array
void addConnectivity(Neo::MeshBase &mesh, Neo::Family &source_family,
                     Neo::ItemRange &source_items,
                     Neo::Family& target_family,
                     int nb_connected_item_per_item,
                     std::vector<Neo::utils::Int64>& connected_item_uids) {
  // add connectivity property if doesn't exist
  std::string connectivity_name = source_family.m_name + "to" + target_family.m_name + "_connectivity";
  source_family.addArrayProperty<Neo::utils::Int32>(connectivity_name);
  mesh.addAlgorithm(
  Neo::InProperty{ source_family, source_family.lidPropName(), Neo::PropertyStatus::ExistingProperty },
  Neo::InProperty{ target_family, target_family.lidPropName(), Neo::PropertyStatus::ExistingProperty },
  Neo::OutProperty{ source_family, connectivity_name },
  [&connected_item_uids, nb_connected_item_per_item, &source_items, &source_family, &target_family](Neo::ItemLidsProperty const& source_family_lids_property,
                                                                                                    Neo::ItemLidsProperty const& target_family_lids_property,
                                                                                                    Neo::ArrayProperty<Neo::utils::Int32>& source2target) {
    std::cout << "Algorithm: register connectivity between " << source_family.m_name << "  and  " << target_family.m_name << std::endl;
    auto connected_item_lids = target_family_lids_property[connected_item_uids];
    std::vector<int> nb_connected_item_per_item_array(source_items.size(), nb_connected_item_per_item);
    if (source2target.isInitializableFrom(source_items)) {
      source2target.resize(std::move(nb_connected_item_per_item_array));
      source2target.init(source_items, std::move(connected_item_lids));
    }
    else {
      source2target.append(source_items, connected_item_lids, nb_connected_item_per_item_array);
    }
    source2target.debugPrint();
  });
}

void updateConnectivity(Neo::MeshBase& mesh, Neo::Family & source_family,Neo::Family &target_family){
  // handle target item removal in connectivity
  const std::string removed_item_property_name{"removed_"+target_family.m_name+"_items"};
  source_family.addProperty<Neo::utils::Int32>(removed_item_property_name);
  std::string connectivity_name = source_family.m_name + "to" + target_family.m_name + "_connectivity";
  mesh.addAlgorithm(
      Neo::InProperty{target_family,removed_item_property_name},
      Neo::OutProperty{source_family,connectivity_name},
      [&source_family,&target_family](
          Neo::PropertyT<Neo::utils::Int32> const& target_family_removed_items,
          Neo::ArrayProperty<Neo::utils::Int32> &connectivity){
        std::cout << "Algorithm update connectivity after remove " << connectivity.m_name << std::endl;
        for (auto item : source_family.all()) {
          auto connected_items = connectivity[item];
          for (auto& connected_item : connected_items){
            if (connected_item != Neo::utils::NULL_ITEM_LID && target_family_removed_items[connected_item] == 1) {
              std::cout << "modify connected item : "<< connected_item << " in family " << target_family.m_name << std::endl;
              connected_item = Neo::utils::NULL_ITEM_LID;
            }
          }
        }
      });
}

//----------------------------------------------------------------------------/

void addConnectivity(Neo::MeshBase &mesh, Neo::Family &source_family,
                     Neo::FutureItemRange &source_items,
                     Neo::Family& target_family,
                     int nb_connected_item_per_item,
                     std::vector<Neo::utils::Int64>& connected_item_uids)
{
  addConnectivity(mesh, source_family, source_items.new_items, target_family,
                  nb_connected_item_per_item, connected_item_uids);
}

//----------------------------------------------------------------------------/

// todo : define 2 signatures to indicate eventual memory stealing...?
void setNodeCoords(Neo::MeshBase & mesh, Neo::Family& node_family, Neo::FutureItemRange & added_node_range, std::vector<Neo::utils::Real3>& node_coords){
  node_family.addProperty<Neo::utils::Real3>(std::string("node_coords"));
  auto& added_nodes = added_node_range.new_items;
  mesh.addAlgorithm(
      Neo::InProperty{node_family,node_family.lidPropName()},
      Neo::OutProperty{node_family,"node_coords"},
      [&node_coords,&added_nodes](Neo::ItemLidsProperty const& node_lids_property,
                                  Neo::PropertyT<Neo::utils::Real3> & node_coords_property){
        std::cout << "Algorithm: register node coords" << std::endl;
        if (node_coords_property.isInitializableFrom(added_nodes))  node_coords_property.init(added_nodes,std::move(node_coords)); // init can steal the input values
        else node_coords_property.append(added_nodes, node_coords);
        node_coords_property.debugPrint();
      });
}

//----------------------------------------------------------------------------/

void moveNodes(Neo::MeshBase & mesh, Neo::Family& node_family, std::vector<Neo::utils::Int64>const& node_uids, std::vector<Neo::utils::Real3>& node_coords){
  mesh.addAlgorithm(
      Neo::InProperty{node_family,node_family.lidPropName()},
      Neo::OutProperty{node_family,"node_coords"},
      [&node_uids,&node_coords](Neo::ItemLidsProperty const& node_lids_property,
                                  Neo::PropertyT<Neo::utils::Real3> & node_coords_property){
        std::cout << "Algorithm: change node coords" << std::endl;
        // get range from uids and append
        auto moved_node_range = Neo::ItemRange{Neo::ItemLocalIds::getIndexes(node_lids_property[node_uids])};
        node_coords_property.append(moved_node_range, node_coords);
        node_coords_property.debugPrint();
      });
}

//----------------------------------------------------------------------------/

void removeItems(Neo::MeshBase & mesh, Neo::Family& family, std::vector<Neo::utils::Int64> const& removed_item_uids){
  const std::string removed_item_property_name{"removed_"+family.m_name+"_items"};
  // Add an algo to clear removed_items property at the beginning of a mesh update
  // This algo will be executed before remove
  const std::string ok_to_start_remove_property_name = "ok_to_start_remove_property";
  family.addProperty<Neo::utils::Int32>(ok_to_start_remove_property_name);
  family.addProperty<Neo::utils::Int32>(removed_item_property_name);
  mesh.addAlgorithm(
      Neo::OutProperty{family,removed_item_property_name},
      Neo::OutProperty{family,ok_to_start_remove_property_name},
      [&family](Neo::PropertyT<Neo::utils::Int32>& removed_item_property,
                Neo::PropertyT<Neo::utils::Int32>& ok_to_start_remove_property){
        std::cout << "Algorithm : clear remove item property for family " << family.m_name<< std::endl;
        removed_item_property.init(family.all(), 0);
        ok_to_start_remove_property.init(family.all(), 1);
      });
  // Remove item algo
  mesh.addAlgorithm(
      Neo::OutProperty{family,family.lidPropName()},
      Neo::OutProperty{family,removed_item_property_name},
      [&removed_item_uids, &family](
          Neo::ItemLidsProperty& item_lids_property,
          Neo::PropertyT<Neo::utils::Int32 > & removed_item_property){
        std::cout << "Algorithm: remove items in " << family.m_name << std::endl;
        auto removed_items = item_lids_property.remove(removed_item_uids);
        item_lids_property.debugPrint();
        std::cout << "removed item range : " << removed_items;
        // Store removed items in internal_end_of_remove_tag
        removed_item_property.init(family.all(),0);
        for (auto removed_item : removed_items) {
          removed_item_property[removed_item] = 1;
        }
      });
}

//----------------------------------------------------------------------------/

static const std::string cell_family_name {"CellFamily"};
static const std::string face_family_name {"FaceFamily"};
static const std::string node_family_name {"NodeFamily"};

//----------------------------------------------------------------------------/

void addCells(Neo::MeshBase &mesh){
  auto& cell_family = addCellFamily(mesh,cell_family_name);
  auto& node_family = addNodeFamily(mesh,node_family_name);
  auto& face_family = addFaceFamily(mesh,face_family_name);
  std::vector<Neo::utils::Int64> node_uids{0,1,2,3,4,5,6,7,8,9,10,11};
  std::vector<Neo::utils::Int64> cell_uids{0,1,2,3};
  std::vector<Neo::utils::Int64> face_uids{0,1,2,3,4,5,6,7,8,9};

  std::vector<Neo::utils::Real3> node_coords{{0,0,-2},{0,2,-2},{0,2,-2},
                                             {0,3,-2},{0,4,-2},{0,5,-2},
                                             {0,0,-2},{0,2,-2},{0,2,-2},
                                             {0,3,-2},{0,4,-2},{0,5,-2}};

  std::vector<Neo::utils::Int64> cell_nodes{0,1,7,6,
                                            2,3,9,8,
                                            3,4,10,9,
                                            4,5,11,10};

  std::vector<Neo::utils::Int64> face_nodes{6,7,8,9,9,10,10,11,1,7,2,8,0,1,2,3,3,4,4,5};

  auto added_cells = Neo::FutureItemRange{};
  auto added_nodes = Neo::FutureItemRange{};
  auto added_faces = Neo::FutureItemRange{};
  addItems(mesh, cell_family, cell_uids, added_cells);
  addItems(mesh, node_family, node_uids, added_nodes);
  addItems(mesh, face_family, face_uids, added_faces);
  setNodeCoords(mesh, node_family, added_nodes, node_coords);
  auto nb_node_per_cell = 4;
  addConnectivity(mesh, cell_family, added_cells, node_family, nb_node_per_cell, cell_nodes);
  auto nb_node_per_face = 2;
  addConnectivity(mesh, face_family, added_faces, node_family, nb_node_per_face, face_nodes);
  auto valid_mesh_state =
      mesh.applyAlgorithms();// retourner un objet qui dévérouille la range
  auto new_cells = added_cells.get(valid_mesh_state);
  auto new_nodes = added_nodes.get(valid_mesh_state);
  auto new_faces = added_faces.get(valid_mesh_state);
  std::cout << "Added cells range after applyAlgorithms: " << new_cells;
  std::cout << "Added nodes range after applyAlgorithms: " << new_nodes;
  std::cout << "Added faces range after applyAlgorithms: " << new_faces;
}

//----------------------------------------------------------------------------/

TEST(EvolutiveMeshTest,AddCells)
{
  auto mesh = Neo::MeshBase{"evolutive_neo_mesh"};
  addCells(mesh);
}

//----------------------------------------------------------------------------/

TEST(EvolutiveMeshTest,MoveNodes)
{
  std::cout << "Move node test " << std::endl;
  auto mesh = Neo::MeshBase{"evolutive_neo_mesh"};
  addCells(mesh);
  std::vector<Neo::utils::Int64> node_uids{6,7,8,9,10,11};
  std::vector<Neo::utils::Real3> node_coords{{0,0,-1},{0,1.5,-1},{0,1.5,-1},
                                             {0,2.7,-1},{0,3.85,-1},{0,5,-1}};
  moveNodes(mesh, mesh.getFamily(Neo::ItemKind::IK_Node, node_family_name),node_uids, node_coords);
  mesh.applyAlgorithms();
}

//----------------------------------------------------------------------------/

TEST(EvolutiveMeshTest,RemoveCells)
{
  std::cout << "Remove cells test " << std::endl;
  auto mesh = Neo::MeshBase{"evolutive_neo_mesh"};
  addCells(mesh);
  // add a connectivity to cell
  std::vector<Neo::utils::Int64> node_to_cell{0,0,1,1,2,3,0,0,1,2,2,3};
  auto &cell_family = mesh.getFamily(Neo::ItemKind::IK_Cell, cell_family_name);
  auto &node_family = mesh.getFamily(Neo::ItemKind::IK_Node, node_family_name);
  addConnectivity(mesh,node_family,node_family.all(),cell_family,1, node_to_cell);
  mesh.applyAlgorithms();
  // Remove cell 0, 1 and 2
  std::vector<Neo::utils::Int64> removed_cells{0,1,2};
  removeItems(mesh,cell_family,removed_cells);
  updateConnectivity(mesh,node_family,cell_family);
  mesh.applyAlgorithms();
  auto node2cells_con_name = node_family_name + "to" + cell_family_name + "_connectivity";
  auto node2cells = node_family.getConcreteProperty<Neo::ArrayProperty<Neo::utils::Int32>>(node2cells_con_name);
  // compute a reference connectivity : replace removed cells by null lid
  std::fill(node_to_cell.begin(), node_to_cell.end(), Neo::utils::NULL_ITEM_LID);
  node_to_cell[5] = 3;
  node_to_cell[11] = 3;
  EXPECT_TRUE(std::equal(node2cells.view().begin(),node2cells.view().end(),node_to_cell.begin()));
}
