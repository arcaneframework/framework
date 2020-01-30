//
// Created by dechaiss on 1/22/20.
//

/*-------------------------
 * Neo library
 * Evolutive mesh test
 * sdc (C)-2020
 *
 *-------------------------
 */

#include "neo/Neo.h"
#include "gtest/gtest.h"

auto& addCellFamily(Neo::Mesh& mesh,std::string family_name){
  auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell,std::move(family_name));
  cell_family.addProperty<Neo::utils::Int64>(family_name+"_uids");
  return cell_family;
}

auto& addNodeFamily(Neo::Mesh& mesh,std::string family_name){
  auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node,std::move(family_name));
  node_family.addProperty<Neo::utils::Int64>(family_name+"_uids");
  node_family.addProperty<Neo::utils::Real3>(std::string("node_coords"));
  return node_family;
}

auto& addFaceFamily(Neo::Mesh& mesh,std::string family_name){
  auto& face_family = mesh.addFamily(Neo::ItemKind::IK_Face,std::move(family_name));
  face_family.addProperty<Neo::utils::Int64>(family_name+"_uids");
  return face_family;
}

// todo create another signature where uids are moved, or make clear that they potentially are...
void addItems(Neo::Mesh& mesh, Neo::Family& family, std::vector<Neo::utils::Int64>& uids, Neo::AddedItemRange& added_item_range)
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



TEST(EvolutiveMeshTest,AddCells)
{
  auto mesh = Neo::Mesh{"evolutive_neo_mesh"};
  auto& cell_family = addCellFamily(mesh,"CellFamily");
  auto& node_family = addNodeFamily(mesh,"NodeFamily");
  auto& face_family = addFaceFamily(mesh,"FaceFamily");
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

  mesh.beginUpdate();
  auto added_cells = Neo::AddedItemRange{};
  auto added_nodes = Neo::AddedItemRange{};
  auto added_faces = Neo::AddedItemRange{};
  // demander au maillage de la créer
  addItems(mesh, cell_family, cell_uids, added_cells);
  addItems(mesh, node_family, node_uids, added_nodes);
  addItems(mesh, face_family, face_uids, added_faces);
  auto valid_mesh_state = mesh.endUpdate();// retourner un objet qui dévérouille la range
  auto& new_cells = added_cells.get(valid_mesh_state);
  auto& new_nodes = added_nodes.get(valid_mesh_state);
  auto& new_faces = added_faces.get(valid_mesh_state);
  std::cout << "Added cells range after endUpdate: " << new_cells;
  std::cout << "Added nodes range after endUpdate: " << new_nodes;
  std::cout << "Added faces range after endUpdate: " << new_faces;
}

