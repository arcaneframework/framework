//
// Created by dechaiss on 2/6/20.
//

/*-------------------------
 * Neo library
 * Polyhedral mesh test
 * sdc (C)-2020
 *
 *-------------------------
 */

#include "neo/Neo.h"
#include "gtest/gtest.h"

namespace StaticMesh {

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

  // todo same interface with nb_connected_item_per_item as an array
  void addConnectivity(Neo::Mesh &mesh, Neo::Family &source_family,
                       Neo::ItemRange &source_items,
                       Neo::Family& target_family,
                       int nb_connected_item_per_item,
                       std::vector<Neo::utils::Int64>& connected_item_uids) {
    // add connectivity property if doesn't exist
    std::string connectivity_name =
        source_family.m_name + "to" + target_family.m_name + "_connectivity";
    source_family.addArrayProperty<Neo::utils::Int32>(connectivity_name);
    mesh.addAlgorithm(
        Neo::InProperty{source_family, source_family.lidPropName()},
        Neo::InProperty{target_family, target_family.lidPropName()},
        Neo::OutProperty{source_family, source_family.m_name + "to" +
                                            target_family.m_name +
                                            "_connectivity"},
        [&connected_item_uids, &nb_connected_item_per_item, &source_items,
         &source_family, &target_family](
            Neo::ItemLidsProperty const &source_family_lids_property,
            Neo::ItemLidsProperty const &target_family_lids_property,
            Neo::ArrayProperty<Neo::utils::Int32> &source2target) {
          std::cout << "Algorithm: register connectivity between "
                    << source_family.m_name << "  and  " << target_family.m_name
                    << std::endl;
          auto connected_item_lids =
              target_family_lids_property[connected_item_uids];
          std::vector<std::size_t> nb_connected_item_per_item_array(
              source_items.size(), nb_connected_item_per_item);
          if (source2target.isInitializableFrom(source_items)) {
            source2target.resize(std::move(nb_connected_item_per_item_array));
            source2target.init(source_items, std::move(connected_item_lids));
          } else {
            source2target.append(source_items, connected_item_lids,
                                 nb_connected_item_per_item_array);
          }
          source2target.debugPrint();
        });
  }
  // todo : define 2 signatures to indicate eventual memory stealing...?
  void setNodeCoords(Neo::Mesh& mesh, Neo::Family& node_family, Neo::AddedItemRange& added_node_range, std::vector<Neo::utils::Real3>& node_coords){
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
  void addConnectivity(Neo::Mesh &mesh, Neo::Family &source_family,
                       Neo::AddedItemRange &source_items,
                       Neo::Family& target_family,
                       int nb_connected_item_per_item,
                       std::vector<Neo::utils::Int64>& connected_item_uids)
  {
    addConnectivity(mesh, source_family, source_items.new_items, target_family,
                    nb_connected_item_per_item, connected_item_uids);
  }
}

namespace PolyhedralMeshTest {

  auto &addCellFamily(Neo::Mesh &mesh, std::string family_name) {
    auto &cell_family =
        mesh.addFamily(Neo::ItemKind::IK_Cell, std::move(family_name));
    cell_family.addProperty<Neo::utils::Int64>(family_name + "_uids");
    return cell_family;
  }

  auto &addNodeFamily(Neo::Mesh &mesh, std::string family_name) {
    auto &node_family =
        mesh.addFamily(Neo::ItemKind::IK_Node, std::move(family_name));
    node_family.addProperty<Neo::utils::Int64>(family_name + "_uids");
    return node_family;
  }

  auto &addFaceFamily(Neo::Mesh &mesh, std::string family_name) {
    auto &face_family =
        mesh.addFamily(Neo::ItemKind::IK_Face, std::move(family_name));
    face_family.addProperty<Neo::utils::Int64>(family_name + "_uids");
    return face_family;
  }

  static const std::string cell_family_name{"CellFamily"};
  static const std::string face_family_name{"FaceFamily"};
  static const std::string node_family_name{"NodeFamily"};

  void addCells(Neo::Mesh &mesh) {
    auto &cell_family = addCellFamily(mesh, cell_family_name);
    auto &node_family = addNodeFamily(mesh, node_family_name);
    auto &face_family = addFaceFamily(mesh, face_family_name);
    std::vector<Neo::utils::Int64> node_uids{0, 1, 2, 3, 4, 5};
    std::vector<Neo::utils::Int64> cell_uids{0};
    std::vector<Neo::utils::Int64> face_uids{0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<Neo::utils::Real3> node_coords{
        {-1, -1, 0}, {-1, 1, 0}, {1, 1, 0}, {1, -1, 0}, {0, 0, 1}, {0, 0, -1}};

    std::vector<Neo::utils::Int64> cell_nodes{0, 1, 2, 3, 4, 5};

    std::vector<Neo::utils::Int64> face_nodes{3, 0, 1, 4, 3, 0, 1, 5, 3, 1, 2,
                                              4, 3, 1, 2, 5, 3, 2, 3, 4, 3, 2,
                                              3, 5, 3, 3, 0, 4, 3, 3, 0, 5};

    mesh.beginUpdate();
    auto added_cells = Neo::AddedItemRange{};
    auto added_nodes = Neo::AddedItemRange{};
    auto added_faces = Neo::AddedItemRange{};
    StaticMesh::addItems(mesh, cell_family, cell_uids, added_cells);
    StaticMesh::addItems(mesh, node_family, node_uids, added_nodes);
    StaticMesh::addItems(mesh, face_family, face_uids, added_faces);
    StaticMesh::setNodeCoords(mesh, node_family, added_nodes, node_coords);
    auto nb_node_per_cell = 5;
    StaticMesh::addConnectivity(mesh, cell_family, added_cells, node_family,
                                nb_node_per_cell, cell_nodes);
    auto nb_node_per_face = 4;
    StaticMesh::addConnectivity(mesh, face_family, added_faces, node_family,
                                nb_node_per_face, face_nodes);
    auto valid_mesh_state =
        mesh.endUpdate(); // retourner un objet qui dévérouille la range
    auto &new_cells = added_cells.get(valid_mesh_state);
    auto &new_nodes = added_nodes.get(valid_mesh_state);
    auto &new_faces = added_faces.get(valid_mesh_state);
    std::cout << "Added cells range after endUpdate: " << new_cells;
    std::cout << "Added nodes range after endUpdate: " << new_nodes;
    std::cout << "Added faces range after endUpdate: " << new_faces;
  }
}

TEST(PolyhedralTest,CreateMesh1)
{
  auto mesh = Neo::Mesh{"PolyhedralMesh"};
  PolyhedralMeshTest::addCells(mesh);
}