//
// Created by dechaiss on 4/21/20.
//
/*-------------------------
 * Neo library
 * Mesh view test
 * sdc (C)-2020
 *
 *-------------------------
 */

#include <iostream>
#include <string>
#include "gtest/gtest.h"
#include "neo/Neo.h"

namespace utils{
  void fillMesh(Neo::Mesh& mesh){
    // creating mesh
    auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node,"NodeFamily");
    auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell,"CellFamily");

    // Adding node family and properties
    std::cout << "Find family " << node_family.m_name << std::endl;
    node_family.addProperty<Neo::utils::Real3>(std::string("node_coords"));
    node_family.addProperty<Neo::utils::Int64>("node_uids");
    node_family.addArrayProperty<Neo::utils::Int32>("node2cells");
    node_family.addProperty<Neo::utils::Int32>("internal_end_of_remove_tag"); // not a user-defined property // todo use byte ?

// Test adds
    auto& property = node_family.getProperty("node_uids");

// Adding cell family and properties
    std::cout << "Find family " << cell_family.m_name << std::endl;
    cell_family.addProperty<Neo::utils::Int64>("cell_uids");
    cell_family.addArrayProperty<Neo::utils::Int32>("cell2nodes");

// given data to create mesh. After mesh creation data is no longer available
    std::vector<Neo::utils::Int64> node_uids{0,1,2};
    std::vector<Neo::utils::Real3> node_coords{{0,0,0}, {0,1,0}, {0,0,1}};
    std::vector<Neo::utils::Int64> cell_uids{0,2,7,9};

// add algos:
    mesh.beginUpdate();

// create nodes
    auto added_nodes = Neo::ItemRange{};
    mesh.addAlgorithm(
        Neo::OutProperty{node_family,node_family.lidPropName()},
        [&node_uids,&added_nodes](Neo::ItemLidsProperty & node_lids_property){
          std::cout << "Algorithm: create nodes" << std::endl;
          added_nodes = node_lids_property.append(node_uids);
          node_lids_property.debugPrint();
          std::cout << "Inserted item range : " << added_nodes;
        });

// register node uids
    mesh.addAlgorithm(
        Neo::InProperty{node_family,node_family.lidPropName()},
        Neo::OutProperty{node_family,"node_uids"},
        [&node_uids,&added_nodes](Neo::ItemLidsProperty const& node_lids_property,
                                  Neo::PropertyT<Neo::utils::Int64>& node_uids_property){
          std::cout << "Algorithm: register node uids" << std::endl;
          if (node_uids_property.isInitializableFrom(added_nodes))  node_uids_property.init(added_nodes,std::move(node_uids)); // init can steal the input values
          else node_uids_property.append(added_nodes, node_uids);
          node_uids_property.debugPrint();
        });// need to add a property check for existing uid

// register node coords
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
//
// Add cells and connectivity

// create cells
    auto added_cells = Neo::ItemRange{};
    mesh.addAlgorithm(
        Neo::OutProperty{cell_family,cell_family.lidPropName()},
        [&cell_uids,&added_cells](Neo::ItemLidsProperty& cell_lids_property) {
          std::cout << "Algorithm: create cells" << std::endl;
          added_cells = cell_lids_property.append(cell_uids);
          cell_lids_property.debugPrint();
          std::cout << "Inserted item range : " << added_cells;
        });

// register cell uids
    mesh.addAlgorithm(
        Neo::InProperty{cell_family,cell_family.lidPropName()},
        Neo::OutProperty{cell_family,"cell_uids"},
        [&cell_uids,&added_cells](Neo::ItemLidsProperty const& cell_lids_property,
                                  Neo::PropertyT<Neo::utils::Int64>& cell_uids_property){
          std::cout << "Algorithm: register cell uids" << std::endl;
          if (cell_uids_property.isInitializableFrom(added_cells))  cell_uids_property.init(added_cells,std::move(cell_uids)); // init can steal the input values
          else cell_uids_property.append(added_cells, cell_uids);
          cell_uids_property.debugPrint();
        });

// register connectivity
// node to cell
    std::vector<Neo::utils::Int64> connected_cell_uids{0,0,2,2,7,9};
    std::vector<std::size_t> nb_cell_per_node{1,2,3};
    mesh.addAlgorithm(
        Neo::InProperty{node_family,node_family.lidPropName()},
        Neo::InProperty{cell_family,cell_family.lidPropName()},
        Neo::OutProperty{node_family,"node2cells"},
        [&connected_cell_uids, &nb_cell_per_node,& added_nodes]
            (Neo::ItemLidsProperty const& node_lids_property,
             Neo::ItemLidsProperty const& cell_lids_property,
             Neo::ArrayProperty<Neo::utils::Int32> & node2cells){
          std::cout << "Algorithm: register node-cell connectivity" << std::endl;
          auto connected_cell_lids = cell_lids_property[connected_cell_uids];
          if (node2cells.isInitializableFrom(added_nodes)) {
            node2cells.resize(std::move(nb_cell_per_node));
            node2cells.init(added_nodes,std::move(connected_cell_lids));
          }
          else {
            node2cells.append(added_nodes,connected_cell_lids, nb_cell_per_node);
          }
          node2cells.debugPrint();
        });

// cell to node
    std::vector<Neo::utils::Int64> connected_node_uids{0,1,2,1,2,0,2,1,0};// on ne connecte volontairement pas toutes les mailles pour vérifier initialisation ok sur la famille
    std::vector<std::size_t> nb_node_per_cell{3,0,3,3};
    mesh.addAlgorithm(Neo::InProperty{node_family,node_family.lidPropName()},
                      Neo::InProperty{cell_family,cell_family.lidPropName()},
                      Neo::OutProperty{cell_family,"cell2nodes"},
                      [&connected_node_uids, &nb_node_per_cell,& added_cells]
                          (
                              Neo::ItemLidsProperty const& node_lids_property,
                              Neo::ItemLidsProperty const& cell_lids_property,
                              Neo::ArrayProperty<Neo::utils::Int32> & cells2nodes){
                        std::cout << "Algorithm: register cell-node connectivity" << std::endl;
                        auto connected_node_lids = node_lids_property[connected_node_uids];
                        if (cells2nodes.isInitializableFrom(added_cells)) {
                          cells2nodes.resize(std::move(nb_node_per_cell));
                          cells2nodes.init(added_cells,std::move(connected_node_lids));
                        }
                        else cells2nodes.append(added_cells,connected_node_lids, nb_node_per_cell);
                        cells2nodes.debugPrint();
                      });
    // launch algos
    mesh.endUpdate();
  }
}

namespace tools{

struct ItemView {
  Neo::utils::Int32 localId() const {return -1;}
  Neo::utils::Int64 uniqueId() const {return -1;}
  int owner() const {return -1;}
};

struct ItemRangeView{ // todo revoir le nom
  ItemView operator*() const {return ItemView{};}
  bool operator==(const ItemRangeView& item_iterator) {return false;}
  bool operator!=(const ItemRangeView& item_iterator) {return !(*this == item_iterator);}
  ItemRangeView& operator++() {return *this;}
  ItemRangeView operator++(int) {auto retval = *this; ++(*this); return retval;}

};

struct GroupView{
  Neo::utils::ConstArrayView<Neo::utils::Int32> item_lids;
  ItemRangeView enumerator() const {return ItemRangeView{};}
  ItemRangeView begin() const {return ItemRangeView{};}
  ItemRangeView end() const {return ItemRangeView{};}
};

struct FamilyView {
  Neo::utils::ConstArrayView<Neo::utils::Int32> item_lids;
  Neo::utils::ConstArrayView<Neo::utils::Int64> item_uids;
  // Neo::utils::ConstArrayView<Neo::ItemGroup> item_groups; // todo Neo::Group
  int nbItem() {return item_lids.size();}
  GroupView allItems() {return GroupView{item_lids};}
};

struct MeshView{
  const std::string& name;
  FamilyView cell_family;
  FamilyView face_family;
  FamilyView edge_family;
  FamilyView node_family;
  std::map<std::string, FamilyView> dof_families;
  GroupView const& allCells() const {}
};


}

TEST(MeshViewTest,MeshIterationTest)
{
  auto mesh = Neo::Mesh{"MeshForView"};
  utils::fillMesh(mesh);
  tools::MeshView view{mesh.m_name};
  for (auto icell : view.allCells())
  {
    std::cout << icell.localId() << std::endl;
    std::cout << icell.uniqueId() << std::endl;
    std::cout << icell.owner() << std::endl;
  }
}


TEST(MeshViewTest,MeshInfoTest)
{
  auto mesh = Neo::Mesh{"MeshForView"};
  utils::fillMesh(mesh);
  tools::MeshView view{mesh.m_name};
  // Nombre de noeuds du maillage
  std::cout << view.node_family.nbItem() << std::endl;
  // Nombre d'arêtes du maillage
  std::cout << view.edge_family.nbItem() << std::endl;
  // Nombre de faces du maillage
  std::cout << view.face_family.nbItem() << std::endl;
  // Nombre de mailles du maillage
  std::cout << view.cell_family.nbItem() << std::endl;
  // Nombre d'éléments de la famille de dof "DofFamily"
  std::cout << view.dof_families["DofFamily"].nbItem() << std::endl;
}


