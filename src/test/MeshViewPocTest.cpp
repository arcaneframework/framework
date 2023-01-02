// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshViewPocTest.h                               (C) 2000-2020             */
/*                                                                           */
/* A POC for a mesh view API                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <iostream>
#include <string>
#include "gtest/gtest.h"
#include "neo/Neo.h"

namespace utils{
  void fillMesh(Neo::MeshBase & mesh){
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
    std::vector<int> nb_cell_per_node{1,2,3};
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
    std::vector<int> nb_node_per_cell{3,0,3,3};
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
    mesh.applyAlgorithms();
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
  bool operator==(const ItemRangeView& item_iterator) {return true;}
  bool operator!=(const ItemRangeView& item_iterator) {return !(*this == item_iterator);}
  ItemRangeView& operator++() {return *this;}
  ItemRangeView operator++(int) {auto retval = *this; ++(*this); return retval;}
};

template <typename T>
struct VariableViewT {
  T operator[]([[maybe_unused]] ItemView & item) {return T{};}
};

struct VariableView {
  using ViewType = std::variant<VariableViewT<Neo::utils::Int32>,VariableViewT<Neo::utils::Int64>,
      VariableViewT<Neo::utils::Real3>,VariableViewT<double>>;
  ViewType view;
};

struct FamilyView;

struct GroupView{
  FamilyView const& family;
  std::string name;
  Neo::ItemKind item_kind = Neo::ItemKind::IK_None;
  Neo::utils::ConstArrayView<Neo::utils::Int32> item_lids;
  Neo::utils::ConstArrayView<int> item_owners;
  ItemRangeView enumerator() const {return ItemRangeView{};}
  ItemRangeView begin() const {return ItemRangeView{};}
  ItemRangeView end() const {return ItemRangeView{};}
  int size() const { return item_lids.size();}
  ItemRangeView nodeGroup() const { return ItemRangeView{};}
  ItemRangeView edgeGroup() const {return ItemRangeView{};}
  ItemRangeView faceGroup() const {return ItemRangeView{};}
  ItemRangeView cellGroup() const {return ItemRangeView{};}
};

struct MeshView;

struct FamilyView {
  MeshView const& mesh;
  std::string name;
  Neo::ItemKind item_kind = Neo::ItemKind::IK_None;
  Neo::utils::Int32 max_local_id = Neo::utils::NULL_ITEM_LID;
  GroupView all_items = GroupView{*this};
  Neo::utils::ConstArrayView<Neo::utils::Int32> item_lids;
  Neo::utils::ConstArrayView<Neo::utils::Int64> item_uids;
  std::map<std::string, GroupView> item_groups;
  // Connectivities
  Neo::utils::ConstArray2View<Neo::utils::Int32> nodes;
  Neo::utils::ConstArray2View<Neo::utils::Int32> edges;
  Neo::utils::ConstArray2View<Neo::utils::Int32> faces;
  Neo::utils::ConstArray2View<Neo::utils::Int32> cells;
  std::map<std::string,Neo::utils::ConstArray2View<Neo::utils::Int32>> connectivities;
  std::map<std::string,VariableView> variables;
  int nbItem() const {return item_lids.size();}
};

struct MeshView{
  std::string const name;
  FamilyView cell_family = FamilyView{*this};
  FamilyView face_family = FamilyView{*this};
  FamilyView edge_family = FamilyView{*this};
  FamilyView node_family = FamilyView{*this};
  std::map<std::string, FamilyView> dof_families;
  VariableViewT<Neo::utils::Real3> node_coords;
  GroupView const& allCells() const {return cell_family.all_items;}
};

class MeshViewBuilder
{
public:
  virtual MeshView build() = 0;
};

struct NeoMeshViewBuilder : public MeshViewBuilder
{
  Neo::MeshBase const & m_neo_mesh;
  MeshView build() override {
    auto & cell_family = m_neo_mesh.getFamily(Neo::ItemKind::IK_Cell,"CellFamily");
    auto & face_family = m_neo_mesh.getFamily(Neo::ItemKind::IK_Face,"FaceFamily");
//    auto cell_lids = Neo::utils::ConstArrayView<Neo::utils::Int32>{cell_family._lidProp().values().};
  }
};

}

TEST(MeshViewTest,MeshIterationTest)
{
  auto mesh = Neo::MeshBase{"MeshForView"};
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
  auto mesh = Neo::MeshBase{"MeshForView"};
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
//  view.dof_families["DoFFamily"] = tools::FamilyView{view,"DoFFamily"};
//  std::cout << view.dof_families["DofFamily"].nbItem() << std::endl;
}

TEST(MeshViewTest,MeshFamilyInfoTest)
{
  auto mesh = Neo::MeshBase{"MeshForView"};
  tools::MeshView view {mesh.m_name};
  auto & cell_family = view.cell_family;
  std::cout << cell_family.name << " ";
  std::cout << " from mesh " << cell_family.mesh.name << " ";
  std::cout << Neo::utils::itemKindName(cell_family.item_kind) << " ";
  std::cout << cell_family.nbItem() << " ";
  std::cout << cell_family.max_local_id << std::endl;
  // iterate all items group
  for (auto icell : cell_family.all_items) {
    std::cout << " Cell in family" << std::endl;
  }
  // iterate all family groups
  for (auto [name, group] : cell_family.item_groups) {
    std::cout << " Cell family group " << name
              << " with size " << group.size()
              << std::endl;
  }
  // Print lids
  for (auto lid : cell_family.item_lids) {
    std::cout << " cell lid " << std::endl;
  }
  // Print uids
  for (auto uid : cell_family.item_uids) {
    std::cout << " cell uid " << std::endl;
  }
}

TEST(MeshViewTest,MeshGroupInfoTest)
{
  tools::MeshView mesh_view{"MeshView4GroupTest"};
  tools::FamilyView family_view{mesh_view,"FamilyView4GroupTest"};
  tools::GroupView group_view{family_view};
  std::cout << "group name " << group_view.name;
  std::cout << " with size " << group_view.size();
  if (group_view.size() == 0) std::cout << " empty group";
  std::cout << " item kind " << Neo::utils::itemKindName(group_view.item_kind);
  std::cout << " from family " << group_view.family.name << " from mesh " << group_view.family.mesh.name;
  // owners : dans la famille également ?
  auto current_rank = 0;
  for (auto item : group_view) {
    if (group_view.item_owners[item.localId()] == current_rank) std::cout << "Item is own " << std::endl;
  }
}

TEST(MeshViewTest,MeshConnectivityInfoTest)
{
  // Item Connectivity
  tools::MeshView mesh_view{"MeshView4ConnectivityTest"};
  for (auto icell : mesh_view.cell_family.all_items) {
      std::cout << "cell lid " << icell.localId() << " connected with nodes : " << std::endl;
    for (auto inode : mesh_view.cell_family.nodes[icell.localId()]) {
      std::cout << " node lid " << inode;
    }
  }
  // Group Connectivity
  auto face_nodes = mesh_view.face_family.all_items.nodeGroup();
}

TEST(MeshViewTest,MeshVariableInfoTest)
{
  tools::MeshView mesh_view{"MeshView4VariableTest"};
  auto& node_coords = mesh_view.node_coords;
  for (auto inode : mesh_view.node_family.all_items){
    std::cout << " node " << inode.localId() << std::endl;
    std::cout << " Coords  {"
              << node_coords[inode].x << ","
              << node_coords[inode].y << ","
              << node_coords[inode].z << ",}" << std::endl;
  }
  auto cell_variable_ite = mesh_view.cell_family.variables.find("my_cell_int_var");
  if (cell_variable_ite != mesh_view.cell_family.variables.end()) {
    auto& cell_variable = cell_variable_ite->second;
    auto concrete_cell_variable = std::get<tools::VariableViewT<int>>(cell_variable.view);
    for (auto icell : mesh_view.cell_family.all_items) {
      std::cout << "cell_variable[icell : " << icell.localId() << "] = ";
      std::cout << concrete_cell_variable[icell] << std::endl;
    }
  }
}


