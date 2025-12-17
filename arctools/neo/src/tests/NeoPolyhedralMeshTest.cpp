// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NeoPolyhedralMeshTest.cpp                       (C) 2000-2025             */
/*                                                                           */
/* First polyhedral mesh tests                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <vector>
#include <set>
#include <array>
#include <algorithm>

#include "neo/Neo.h"
#include "neo/Mesh.h"
#include "gtest/gtest.h"

#ifdef  HAS_XDMF
#include "XdmfDomain.hpp"
#include "XdmfInformation.hpp"
#include "XdmfUnstructuredGrid.hpp"
#include "XdmfSystemUtils.hpp"
#include "XdmfHDF5Writer.hpp"
#include "XdmfWriter.hpp"
#include "XdmfReader.hpp"
#include "neo/Mesh.h"
#endif

template <typename Container>
void _printContainer(Container&& container, std::string const& name){
  std::cout << name << " , size : " << container.size() << std::endl;
  for (auto element : container) {
    std::cout << element << " " ;
  }
  std::cout << std::endl;
}

namespace StaticMesh
{

static const std::string cell_family_name{ "CellFamily" };
static const std::string face_family_name{ "FaceFamily" };
static const std::string node_family_name{ "NodeFamily" };
static const std::string face_node_connectivity_name{ "face2nodes" };
static const std::string cell_face_connectivity_name{ "cell2faces" };
static const std::string face_cell_connectivity_name{ "face2cells" };
static const std::string cell_node_connectivity_name{ "cell2nodes" };

std::vector<Neo::utils::Int64> lid2uids(Neo::Family const& family,
                                        Neo::utils::ConstSpan<Neo::utils::Int32> item_lids) {
  auto& uid_property = family.getConcreteProperty<Neo::MeshScalarPropertyT<Neo::utils::Int64>>(family.m_name + "_uids");
  std::vector<Neo::utils::Int64> item_uids(item_lids.size());
  std::transform(item_lids.begin(), item_lids.end(), item_uids.begin(),
                 [&uid_property](auto lid) { return uid_property[lid]; });
  return item_uids;
}

namespace utilities
{
  using ItemNodesInCell = std::vector<std::vector<int>>; // [face_index_in_cell][node_index_in_face] = node_index_in_cell
  using NbNodeInCell = int;
  using CellTypes = std::vector<std::pair<NbNodeInCell, ItemNodesInCell>>;
  struct DefaultItemOrientation
  {
    static bool isOrdered(Neo::utils::ConstSpan<Neo::utils::Int64> item_nodes) {
      auto nb_nodes = item_nodes.size();
      if (nb_nodes == 2) {
        if (item_nodes[0] < item_nodes[1])
          return true;
        else
          return false;
      }
      auto min_position = std::distance(item_nodes.begin(), std::min_element(item_nodes.begin(), item_nodes.end()));
      if (item_nodes[(min_position - 1 + nb_nodes) % nb_nodes] > item_nodes[(min_position + 1) % nb_nodes])
        return true;
      else
        return false;
    }
  };
  template <typename ItemOrientation = DefaultItemOrientation>
  void getItemConnectivityFromCell(std::vector<Neo::utils::Int64> const& cell_nodes,
                                   std::vector<int> cell_type_indexes, CellTypes const& cell_types,
                                   int& nb_items,
                                   std::vector<Neo::utils::Int64>& item_nodes,
                                   std::vector<Neo::utils::Int32>& cell_items,
                                   std::vector<int>& cell_item_orientations,
                                   ItemOrientation const& item_orientation = DefaultItemOrientation{}) {
    nb_items = 0;
    auto cell_index = 0;
    auto item_index = 0;
    using ItemNodes = std::set<int>;
    using ItemUid = Neo::utils::Int64;
    using ItemInfo = std::pair<ItemNodes, ItemUid>;
    auto item_info_comp = [](ItemInfo const& item_info1,
                             ItemInfo const& item_info2) {
      return item_info1.first < item_info2.first;
    };
    std::set<ItemInfo, decltype(item_info_comp)> item_nodes_set(item_info_comp);
    item_nodes.reserve(4 * cell_nodes.size()); // rough approx
    cell_items.reserve(cell_nodes.size()); // rough approx
    cell_item_orientations.reserve(2 * cell_nodes.size()); // rough approx
    for (int cell_nodes_index = 0; cell_nodes_index < (int)cell_nodes.size();) {
      auto [nb_node_in_cell, item_nodes_all_items] = cell_types[cell_type_indexes[cell_index++]];
      auto current_cell_nodes = Neo::utils::ConstSpan<Neo::utils::Int64>{ &cell_nodes[cell_nodes_index], nb_node_in_cell };
      for (auto current_item_node_indexes_in_cell : item_nodes_all_items) {
        std::vector<Neo::utils::Int64> current_item_nodes;
        current_item_nodes.reserve(current_item_node_indexes_in_cell.size());
        std::transform(current_item_node_indexes_in_cell.begin(),
                       current_item_node_indexes_in_cell.end(),
                       std::back_inserter(current_item_nodes),
                       [&current_cell_nodes](auto& node_index) { return current_cell_nodes[node_index]; });
        auto [item_info, is_new_item] = item_nodes_set.emplace(
        ItemNodes{ current_item_nodes.begin(),
                   current_item_nodes.end() },
        item_index);
        if (!is_new_item)
          std::cout << "Item not inserted " << item_index << std::endl;
        if (is_new_item) {
          item_nodes.insert(item_nodes.end(), current_item_nodes.begin(), current_item_nodes.end());
        }
        auto orientation = item_orientation.isOrdered({ current_item_nodes.data(), current_item_nodes.size()  }) ? 1 : -1;
        cell_item_orientations.push_back(orientation);
        cell_items.push_back(item_info->second);
        if (is_new_item) {
          nb_items++;
          ++item_index;
        }
      }
      cell_nodes_index += nb_node_in_cell;
    }
  }

  void reverseConnectivity(std::vector<Neo::utils::Int64> const& original_source_item_uids,
                           std::vector<Neo::utils::Int64> const& original_connectivity,
                           std::vector<int> const& nb_connected_items_per_item_original,
                           std::vector<Neo::utils::Int64>& new_source_item_uids,
                           std::vector<Neo::utils::Int64>& reversed_connectivity,
                           std::vector<int>& nb_connected_items_per_item_reversed,
                           std::vector<int> const& original_source_item_connected_item_orientations,
                           std::vector<int>& new_source_item_orientation_in_connected_items) {
    assert(("Invalid argument size, utilities::reverseConnectivity", original_source_item_uids.size() == nb_connected_items_per_item_original.size()));
    assert(("Invalid argument size, utilities::reverseConnectivity",
            (int)original_connectivity.size() == std::accumulate(nb_connected_items_per_item_original.begin(), nb_connected_items_per_item_original.end(), 0)));
    assert(("Invalid argument size, utilities::reverseConnectivity", (original_source_item_connected_item_orientations.size() == 0 || original_source_item_connected_item_orientations.size() == original_connectivity.size())));
    bool reverse_orientation = original_source_item_connected_item_orientations.size() != 0;
    auto source_item_index = 0;
    std::map<Neo::utils::Int64, std::vector<Neo::utils::Int64>> reversed_connectivity_map;
    std::map<Neo::utils::Int64, std::vector<Neo::utils::Int64>> reversed_orientation_map;
    for (int original_connectivity_index = 0; original_connectivity_index < (int)original_connectivity.size();) {
      auto current_item_nb_connected_items = nb_connected_items_per_item_original[source_item_index];
      auto current_item_connected_items = Neo::utils::ConstSpan<Neo::utils::Int64>{ &original_connectivity[original_connectivity_index], current_item_nb_connected_items };
      auto original_orientation_index = original_connectivity_index;
      for (auto connected_item : current_item_connected_items) {
        reversed_connectivity_map[connected_item].push_back(original_source_item_uids[source_item_index]);
        if (reverse_orientation) {
          reversed_orientation_map[connected_item].push_back(
          original_source_item_connected_item_orientations[original_orientation_index++]);
        }
      }
      source_item_index++;
      original_connectivity_index += current_item_nb_connected_items;
    }
    new_source_item_uids.resize(reversed_connectivity_map.size());
    nb_connected_items_per_item_reversed.resize(reversed_connectivity_map.size());
    auto new_source_item_uids_index = 0;
    reversed_connectivity.clear();
    reversed_connectivity.reserve(4 * reversed_connectivity_map.size()); // choose an average of 4 connected elements per item
    if (reverse_orientation) {
      new_source_item_orientation_in_connected_items.clear();
      new_source_item_orientation_in_connected_items.reserve(4 * reversed_connectivity_map.size());
    }
    for (auto [new_source_item_uid, new_source_item_connected_items] : reversed_connectivity_map) {
      new_source_item_uids[new_source_item_uids_index] = new_source_item_uid;
      reversed_connectivity.insert(reversed_connectivity.end(),
                                   new_source_item_connected_items.begin(),
                                   new_source_item_connected_items.end());
      nb_connected_items_per_item_reversed[new_source_item_uids_index++] =
      new_source_item_connected_items.size();
    }
    if (reverse_orientation) {
      for (auto [new_source_item_uid, new_source_item_orientation] : reversed_orientation_map) {
        new_source_item_orientation_in_connected_items.insert(new_source_item_orientation_in_connected_items.end(),
                                                              new_source_item_orientation.begin(),
                                                              new_source_item_orientation.end());
      }
    }
  }

  void reverseConnectivity(std::vector<Neo::utils::Int64> const& original_source_item_uids,
                           std::vector<Neo::utils::Int64> const& original_connectivity,
                           std::vector<int> const& nb_connected_items_per_item_original,
                           std::vector<Neo::utils::Int64>& new_source_item_uids,
                           std::vector<Neo::utils::Int64>& reversed_connectivity,
                           std::vector<int>& nb_connected_items_per_item_reversed) {
    std::vector<int> original_source_item_connected_item_orientations{};
    std::vector<int> new_source_item_orientation_in_connected_items{};
    reverseConnectivity(original_source_item_uids, original_connectivity,
                        nb_connected_items_per_item_original,
                        new_source_item_uids, reversed_connectivity,
                        nb_connected_items_per_item_reversed,
                        original_source_item_connected_item_orientations,
                        new_source_item_orientation_in_connected_items);
  }
} // namespace utilities
} // namespace StaticMesh

namespace PolyhedralMeshTest
{

auto& addCellFamily(Neo::Mesh& mesh, std::string const& family_name) {
  auto& cell_family =
  mesh.addFamily(Neo::ItemKind::IK_Cell, family_name);
  cell_family.addMeshScalarProperty<Neo::utils::Int64>(family_name + "_uids");
  return cell_family;
}

auto& addNodeFamily(Neo::Mesh& mesh, std::string const& family_name) {
  auto& node_family =
  mesh.addFamily(Neo::ItemKind::IK_Node, family_name);
  node_family.addMeshScalarProperty<Neo::utils::Int64>(family_name + "_uids");
  return node_family;
}

auto& addFaceFamily(Neo::Mesh& mesh, std::string const& family_name) {
  auto& face_family =
  mesh.addFamily(Neo::ItemKind::IK_Face, family_name);
  face_family.addMeshScalarProperty<Neo::utils::Int64>(family_name + "_uids");
  return face_family;
}

void _createMesh(Neo::Mesh& mesh,
                 std::vector<Neo::utils::Int64> const& node_uids,
                 std::vector<Neo::utils::Int64> const& cell_uids,
                 std::vector<Neo::utils::Int64> const& face_uids,
                 std::vector<Neo::utils::Real3>& node_coords, // not const since they can be moved
                 std::vector<Neo::utils::Int64>& cell_nodes,
                 std::vector<Neo::utils::Int64>& cell_faces,
                 std::vector<Neo::utils::Int64>& face_nodes,
                 std::vector<Neo::utils::Int64>& face_cells,
                 std::vector<int>& face_orientation_in_cells,
                 std::vector<int> nb_node_per_cells,
                 std::vector<int> nb_face_per_cells,
                 std::vector<int> nb_node_per_faces,
                 std::vector<int> nb_cell_per_faces) {
  auto& cell_family = addCellFamily(mesh, StaticMesh::cell_family_name);
  auto& node_family = addNodeFamily(mesh, StaticMesh::node_family_name);
  auto& face_family = addFaceFamily(mesh, StaticMesh::face_family_name);
  auto added_cells = Neo::FutureItemRange{};
  auto added_nodes = Neo::FutureItemRange{};
  auto added_faces = Neo::FutureItemRange{};
  mesh.scheduleAddItems(cell_family, cell_uids, added_cells);
  mesh.scheduleAddItems(node_family, node_uids, added_nodes);
  mesh.scheduleAddItems(face_family, face_uids, added_faces);
  mesh.scheduleSetItemCoords(node_family, added_nodes, node_coords);
  mesh.scheduleAddConnectivity(cell_family, added_cells, node_family, std::move(nb_node_per_cells), cell_nodes, StaticMesh::cell_node_connectivity_name);
  mesh.scheduleAddConnectivity(face_family, added_faces, node_family, std::move(nb_node_per_faces), face_nodes, StaticMesh::face_node_connectivity_name);
  mesh.scheduleAddConnectivity(cell_family, added_cells, face_family, std::move(nb_face_per_cells), cell_faces, StaticMesh::cell_face_connectivity_name);
  auto do_check_orientation = true;
  mesh.scheduleAddOrientedConnectivity(face_family, added_faces, cell_family, std::move(nb_cell_per_faces), face_cells, StaticMesh::face_cell_connectivity_name,
                                       face_orientation_in_cells, Neo::Mesh::ConnectivityOperation::Add, do_check_orientation);
  auto valid_mesh_state = mesh.applyScheduledOperations();
  auto new_cells = added_cells.get(valid_mesh_state);
  auto new_nodes = added_nodes.get(valid_mesh_state);
  auto new_faces = added_faces.get(valid_mesh_state);
  std::cout << "Added cells range after applyAlgorithms: " << new_cells;
  std::cout << "Added nodes range after applyAlgorithms: " << new_nodes;
  std::cout << "Added faces range after applyAlgorithms: " << new_faces;
}

void createMesh(Neo::Mesh& mesh) {
  std::vector<Neo::utils::Int64> node_uids{ 0, 1, 2, 3, 4, 5 };
  std::vector<Neo::utils::Int64> cell_uids{ 0 };
  std::vector<Neo::utils::Int64> face_uids{ 0, 1, 2, 3, 4, 5, 6, 7 };

  std::vector<Neo::utils::Real3> node_coords{
    { -1, -1, 0 }, { -1, 1, 0 }, { 1, 1, 0 }, { 1, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }
  };

  std::vector<Neo::utils::Int64> cell_nodes{ 0, 1, 2, 3, 4, 5 };
  std::vector<Neo::utils::Int64> cell_faces{ 0, 1, 2, 3, 4, 5, 6, 7 };

  std::vector<Neo::utils::Int64> face_nodes{ 0, 1, 4, 0, 1, 5, 1, 2, 4, 1, 2, 5,
                                             2, 3, 4, 2, 3, 5, 3, 0, 4, 3, 0, 5 };
  std::vector<Neo::utils::Int64> face_cells(face_uids.size(), 0);
  auto nb_node_per_cell = 6;
  auto nb_node_per_face = 3;
  auto nb_face_per_cell = 8;
  auto nb_cell_per_face = 1;
  std::vector<int> face_orientation_in_cells{ 1, 1, 1, 1, 1, 1, -1, -1 };
  _createMesh(mesh, node_uids, cell_uids, face_uids, node_coords, cell_nodes,
              cell_faces, face_nodes, face_cells, face_orientation_in_cells,
              std::vector<int>(cell_uids.size(), nb_node_per_cell),
              std::vector<int>(cell_uids.size(), nb_face_per_cell),
              std::vector<int>(face_uids.size(), nb_node_per_face),
              std::vector<int>(face_uids.size(), nb_cell_per_face));
  // Validation
  auto& cell_family = mesh.findFamily(Neo::ItemKind::IK_Cell, StaticMesh::cell_family_name);
  auto& node_family = mesh.findFamily(Neo::ItemKind::IK_Node, StaticMesh::node_family_name);
  auto& face_family = mesh.findFamily(Neo::ItemKind::IK_Face, StaticMesh::face_family_name);
  EXPECT_EQ(cell_uids.size(), cell_family.nbElements());
  EXPECT_EQ(node_uids.size(), node_family.nbElements());
  EXPECT_EQ(face_uids.size(), face_family.nbElements());
  // Check cell to nodes connectivity
  std::vector<Neo::utils::Int64> reconstructed_cell_nodes;
  auto cell_to_nodes = mesh.getConnectivity(cell_family, node_family, StaticMesh::cell_node_connectivity_name);
  for (auto cell : cell_family.all()) {
    auto current_cell_nodes = cell_to_nodes[cell];
    reconstructed_cell_nodes.insert(reconstructed_cell_nodes.end(), current_cell_nodes.begin(), current_cell_nodes.end());
  }
  EXPECT_TRUE(std::equal(cell_nodes.begin(), cell_nodes.end(), reconstructed_cell_nodes.begin()));
  _printContainer(reconstructed_cell_nodes, "Recons cell nodes ");
  // Check face to cells connectivity
  std::vector<Neo::utils::Int64> reconstructed_face_cells;
  auto face_to_cells = mesh.getConnectivity(face_family, cell_family, StaticMesh::face_cell_connectivity_name);
  for (auto face : face_family.all()) {
    auto current_face_cells = face_to_cells[face];
    reconstructed_face_cells.insert(reconstructed_face_cells.end(), current_face_cells.begin(), current_face_cells.end());
  }
  EXPECT_TRUE(std::equal(face_cells.begin(), face_cells.end(), reconstructed_face_cells.begin()));
  _printContainer(reconstructed_face_cells, "Recons face cells ");
}
}

#ifdef HAS_XDMF
namespace XdmfTest {
  auto _changeNodeOrder(std::vector<Neo::utils::Int64> const& face_nodes,
                      Neo::utils::Int32 cell_lid,
                      Neo::utils::ConstSpan<Neo::utils::Int32> face_cells,
                      Neo::utils::ConstSpan<int> face_orientation) {
    auto count = 0;
    for (auto connected_cell_lid : face_cells){
      if (connected_cell_lid == cell_lid){
        break;
      }
      count++;
    }
    auto do_change_node_order = false;
    if (count < face_orientation.size()) {
      do_change_node_order = (face_orientation[count] == -1);
    }
    if (do_change_node_order) {
      std::vector<Neo::utils::Int64> new_node_order(face_nodes);
      std::sort(new_node_order.begin(),new_node_order.end());
      std::sort(new_node_order.begin() + 1, new_node_order.end(),
                std::greater<Neo::utils::Int64>());
      // implicit in test examples : first min_id then max and strictly decreasing order
      // DEBUG
      _printContainer(face_nodes," DEBUG FACE NODES ");
      // DEBUG
      return std::make_pair(do_change_node_order, new_node_order);
    }
    else {
      // check if face is stored positive, otherwise reorder
      if (StaticMesh::utilities::DefaultItemOrientation::isOrdered({ face_nodes.data(), face_nodes.size() }))
        return std::make_pair(do_change_node_order, std::vector<Neo::utils::Int64>{});
      else {
        // face is +1 oriented in the current cell, but was stored negatively oriented
        // return ascending sorted node order (is positive and fit with initial mesh)
        do_change_node_order = true;
        std::vector<Neo::utils::Int64> new_node_order(face_nodes);
        std::sort(new_node_order.begin(), new_node_order.end());
        return std::make_pair(do_change_node_order, new_node_order);
      }
    }
  }

  void exportMesh(Neo::Mesh const& mesh, std::string const& file_name) {
    auto domain = XdmfDomain::New();
    auto domain_info = XdmfInformation::New("Domain", " For polyhedral data from Neo");
    domain->insert(domain_info);
    // Needed ?
    auto xdmf_grid = XdmfUnstructuredGrid::New();
    auto xdmf_geom = XdmfGeometry::New();
    xdmf_geom->setType(XdmfGeometryType::XYZ());
    auto node_coords = mesh.getItemCoordProperty(mesh.findFamily(Neo::ItemKind::IK_Node, StaticMesh::node_family_name));
    xdmf_geom->insert(0, (double*)&(*(node_coords.begin())), node_coords.size() * 3, 1, 1);
    xdmf_grid->setGeometry(xdmf_geom);
    auto xdmf_topo = XdmfTopology::New();
    xdmf_topo->setType(XdmfTopologyType::Polyhedron());
    std::vector<int> cell_data;
    auto& cell_family = mesh.findFamily(Neo::ItemKind::IK_Cell, StaticMesh::cell_family_name);
    auto& face_family = mesh.findFamily(Neo::ItemKind::IK_Face, StaticMesh::face_family_name);
    cell_data.reserve(cell_family.nbElements() * 4); // 4 faces by cell approx
    auto cell_to_faces = mesh.faces(cell_family)[0];
    auto face_to_nodes = mesh.nodes(face_family)[0];
    auto face_cells = mesh.getConnectivity(face_family, cell_family, StaticMesh::face_cell_connectivity_name);
    auto const& face_orientation_in_cells = face_cells.connectivity_orientation;
    for (auto cell : cell_family.all()) {
      auto cell_faces = cell_to_faces[cell];
      cell_data.push_back(cell_faces.size());
      for (auto face : cell_faces) {
        auto face_nodes = face_to_nodes[face];
        auto face_nodes_uids = StaticMesh::lid2uids(face_family, face_nodes);
        cell_data.push_back(face_nodes.size());
        auto [do_change_node_order, face_nodes_new_order] = _changeNodeOrder(face_nodes_uids, cell, face_cells[face], face_orientation_in_cells[face]);
        if (!do_change_node_order)
        cell_data.insert(cell_data.end(), face_nodes.begin(), face_nodes.end());
        else
        cell_data.insert(cell_data.end(), face_nodes_new_order.begin(), face_nodes_new_order.end());
      }
    }
    xdmf_topo->insert(0, cell_data.data(), cell_data.size(), 1, 1);
    xdmf_grid->setTopology(xdmf_topo);
    domain->insert(xdmf_grid);
    //  auto heavydata_writer = XdmfHDF5Writer::New(file_name);
    //  auto writer = XdmfWriter::New(file_name, heavydata_writer);
    auto writer = XdmfWriter::New(file_name);
    //  domain->accept(heavydata_writer);
    writer->setLightDataLimit(1000);
    domain->accept(writer);
  }
  } // namespace XdmfTest
#endif //HAS_XDMF

  TEST(PolyhedralTest, CreateMesh1) {
    auto mesh = Neo::Mesh{ "PolyhedralMesh" };
    PolyhedralMeshTest::createMesh(mesh);
  }

  TEST(PolyhedralTest, ConnectivityUtilitiesTest) {
    // get face cells by reversing connectivity
    std::vector<Neo::utils::Int64> cell_uids{ 0, 1, 2, 3 };
    std::vector<Neo::utils::Int64> cell_faces{ 0, 1, 2, 2, 3, 4, 4, 6, 5, 6, 7, 8, 9 };
    std::vector<int> nb_face_per_cells{ 3, 3, 3, 4 };
    std::vector<int> cell_face_orientations{ 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1 };
    std::vector<Neo::utils::Int64> face_uids;
    std::vector<Neo::utils::Int64> face_cells;
    std::vector<int> nb_cell_per_faces;
    std::vector<int> face_orientation_in_cells;
    StaticMesh::utilities::reverseConnectivity(cell_uids, cell_faces, nb_face_per_cells,
                                               face_uids, face_cells, nb_cell_per_faces,
                                               cell_face_orientations, face_orientation_in_cells);
    _printContainer(face_uids, "Face uids ");
    _printContainer(face_cells, "Face cells ");
    _printContainer(nb_cell_per_faces, "Cell per faces ");
    _printContainer(face_orientation_in_cells, "Face orientation in cells ");
    std::vector<Neo::utils::Int64> face_uids_ref{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<Neo::utils::Int64> face_cells_ref{ 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3 };
    std::vector<int> nb_cell_per_faces_ref{ 1, 1, 2, 1, 2, 1, 2, 1, 1, 1 };
    std::vector<int> face_orientation_in_cells_ref{ 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1 };
    EXPECT_TRUE(std::equal(face_uids.begin(), face_uids.end(),
                           face_uids_ref.begin(), face_uids_ref.end()));
    EXPECT_TRUE(std::equal(face_cells.begin(), face_cells.end(),
                           face_cells_ref.begin(), face_cells_ref.end()));
    EXPECT_TRUE(std::equal(nb_cell_per_faces.begin(), nb_cell_per_faces.end(),
                           nb_cell_per_faces_ref.begin(), nb_cell_per_faces_ref.end()));
    EXPECT_TRUE(std::equal(face_orientation_in_cells.begin(), face_orientation_in_cells.end(),
                           face_orientation_in_cells_ref.begin(), face_orientation_in_cells_ref.end()));
  }

  TEST(PolyhedralTest, TypedUtilitiesOrientationTest) {
    // Test item orientation
    StaticMesh::utilities::DefaultItemOrientation item_orientation{};
    std::vector<Neo::utils::Int64> face_nodes_mock{ 1, 2, 3 };
    EXPECT_TRUE(item_orientation.isOrdered(
    { face_nodes_mock.data(), (int)face_nodes_mock.size() }));
    face_nodes_mock = { 7, 6, 5, 9, 10, 11 };
    EXPECT_FALSE(item_orientation.isOrdered(
    {face_nodes_mock.data(), face_nodes_mock.size() }));
    std::vector<Neo::utils::Int64> edge_node_mock{ 0, 100 };
    EXPECT_TRUE(item_orientation.isOrdered(
    { edge_node_mock.data(), edge_node_mock.size()  }));
    edge_node_mock = { 6, 1 };
    EXPECT_FALSE(item_orientation.isOrdered(
    { edge_node_mock.data(), edge_node_mock.size()  }));
  }

  TEST(PolyhedralTest, ItemOrientationCheckTest) {
    Neo::Mesh mesh{ "test_orientation_check_mesh" };
    // mock values, does not represent a real mesh
    std::vector<Neo::utils::Int64> cell_uids{ 0, 1, 2 };
    std::vector<Neo::utils::Int64> face_uids{ 0, 1, 2, 3, 4 };
    std::vector<Neo::utils::Int64> face_cells{ 0, 1, 1, 2, 2, 0, 0, 1 };
    std::vector<int> nb_cell_per_faces{ 2, 2, 2, 1, 1 };
    std::vector<int> face_orientation_in_cells{ 1, -1, -1, 1, -1, 1, -1, -1 };
    auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
    auto& face_family = mesh.addFamily(Neo::ItemKind::IK_Face, "FaceFamily");
    auto added_cells = Neo::FutureItemRange{};
    auto added_faces = Neo::FutureItemRange{};
    mesh.scheduleAddItems(cell_family, cell_uids, added_cells);
    mesh.scheduleAddItems(face_family, face_uids, added_faces);
    auto do_check_orientation = true;
    std::string connectivity_name = face_family.name() + "to" + cell_family.name() + "_connectivity";
    mesh.scheduleAddOrientedConnectivity(face_family, added_faces, cell_family,
                                         std::move(nb_cell_per_faces), face_cells, connectivity_name,
                                         face_orientation_in_cells, Neo::Mesh::ConnectivityOperation::Add,
                                         do_check_orientation);
    mesh.applyScheduledOperations();
    Neo::MeshScalarPropertyT<int> orientation_check_result = face_family.getConcreteProperty<Neo::MeshScalarPropertyT<int>>(
    "FaceFamilytoCellFamily_connectivity_orientation_check");
    _printContainer(orientation_check_result, "orientation check result");
    std::vector<int> orientation_check_result_ref{ 1, 1, 1, 1, 1 };
    EXPECT_TRUE(std::equal(orientation_check_result.begin(),
                           orientation_check_result.end(),
                           orientation_check_result_ref.begin()));
  }

  TEST(PolyhedralTest, ItemOrientationCheckTestWrongOrientation) {
    Neo::Mesh mesh{ "test_orientation_check_mesh" };
    // mock values, does not represent a real mesh
    std::vector<Neo::utils::Int64> cell_uids{ 0, 1, 2 };
    std::vector<Neo::utils::Int64> face_uids{ 0, 1, 2, 3, 4 };
    std::vector<Neo::utils::Int64> face_cells{ 0, 1, 1, 2, 2, 0, 0, 1 };
    std::vector<int> nb_cell_per_faces{ 2, 2, 2, 1, 1 };
    std::vector<int> face_orientation_in_cells{ 1, 1, -1, 1, -1, -1, -1, -1 };
    auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell, "CellFamily");
    auto& face_family = mesh.addFamily(Neo::ItemKind::IK_Face, "FaceFamily");
    auto added_cells = Neo::FutureItemRange{};
    auto added_faces = Neo::FutureItemRange{};
    mesh.scheduleAddItems(cell_family, cell_uids, added_cells);
    mesh.scheduleAddItems(face_family, face_uids, added_faces);
    auto do_check_orientation = true;
    std::string connectivity_name = face_family.name() + "to" + cell_family.name() + "_connectivity";
    mesh.scheduleAddOrientedConnectivity(face_family, added_faces, cell_family,
                                         std::move(nb_cell_per_faces), face_cells, connectivity_name, face_orientation_in_cells,
                                         Neo::Mesh::ConnectivityOperation::Add, do_check_orientation);
    EXPECT_THROW(mesh.applyScheduledOperations(), std::runtime_error);
    Neo::MeshScalarPropertyT<int> orientation_check_result = face_family.getConcreteProperty<Neo::MeshScalarPropertyT<int>>(
    "FaceFamilytoCellFamily_connectivity_orientation_check");
    _printContainer(orientation_check_result, "orientation check result");
    std::vector<int> orientation_check_result_ref{ 0, 1, 0, 1, 1 };
    EXPECT_TRUE(std::equal(orientation_check_result.begin(),
                           orientation_check_result.end(),
                           orientation_check_result_ref.begin()));
  }

  TEST(PolyhedralTest, TypedUtilitiesConnectivityTest) {
    // Test item connectivity
    std::vector<Neo::utils::Int64> cell_nodes{ 1, 8, 10, 15, 25, 27, 29, 30, // hexa
                                               8, 9, 11, 10, 27, 28, 31, 29, // hexa
                                               28, 9, 11, 31, 32 }; // prism
    using CellTypeIndexes = std::vector<int>;
    // Get Face Connectivity info
    std::vector<Neo::utils::Int64> face_nodes;
    std::vector<Neo::utils::Int32> cell_face_indexes;
    std::vector<int> cell_face_orientations;
    int nb_faces = 0;
    StaticMesh::utilities::getItemConnectivityFromCell(
    cell_nodes, CellTypeIndexes{ 0, 0, 1 },
    { { 8,
        { { 0, 3, 2, 1 },
          { 1, 2, 6, 5 },
          { 4, 5, 6, 7 },
          { 2, 3, 7, 6 },
          { 0, 3, 7, 4 },
          { 0, 1, 5, 4 } } },
      { 5, { { 0, 3, 2, 1 }, { 1, 2, 4 }, { 2, 3, 4 }, { 3, 0, 4 }, { 0, 1, 4 } } } },
    nb_faces, face_nodes, cell_face_indexes, cell_face_orientations);
    std::cout << "Nb faces found from cell info " << nb_faces << std::endl;
    _printContainer(face_nodes, "Face nodes from cell info");
    _printContainer(cell_face_indexes, "Cell faces (indexes) from cell info");
    _printContainer(cell_face_orientations, "Cell faces orientation ");
    // Get Edge Connectivity info
    std::vector<Neo::utils::Int64> edge_nodes;
    std::vector<Neo::utils::Int32> cell_edge_indexes;
    std::vector<int> cell_edge_orientations;
    int nb_edges = 0;
    StaticMesh::utilities::getItemConnectivityFromCell(
    cell_nodes, CellTypeIndexes{ 0, 0, 1 },
    { { 8, { { 0, 3 }, { 3, 2 }, { 2, 1 }, { 1, 0 }, { 2, 6 }, { 6, 5 }, { 5, 1 }, { 4, 5 }, { 6, 7 }, { 7, 4 }, { 3, 7 }, { 4, 0 } } },
      { 5, { { 0, 3 }, { 3, 2 }, { 2, 1 }, { 1, 0 }, { 2, 4 }, { 4, 1 }, { 3, 4 }, { 0, 4 } } } },
    nb_edges, edge_nodes, cell_edge_indexes, cell_edge_orientations);
    std::cout << "Nb edges found from cell info " << nb_edges << std::endl;
    _printContainer(edge_nodes, "Edge nodes from cell info");
    _printContainer(cell_edge_indexes, "Cell edges (indexes) from cell info");
    _printContainer(cell_edge_orientations, "Cell edges orientations ");
    // Validation
    EXPECT_EQ(15, nb_faces);
    std::vector<int> cell_face_indexes_ref{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 10, 7, 11, 12, 13, 14 };
    EXPECT_TRUE(std::equal(
    cell_face_indexes.begin(), cell_face_indexes.end(),
    cell_face_indexes_ref.begin(), cell_face_indexes_ref.end()));
    std::vector<Neo::utils::Int64> face_nodes_ref{ 1, 15, 10, 8,
                                                   8, 10, 29, 27,
                                                   25, 27, 29, 30,
                                                   10, 15, 30, 29,
                                                   1, 15, 30, 25,
                                                   1, 8, 27, 25,
                                                   8, 10, 11, 9,
                                                   9, 11, 31, 28,
                                                   27, 28, 31, 29,
                                                   11, 10, 29, 31,
                                                   8, 9, 28, 27,
                                                   9, 11, 32,
                                                   11, 31, 32,
                                                   31, 28, 32,
                                                   28, 9, 32 };
    EXPECT_TRUE(std::equal(
    face_nodes.begin(), face_nodes.end(),
    face_nodes_ref.begin(), face_nodes_ref.end()));
    EXPECT_EQ(24, nb_edges);
    std::vector<int> cell_edge_indexes_ref{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 12, 13, 14, 15, 16, 17, 18, 19, 5, 4, 6, 16, 15, 13, 17, 20, 21, 22, 23 };
    std::vector<Neo::utils::Int64> edge_nodes_ref{ 1, 15, 15, 10, 10, 8, 8, 1,
                                                   10, 29, 29, 27, 27, 8, 25, 27, 29, 30,
                                                   30, 25, 15, 30, 25, 1, 10, 11, 11, 9,
                                                   9, 8, 11, 31, 31, 28, 28, 9, 27, 28, 31, 29,
                                                   11, 32, 32, 9, 31, 32, 28, 32 };
    EXPECT_TRUE(std::equal(
    edge_nodes.begin(), edge_nodes.end(),
    edge_nodes_ref.begin(), edge_nodes_ref.end()));
    EXPECT_TRUE(std::equal(
    cell_edge_indexes.begin(), cell_edge_indexes.end(),
    cell_edge_indexes_ref.begin(), cell_edge_indexes_ref.end()));
    std::vector<int> cell_edge_orientations_ref{ 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1,
                                                 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
                                                 1, -1, -1, 1, 1, -1, 1, 1 };
    EXPECT_TRUE(std::equal(cell_edge_orientations.begin(),
                           cell_edge_orientations.end(),
                           cell_edge_orientations_ref.begin()));
  }

  TEST(PolyhedralTest, OrientedConnectivityTest) {
    auto mesh = Neo::Mesh{ "PolyhedralMeshWithOrientation" };
    PolyhedralMeshTest::createMesh(mesh);
    // create simple 2D mesh with orientation
    auto& face_family = mesh.findFamily(Neo::ItemKind::IK_Face,
                                        StaticMesh::face_family_name);
    auto& cell_family = mesh.findFamily(Neo::ItemKind::IK_Cell,
                                        StaticMesh::cell_family_name);
    auto oriented_connectivity = mesh.getConnectivity(face_family, cell_family, StaticMesh::face_cell_connectivity_name);
    std::vector<int> orientation_ref{ 1, 1, 1, 1, 1, 1, -1, -1 };
    auto const& orientation = oriented_connectivity.connectivity_orientation;
    auto const& connectivity = oriented_connectivity.connectivity_value;
    EXPECT_TRUE(std::equal(orientation.begin(), orientation.end(),
                           orientation_ref.begin()));
    for (auto face : face_family.all()) {
      for (auto cell : connectivity[face]) {
        std::cout << "Face connected cell lid (?) " << cell << std::endl;
      }
      for (auto cell_orientation : orientation[face]) {
        if (cell_orientation == 1)
        std::cout << "Cell is back cell" << std::endl;
        else if (cell_orientation == -1)
        std::cout << "Cell is front cell" << std::endl;
        else
        throw std::runtime_error("Orientation must be 1 or -1");
      }
    }
  }

#ifdef HAS_XDMF
  TEST(PolyhedralTest, CreateXdmfMesh) {
    auto mesh = Neo::Mesh{ "PolyhedralMesh" };
    PolyhedralMeshTest::createMesh(mesh);
    auto exported_mesh{ "test_output.xmf" };
    XdmfTest::exportMesh(mesh, exported_mesh);
    // todo reimport to check
    auto reader = XdmfReader::New();
    auto exported_primaryDomain = shared_dynamic_cast<XdmfDomain>(reader->read(exported_mesh));
    auto ref_primaryDomain = shared_dynamic_cast<XdmfDomain>(reader->read("meshes/example_cell.xmf"));
    auto exported_topology_str = exported_primaryDomain->getUnstructuredGrid("Grid")->getTopology()->getValuesString();
    auto ref_topology_str = ref_primaryDomain->getUnstructuredGrid("Octahedron")->getTopology()->getValuesString();
    auto exported_geometry_str = exported_primaryDomain->getUnstructuredGrid("Grid")->getGeometry()->getValuesString();
    auto ref_geometry_str = ref_primaryDomain->getUnstructuredGrid("Octahedron")->getGeometry()->getValuesString();
    std::cout << "original topology " << ref_topology_str << std::endl;
    std::cout << "exported topology " << exported_topology_str << std::endl;
    std::cout << "original geometry " << ref_geometry_str << std::endl;
    std::cout << "exported geometry " << exported_geometry_str << std::endl;
    EXPECT_EQ(std::string{ ref_topology_str }, std::string{ exported_topology_str }); // comparer avec std::equal
    EXPECT_EQ(std::string{ ref_geometry_str }, std::string{ exported_geometry_str }); // comparer avec std::equal
  }

  TEST(PolyhedralTest, ImportXdmfPolyhedronMesh) {
    auto reader = XdmfReader::New();
    auto primaryDomain = shared_dynamic_cast<XdmfDomain>(reader->read("meshes/example_mesh.xmf"));
    auto grid = primaryDomain->getUnstructuredGrid("Polyhedra");
    auto geometry = grid->getGeometry();
    geometry->read();
    EXPECT_EQ(geometry->getType()->getName(), XdmfGeometryType::XYZ()->getName());
    std::vector<Neo::utils::Real3> node_coords(geometry->getNumberPoints(), { -1e6, -1e6, -1e6 });
    geometry->getValues(0, (double*)node_coords.data(), geometry->getNumberPoints() * 3, 1, 1);
    auto topology = grid->getTopology();
    topology->read();
    // Read only polyhedrons
    EXPECT_EQ(XdmfTopologyType::Polyhedron()->getName(), topology->getType()->getName());
    std::vector<Neo::utils::Int64> cell_data(topology->getSize(), -1);
    topology->getValues(0, cell_data.data(), topology->getSize());
    std::vector<Neo::utils::Int64> cell_uids;
    std::vector<Neo::utils::Int64> face_uids;
    std::vector<Neo::utils::Int64> face_nodes;
    std::set<Neo::utils::Int64> node_uids_set;
    std::vector<Neo::utils::Int64> node_uids;
    std::set<Neo::utils::Int32> current_cell_nodes;
    std::vector<Neo::utils::Int64> cell_nodes;
    std::vector<Neo::utils::Int64> cell_faces;
    std::vector<int> cell_faces_orientation;
    std::vector<int> nb_node_per_cells;
    std::vector<int> nb_face_per_cells;
    using FaceNodes = std::set<int>;
    using FaceUid = Neo::utils::Int64;
    using FaceInfo = std::pair<FaceNodes, FaceUid>;
    auto face_info_comp = [](FaceInfo const& face_info1, FaceInfo const& face_info2) {
      return face_info1.first < face_info2.first;
    };
    std::set<FaceInfo, decltype(face_info_comp)> face_nodes_set(face_info_comp);
    face_nodes.reserve(topology->getSize());
    std::vector<int> nb_node_per_faces;
    auto cell_index = 0;
    auto face_uid = 0;
    for (auto cell_data_index = 0; cell_data_index < (int)cell_data.size();) {
      cell_uids.push_back(cell_index++);
      auto cell_nb_face = cell_data[cell_data_index++];
      nb_face_per_cells.push_back(cell_nb_face);
      for (auto face_index = 0; face_index < cell_nb_face; ++face_index) {
        int face_nb_node = cell_data[cell_data_index++];
        auto current_face_nodes = Neo::utils::ConstSpan<Neo::utils::Int64>{ &cell_data[cell_data_index], face_nb_node };
        auto [face_info, is_new_face] = face_nodes_set.emplace(FaceNodes{ current_face_nodes.begin(),
                                                                          current_face_nodes.end() },
                                                               face_uid);
        if (!is_new_face)
        std::cout << "Face not inserted " << face_uid << std::endl;
        if (is_new_face) {
        face_nodes.insert(face_nodes.end(), current_face_nodes.begin(), current_face_nodes.end());
        nb_node_per_faces.push_back(face_nb_node);
        }
        cell_faces.push_back(face_info->second);
        auto face_orientation = StaticMesh::utilities::DefaultItemOrientation::isOrdered(current_face_nodes) ? 1 : -1;
        cell_faces_orientation.push_back(face_orientation);
        if (is_new_face)
        face_uids.push_back(face_uid++);
        current_cell_nodes.insert(current_face_nodes.begin(), current_face_nodes.end());
        cell_data_index += face_nb_node;
      }
      cell_nodes.insert(cell_nodes.end(), current_cell_nodes.begin(), current_cell_nodes.end());
      nb_node_per_cells.push_back(current_cell_nodes.size());
      node_uids_set.insert(current_cell_nodes.begin(), current_cell_nodes.end());
      current_cell_nodes.clear();
    }
    node_uids.insert(node_uids.end(), std::begin(node_uids_set), std::end(node_uids_set));
    _printContainer(face_nodes, "face nodes ");
    _printContainer(face_uids, "face uids ");
    _printContainer(nb_node_per_faces, "nb node per face ");
    _printContainer(node_uids_set, "node uids ");
    _printContainer(cell_nodes, "cell nodes ");
    _printContainer(nb_node_per_cells, "nb node per cell ");
    _printContainer(cell_faces, "cell faces ");
    _printContainer(nb_face_per_cells, "nb face per cell ");
    _printContainer(cell_faces_orientation, " cell faces orientation");
    // local checks
    std::vector<Neo::utils::Int64> cell_uids_ref = { 0, 1, 2 };
    EXPECT_TRUE(std::equal(cell_uids.begin(), cell_uids.end(), cell_uids_ref.begin(), cell_uids_ref.end()));
    EXPECT_EQ(27, face_uids.size());
    EXPECT_EQ(geometry->getNumberPoints(), node_uids_set.size());
    // get face cells by reversing connectivity
    std::vector<Neo::utils::Int64> face_cells;
    std::vector<Neo::utils::Int64> connected_face_uids;
    std::vector<int> nb_cell_per_faces;
    std::vector<int> face_orientation_in_cells;
    StaticMesh::utilities::reverseConnectivity(cell_uids, cell_faces, nb_face_per_cells,
                                               connected_face_uids, face_cells,
                                               nb_cell_per_faces, cell_faces_orientation,
                                               face_orientation_in_cells);
    _printContainer(face_cells, "  Face cells ");
    _printContainer(nb_cell_per_faces, "  Nb cell per faces ");
    _printContainer(face_orientation_in_cells, "  Face orientation in cells");
    // import mesh in Neo data structure
    auto mesh = Neo::Mesh{ "'ImportedMesh" };
    PolyhedralMeshTest::_createMesh(mesh, node_uids, cell_uids, face_uids,
                                    node_coords, cell_nodes, cell_faces,
                                    face_nodes, face_cells, face_orientation_in_cells,
                                    std::move(nb_node_per_cells),
                                    std::move(nb_face_per_cells),
                                    std::move(nb_node_per_faces),
                                    std::move(nb_cell_per_faces));
    std::string imported_mesh{ "imported_mesh.xmf" };
    XdmfTest::exportMesh(mesh, imported_mesh);
    // Compare with original mesh
    auto created_primaryDomain = shared_dynamic_cast<XdmfDomain>(reader->read(imported_mesh));
    std::cout << "original topology " << topology->getValuesString().c_str() << std::endl;
    std::cout << "created topology " << created_primaryDomain->getUnstructuredGrid("Grid")->getTopology()->getValuesString().c_str() << std::endl;
    std::cout << "original geometry " << geometry->getValuesString().c_str() << std::endl;
    std::cout << "created geometry " << created_primaryDomain->getUnstructuredGrid("Grid")->getGeometry()->getValuesString().c_str() << std::endl;
    EXPECT_EQ(std::string{ geometry->getValuesString().c_str() }, std::string{ created_primaryDomain->getUnstructuredGrid("Grid")->getGeometry()->getValuesString().c_str() }); // comparer avec std::equal
    EXPECT_EQ(std::string{ topology->getValuesString().c_str() }, std::string{ created_primaryDomain->getUnstructuredGrid("Grid")->getTopology()->getValuesString().c_str() }); // comparer avec std::equal
  }

  TEST(PolyhedralTest, ImportXdmfHexahedronMesh) {
    auto reader = XdmfReader::New();
    auto primaryDomain = shared_dynamic_cast<XdmfDomain>(
    reader->read("meshes/example_hexahedron.xmf"));
    auto grid = primaryDomain->getUnstructuredGrid("Hexahedron");
    auto geometry = grid->getGeometry();
    geometry->read();
    EXPECT_EQ(geometry->getType()->getName(), XdmfGeometryType::XYZ()->getName());
    std::vector<Neo::utils::Real3> node_coords(geometry->getNumberPoints(),
                                               { -1e6, -1e6, -1e6 });
    geometry->getValues(0, (double*)node_coords.data(),
                        geometry->getNumberPoints() * 3, 1, 1);
    auto topology = grid->getTopology();
    topology->read();
    // Read only polyhedrons
    EXPECT_EQ(XdmfTopologyType::Hexahedron()->getName(),
              topology->getType()->getName());
    std::vector<Neo::utils::Int32> cell_data(topology->getSize(), -1);
    topology->getValues(0, cell_data.data(), topology->getSize());
    std::vector<Neo::utils::Int64> cell_uids;
    std::set<Neo::utils::Int64> node_uids_set;
    std::vector<Neo::utils::Int64> node_uids;
    std::set<Neo::utils::Int32> current_cell_nodes;
    std::vector<Neo::utils::Int64> cell_nodes;
    int cell_nb_nodes = 8;
    auto cell_index = 0;
    for (auto cell_data_index = 0; cell_data_index < (int)cell_data.size();) {
      cell_uids.push_back(cell_index++);
      auto current_cell_nodes = Neo::utils::ConstSpan<Neo::utils::Int32>{ &cell_data[cell_data_index], cell_nb_nodes };
      cell_nodes.insert(cell_nodes.end(), current_cell_nodes.begin(),
                        current_cell_nodes.end());
      node_uids_set.insert(current_cell_nodes.begin(), current_cell_nodes.end());
      cell_data_index += cell_nb_nodes;
    }
    node_uids.insert(node_uids.end(), std::begin(node_uids_set),
                     std::end(node_uids_set));
    _printContainer(node_uids, "node uids ");
    _printContainer(cell_nodes, "cell nodes ");
    auto mesh = Neo::Mesh{ "ImportedHexMesh" };
    using CellTypeIndexes = std::vector<int>;
    std::vector<Neo::utils::Int64> face_nodes;
    std::vector<Neo::utils::Int32> cell_face_indexes;
    std::vector<int> cell_face_orientations;
    int nb_faces = 0;
    StaticMesh::utilities::getItemConnectivityFromCell(
    cell_nodes, CellTypeIndexes{ 0 },
    { { 8,
        { { 0, 3, 2, 1 },
          { 1, 2, 6, 5 },
          { 4, 5, 6, 7 },
          { 2, 3, 7, 6 },
          { 0, 3, 7, 4 },
          { 0, 1, 5, 4 } } } },
    nb_faces, face_nodes, cell_face_indexes, cell_face_orientations);
    std::vector<Neo::utils::Int64> face_uids(nb_faces);
    std::vector<Neo::utils::Int64> cell_faces(cell_face_indexes.size());
    std::copy(cell_face_indexes.begin(), cell_face_indexes.end(), cell_faces.begin()); // face indexes are taken as uids
    std::iota(face_uids.begin(), face_uids.end(), 0);
    _printContainer(cell_faces, " cell faces");
    // get face cells by reversing connectivity
    std::vector<Neo::utils::Int64> face_cells;
    std::vector<Neo::utils::Int64> connected_face_uids;
    std::vector<int> nb_cell_per_faces;
    std::vector<int> face_orientation_in_cells;
    StaticMesh::utilities::reverseConnectivity(cell_uids, cell_faces, std::vector<int>(cell_uids.size(), 6),
                                               connected_face_uids, face_cells, nb_cell_per_faces,
                                               cell_face_orientations, face_orientation_in_cells);
    _printContainer(face_cells, "  Face cells ");
    _printContainer(nb_cell_per_faces, "  Nb cell per faces ");
    _printContainer(cell_face_orientations, " Cell faces orientations ");
    _printContainer(face_orientation_in_cells, "Face orientation in cells");
    PolyhedralMeshTest::_createMesh(mesh, node_uids, cell_uids, face_uids,
                                    node_coords, cell_nodes, cell_faces,
                                    face_nodes, face_cells, face_orientation_in_cells,
                                    std::vector<int>(cell_uids.size(), 8),
                                    std::vector<int>(cell_uids.size(), 6),
                                    std::vector<int>(face_uids.size(), 4),
                                    std::move(nb_cell_per_faces));
  }

#endif //HAS_XDMF
