// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NeoBaseTest.cpp                                 (C) 2000-2023             */
/*                                                                           */
/* Base tests for Neo kernel                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <vector>
#include <array>
#include <algorithm>

#include "gtest/gtest.h"
#include "neo/Neo.h"
#include "neo/MeshKernel.h"

TEST(NeoUtils, test_array_view) {
  std::vector<int> vec{ 0, 1, 2 };
  // build a view from a vector
  Neo::utils::Span<int> view{ vec.data(), vec.size()  };
  Neo::utils::ConstSpan<int> constview{vec.data(), vec.size()};
  EXPECT_TRUE(std::equal(view.begin(), view.end(), vec.begin()));
  EXPECT_TRUE(std::equal(constview.begin(), constview.end(), vec.begin()));
  // build a vector from a view
  std::vector<int> vec2 = view.copy();
  std::vector<int> vec3(view.copy());
  std::vector<int> vec4(constview.copy());
  EXPECT_TRUE(std::equal(vec2.begin(), vec2.end(), view.begin()));
  EXPECT_TRUE(std::equal(vec3.begin(), vec3.end(), view.begin()));
  EXPECT_TRUE(std::equal(vec4.begin(), vec4.end(), constview.begin()));
  std::vector<int> dim2_vec{ 0, 1, 2, 3, 4, 5 };
  // build a dim2 view from vector
  auto dim1_size = 2;
  auto dim2_size = 3;
  Neo::utils::Span2<int> dim2_view{ dim2_vec.data(), dim1_size, dim2_size };
  Neo::utils::ConstSpan2<int> dim2_const_view{ dim2_vec.data(), dim2_size, dim1_size };
  for (auto i = 0; i < dim1_size; ++i) {
    for (auto j = 0; j < dim2_size; ++j) {
      EXPECT_EQ(dim2_view[i][j], dim2_vec[i * dim2_size + j]);
      EXPECT_EQ(dim2_const_view[j][i], dim2_vec[j * dim1_size + i]);
    }
  }
  // Copy all Array2View data into a 1D vector
  std::vector<int> dim2_view_vector_copy{ dim2_view.copy() };
  std::vector<int> dim2_const_view_vector_copy{ dim2_const_view.copy() };
  EXPECT_TRUE(std::equal(dim2_vec.begin(), dim2_vec.end(), dim2_const_view_vector_copy.begin()));
  // Try out of bound error
#ifndef _MS_REL_ // if constepxr still experiencing problems with
#ifdef USE_GTEST_DEATH_TEST
  if constexpr (_debug) {
    EXPECT_DEATH(view[4], ".*i < m_size.*");
  }
  if constexpr (_debug) {
    EXPECT_DEATH(dim2_view[0][4], ".*i < m_size.*");
  }
#endif
#endif
}

void _testItemLocalIds(Neo::utils::Int32 const& first_lid,
                       int const& nb_lids,
                       std::vector<Neo::utils::Int32> const& non_contiguous_lids = {}) {
  auto item_local_ids = Neo::ItemLocalIds{ non_contiguous_lids, first_lid, nb_lids };
  auto item_array = item_local_ids.itemArray();
  EXPECT_EQ(item_local_ids.size(), nb_lids + non_contiguous_lids.size());
  Neo::utils::printContainer(item_array, "ItemLocalIds");
  std::vector<Neo::utils::Int32> item_array_ref(nb_lids);
  item_array_ref.insert(item_array_ref.begin(), non_contiguous_lids.begin(),
                        non_contiguous_lids.end());
  std::iota(item_array_ref.begin() + non_contiguous_lids.size(), item_array_ref.end(), first_lid);
  Neo::utils::printContainer(item_array_ref, "ItemArrayRef");
  EXPECT_TRUE(std::equal(item_array.begin(), item_array.end(), item_array_ref.begin()));
  for (int i = 0; i < item_local_ids.size(); ++i) {
    EXPECT_EQ(item_local_ids(i), item_array_ref[i]);
  }
  item_local_ids.clear();
  EXPECT_EQ(item_local_ids.size(),0);
  EXPECT_EQ(item_local_ids.itemArray().size(),0);
  EXPECT_EQ(item_local_ids.maxLocalId(),Neo::utils::NULL_ITEM_LID);
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestItemLocalIds, test_item_local_ids) {
  _testItemLocalIds(0, 10);
  _testItemLocalIds(5, 10);
  _testItemLocalIds(0, 0, { 1, 3, 5, 9 });
  _testItemLocalIds(0, 10, { 1, 3, 5, 9 });
  _testItemLocalIds(5, 10, { 1, 3, 5, 9 });
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestItemRange, test_item_range) {
  // Test with only contiguous local ids
  std::cout << "== Testing contiguous item range from 0 with 5 items ==" << std::endl;
  auto nb_item = 5;
  auto ir = Neo::ItemRange{ Neo::ItemLocalIds{ {}, 0, nb_item } };
  EXPECT_EQ(ir.size(), nb_item);
  EXPECT_EQ(ir.maxLocalId(), nb_item - 1);
  std::vector<Neo::utils::Int32> local_ids;
  for (auto item : ir) {
    std::cout << "item lid " << item << std::endl;
    local_ids.push_back(item);
  }
  auto local_ids_stored = ir.localIds();
  std::cout << local_ids_stored << std::endl;
  EXPECT_TRUE(std::equal(local_ids_stored.begin(), local_ids_stored.end(), local_ids.begin()));
  local_ids.clear();
  // Test with only non contiguous local ids
  std::cout << "== Testing non contiguous item range {3,5,7} ==" << std::endl;
  std::vector non_contiguous_lids = { 3, 5, 7 };
  ir = Neo::ItemRange{ Neo::ItemLocalIds{ non_contiguous_lids, 0, 0 } };
  EXPECT_EQ(ir.size(), non_contiguous_lids.size());
  EXPECT_EQ(ir.maxLocalId(), *std::max_element(non_contiguous_lids.begin(), non_contiguous_lids.end()));
  for (auto item : ir) {
    std::cout << "item lid " << item << std::endl;
    local_ids.push_back(item);
  }
  local_ids_stored = ir.localIds();
  std::cout << local_ids_stored << std::endl;
  EXPECT_TRUE(std::equal(local_ids_stored.begin(), local_ids_stored.end(), local_ids.begin()));
  local_ids.clear();
  // Test range mixing contiguous and non contiguous local ids
  std::cout << "== Testing non contiguous item range {3,5,7} + 8 to 11 ==" << std::endl;
  auto nb_contiguous_lids = 4;
  auto first_contiguous_lid = 8;
  ir = Neo::ItemRange{ Neo::ItemLocalIds{ non_contiguous_lids, first_contiguous_lid, nb_contiguous_lids } };
  EXPECT_EQ(ir.size(), non_contiguous_lids.size() + nb_contiguous_lids);
  EXPECT_EQ(ir.maxLocalId(), first_contiguous_lid + nb_contiguous_lids - 1);
  for (auto item : ir) {
    std::cout << "item lid " << item << std::endl;
    local_ids.push_back(item);
  }
  local_ids_stored = ir.localIds();
  std::cout << local_ids_stored << std::endl;
  EXPECT_TRUE(std::equal(local_ids_stored.begin(), local_ids_stored.end(), local_ids.begin()));
  local_ids.clear();
  // Internal test for out of bound
  std::cout << "Get out of bound values (index > size) " << ir.m_item_lids(100) << std::endl;
  std::cout << "Get out of bound values (index < 0) " << ir.m_item_lids(-100) << std::endl;
  EXPECT_EQ(ir.m_item_lids(100),ir.m_item_lids.size()); // for iterator, past end is equal to size
  EXPECT_EQ(ir.m_item_lids(-100),Neo::utils::NULL_ITEM_LID);
  ir.clear();
  EXPECT_EQ(ir.size(),0);
  EXPECT_EQ(ir.localIds().size(),0);
  auto check_null = 0;
  for (auto item : ir) {
    check_null += item;
  }
  EXPECT_EQ(check_null,0);
  EXPECT_EQ(ir.maxLocalId(),Neo::utils::NULL_ITEM_LID);
  EXPECT_TRUE(ir.isEmpty());

  // todo test out reverse range
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestFutureItemRange, test_future_item_range) {
  Neo::FutureItemRange future_item_range{};
  EXPECT_EQ(future_item_range.size(), 0);
  // Manually fill contained ItemRange
  std::vector<Neo::utils::Int32> lids{ 0, 2, 4, 6 };
  future_item_range.new_items = Neo::ItemRange{ Neo::ItemLocalIds{ lids } };
  EXPECT_EQ(future_item_range.size(), lids.size());
  Neo::ItemRange& internal_range = future_item_range;
  EXPECT_EQ(&future_item_range.new_items, &internal_range);
  auto end_update = Neo::EndOfMeshUpdate{};
  {
    // declare a filtered range -- filtered by indexes
    std::vector<int> filter{ 0, 1, 2 };
    auto filtered_future_range =
    Neo::make_future_range(future_item_range, filter);
    EXPECT_EQ(filtered_future_range.size(), 0);
    // Get item_ranges
    auto filtered_range = filtered_future_range.get(end_update);
    auto item_range = future_item_range.get(end_update);
    // Check item ranges
    auto lids_in_range = item_range.localIds();
    EXPECT_TRUE(std::equal(lids.begin(), lids.end(), lids_in_range.begin()));
    EXPECT_THROW(future_item_range.get(end_update), std::runtime_error);

    std::vector<Neo::utils::Int32> filtered_lids = filtered_range.localIds();
    std::vector<Neo::utils::Int32> filtered_lids_ref;
    for (auto i : filter) {
      filtered_lids_ref.push_back(lids[i]);
    }
    EXPECT_EQ(filtered_lids.size(), filtered_lids_ref.size());
    EXPECT_TRUE(std::equal(filtered_lids_ref.begin(), filtered_lids_ref.end(),
                           filtered_lids.begin()));
  }
  Neo::FutureItemRange future_item_range2{};
  // Manually fill contained ItemRange
  future_item_range2.new_items = Neo::ItemRange{ Neo::ItemLocalIds{ lids } };
  {
    // declare a filtered range -- filtered by values (the filter is computed)
    std::vector<Neo::utils::Int32> value_subset{ 2, 6 };
    auto filtered_future_range = Neo::make_future_range(future_item_range2, lids, value_subset);
    // Get item_range
    auto filtered_range = filtered_future_range.get(end_update);
    auto filtered_range_lids = filtered_range.localIds();
    EXPECT_TRUE(std::equal(value_subset.begin(), value_subset.end(), filtered_range_lids.begin()));
  }
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestPropertyGraph, test_property_graph_info) {
  std::cout << "Test Property Graph" << std::endl;
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test" };

  // Add a family : property always belong to a family
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cells" };

  // Add a consuming/producing algo
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "in_property" }, Neo::MeshKernel::OutProperty{ cell_family, "out_property" }, []() {});
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "in_property" }, Neo::MeshKernel::InProperty{ cell_family, "in_property2" },
                    Neo::MeshKernel::OutProperty{ cell_family, "out_property" }, []() {});
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "in_property2" }, Neo::MeshKernel::InProperty{ cell_family, "in_property2" },
                    Neo::MeshKernel::OutProperty{ cell_family, "out_property2" }, []() {});

  // add producing algos
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ cell_family, "out_property2" }, []() {});
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ cell_family, "out_property2" },
                    Neo::MeshKernel::OutProperty{ cell_family, "out_property3" }, []() {});

  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "in_property" },
                    Neo::MeshKernel::InProperty{ cell_family, "in_property2" },
                    Neo::MeshKernel::OutProperty{ cell_family, "out_property2" },
                    []() {});

  // Check number of consuming algo
  auto nb_algo_consuming_in_property = 3;
  EXPECT_EQ(nb_algo_consuming_in_property, mesh.m_property_algorithms.find(Neo::MeshKernel::InProperty{ cell_family, "in_property" })->second.second.size());

  auto nb_algo_consuming_in_property2 = 4;
  EXPECT_EQ(nb_algo_consuming_in_property2, mesh.m_property_algorithms.find(Neo::MeshKernel::InProperty{ cell_family, "in_property2" })->second.second.size());

  // check number of producing algos
  auto nb_algo_producing_out_property = 2;
  EXPECT_EQ(nb_algo_producing_out_property, mesh.m_property_algorithms.find(Neo::MeshKernel::OutProperty{ cell_family, "out_property" })->second.first.size());
  auto nb_algo_producing_out_property2 = 4;
  EXPECT_EQ(nb_algo_producing_out_property2, mesh.m_property_algorithms.find(Neo::MeshKernel::OutProperty{ cell_family, "out_property2" })->second.first.size());
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestLidsProperty, test_lids_property) {
  std::cout << "Test lids_range Property" << std::endl;
  auto lid_prop = Neo::ItemLidsProperty{ "test_property" };
  EXPECT_EQ(lid_prop.name(), "test_property");
  // Check empty property
  auto empty_range = lid_prop.values();
  EXPECT_EQ(empty_range.size(), 0);
  std::vector<Neo::utils::Int64> uids{ 1, 2, 3, 4, 5 };
  auto nb_item = uids.size();
  // Checking append
  auto added_item_range = lid_prop.append(uids);
  lid_prop.debugPrint();
  EXPECT_EQ(uids.size(), lid_prop.size());
  auto i = 0;
  auto added_local_ids = lid_prop[uids];
  auto added_local_ids_ref = added_item_range.localIds();
  EXPECT_TRUE(std::equal(added_local_ids.begin(), added_local_ids.end(), added_local_ids_ref.begin()));
  for (auto item : added_item_range) {
    std::cout << " uid " << uids[i++] << " lid " << item << std::endl;
  }
  // Check values function
  auto lids_range = lid_prop.values();
  EXPECT_EQ(lids_range.size(), uids.size());
  EXPECT_TRUE(std::equal(lids_range.begin(), lids_range.end(), added_local_ids_ref.begin()));
  // Checking append with duplicates
  uids = { 6, 7, 7, 8, 1, 5, 9 };
  auto one_lid = lid_prop[{ 1 }]; // store lid of uid =1 must not change
  auto five_lid = lid_prop[{ 5 }]; // store lid of uid =5 must not change
  auto nb_duplicates = 3;
  nb_item += uids.size() - nb_duplicates;
  added_item_range = lid_prop.append(uids);
  lid_prop.debugPrint();
  i = 0;
  for (auto item : added_item_range) {
    std::cout << " uid " << uids[i++] << " lid " << item << std::endl;
  }
  added_local_ids = lid_prop[uids];
  added_local_ids_ref = added_item_range.localIds();
  EXPECT_TRUE(std::equal(added_local_ids.begin(), added_local_ids.end(), added_local_ids_ref.begin()));
  EXPECT_EQ(one_lid, lid_prop[{ 1 }]);
  EXPECT_EQ(five_lid, lid_prop[{ 5 }]);
  // Checking remove
  auto removed_uids = std::vector<Neo::utils::Int64>{ 1, 3, 5, 9 };
  auto removed_lids_ref = lid_prop[removed_uids];
  auto removed_lids = lid_prop.remove(removed_uids);
  nb_item -= removed_uids.size();
  EXPECT_TRUE(std::equal(removed_lids_ref.begin(), removed_lids_ref.end(), removed_lids.begin()));
  for (auto lid : removed_lids) {
    std::cout << "removed lids_range: " << lid << std::endl;
  }
  // Checking value function
  for (auto item : lid_prop.values()) {
    std::cout << "Item range, lid " << item << std::endl;
  }
  EXPECT_EQ(lid_prop.values().size(), lid_prop.size());
  EXPECT_EQ(lid_prop.values().size(), nb_item);
  std::vector<Neo::utils::Int64> remaining_uids{ 2, 4, 6, 7, 8 };
  auto lids_ref = lid_prop[remaining_uids];
  auto lids = lid_prop.values().localIds();
  EXPECT_TRUE(std::equal(lids_ref.begin(), lids_ref.end(), lids.begin()));
  lid_prop.debugPrint();
  // Check re-add removed items
  std::vector<Neo::utils::Int64> added_uids(removed_uids);
  auto added_items = lid_prop.append(removed_uids);
  nb_item += removed_lids.size();
  lid_prop.debugPrint();
  EXPECT_EQ(added_items.size(), removed_uids.size());
  EXPECT_EQ(std::count(added_items.begin(), added_items.end(), Neo::utils::NULL_ITEM_LID), 0);
  auto added_lids = added_items.localIds();
  auto added_lids_ref = lid_prop[added_uids];
  EXPECT_TRUE(std::equal(added_lids.begin(), added_lids.end(), added_lids_ref.begin()));
  //Check add new items
  added_uids = { 10, 11, 12 };
  added_items = lid_prop.append(added_uids);
  nb_item += added_items.size();
  lid_prop.debugPrint();
  EXPECT_EQ(added_items.size(), 3);
  EXPECT_EQ(std::count(added_items.begin(), added_items.end(), Neo::utils::NULL_ITEM_LID), 0);
  added_lids = added_items.localIds();
  added_lids_ref = lid_prop[added_uids];
  EXPECT_TRUE(std::equal(added_lids.begin(), added_lids.end(), added_lids_ref.begin()));
  // Checking value function
  for (auto item : lid_prop.values()) {
    std::cout << "Item range, lid " << item << std::endl;
  }
  EXPECT_EQ(lid_prop.values().size(), lid_prop.size());
  EXPECT_EQ(lid_prop.values().size(), nb_item);
  remaining_uids = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  lids_ref = lid_prop[remaining_uids];
  lids = lid_prop.values().localIds();
  // reorder lids in a set since the order is not warrantied (use of removed lids)
  std::set<Neo::utils::Int32> reordered_lids_ref{ lids_ref.begin(), lids_ref.end() };
  std::set<Neo::utils::Int32> reordered_lids{ lids.begin(), lids.end() };
  EXPECT_EQ(reordered_lids_ref.size(), reordered_lids.size());
  EXPECT_TRUE(std::equal(reordered_lids_ref.begin(), reordered_lids_ref.end(), reordered_lids.begin()));
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestFamily, test_family) {
  Neo::Family family(Neo::ItemKind::IK_Dof, "MyFamily");
  EXPECT_EQ(family.lidPropName(), family._lidProp().m_name);
  EXPECT_TRUE(family.hasProperty(family.lidPropName()));
  EXPECT_FALSE(family.hasProperty());
  EXPECT_FALSE(family.hasProperty("toto"));
  std::vector<Neo::utils::Int64> uids{ 0, 1, 2 };
  family._lidProp().append(uids); // internal
  EXPECT_EQ(3, family.nbElements());
  std::string scalar_prop_name("MyScalarProperty");
  std::string array_prop_name("MyArrayProperty");
  family.addMeshScalarProperty<Neo::utils::Int32>(scalar_prop_name);
  family.addMeshArrayProperty<Neo::utils::Int32>(array_prop_name);
  EXPECT_NO_THROW(family.getProperty(scalar_prop_name));
  EXPECT_NO_THROW(family.getProperty(array_prop_name));
  EXPECT_THROW(family.getProperty("UnexistingProperty"), std::invalid_argument);
  EXPECT_EQ(scalar_prop_name, family.getConcreteProperty<Neo::MeshScalarPropertyT<Neo::utils::Int32>>(scalar_prop_name).m_name);
  EXPECT_EQ(array_prop_name, family.getConcreteProperty<Neo::MeshArrayPropertyT<Neo::utils::Int32>>(array_prop_name).m_name);
  EXPECT_EQ(3, family.all().size());
  auto i = 0;
  auto local_ids = family.itemUniqueIdsToLocalids(uids);
  for (auto item : family.all()) {
    EXPECT_EQ(local_ids[i++], item);
  }
  family.itemUniqueIdsToLocalids(local_ids, uids);
  i = 0;
  for (auto item : family.all()) {
    EXPECT_EQ(local_ids[i++], item);
  }
  family.removeProperty(scalar_prop_name);
  EXPECT_FALSE(family.hasProperty(scalar_prop_name));
  family.removeProperty(array_prop_name);
  EXPECT_FALSE(family.hasProperty(array_prop_name));
  EXPECT_FALSE(family.hasProperty());
  // try removing an unexisting property
  family.removeProperty(scalar_prop_name);
  family.addMeshScalarProperty<Neo::utils::Int32>(scalar_prop_name);
  family.addMeshArrayProperty<Neo::utils::Int32>(array_prop_name);
  family.removeProperties();
  EXPECT_FALSE(family.hasProperty());
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestBaseMesh, base_mesh_unit_test) {
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test" };
  Neo::Family family1{ Neo::ItemKind::IK_Cell, "family1" };
  Neo::Family family2{ Neo::ItemKind::IK_Cell, "family2" };
  Neo::Family family3{ Neo::ItemKind::IK_Node, "family3" };
  Neo::Family family4{ Neo::ItemKind::IK_Edge, "family4" };
  Neo::Family family5{ Neo::ItemKind::IK_Dof, "family5" };
  bool is_called = false;
  family1.addMeshScalarProperty<int>("prop1");
  family2.addMeshScalarProperty<int>("prop2");
  family3.addMeshScalarProperty<int>("prop3");
  family4.addMeshScalarProperty<int>("prop4");
  family5.addMeshScalarProperty<int>("prop5");
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ family1, "prop1", Neo::PropertyStatus::ExistingProperty }, Neo::MeshKernel::OutProperty{ family2, "prop2" },
                    [&is_called]([[maybe_unused]] Neo::MeshScalarPropertyT<int> const& prop1,
                                 [[maybe_unused]] Neo::MeshScalarPropertyT<int>& prop2) {
                      is_called = true;
                    });
  // copy mesh
  auto mesh2 = mesh;
  // call algorithm on mesh 2
  mesh2.applyAlgorithms();
  EXPECT_TRUE(is_called);
  // now mesh 2 does not contain any algorithm
  is_called = false;
  mesh2.applyAlgorithms();
  EXPECT_FALSE(is_called);
  // call and keep algorithm on mesh1: algo will not be removed from mesh
  mesh.applyAndKeepAlgorithms();
  EXPECT_TRUE(is_called);
  // algo is still there and can be applied again
  is_called = false;
  mesh.applyAlgorithms();
  EXPECT_TRUE(is_called);
}

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
