// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NeoBaseTest.cpp                                 (C) 2000-2026             */
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

/*-----------------------------------------------------------------------------*/

TEST(NeoTestArrayProperty, test_mesh_array_property) {
  auto mesh_array_property = Neo::MeshArrayPropertyT<Neo::utils::Int32>{ "test_mesh_array_property" };
  EXPECT_EQ(mesh_array_property.name(), "test_mesh_array_property");
  // check assert (debug only)
#ifndef _MS_REL_ // if constepxr still experiencing problems with MSVC
#ifdef USE_GTEST_DEATH_TEST
  if constexpr (_debug) {
    EXPECT_DEATH(mesh_array_property[Neo::utils::NULL_ITEM_LID], ".*Item local id.*");
  }
#endif
#endif
  // Check data allocation with resize (not to be used if init method is used)
  std::vector<int> sizes{ 1, 2, 3 };
  mesh_array_property.resize(sizes, true); // 3 elements with respectively 1, 2 and 3 values
  auto array_property_sizes = mesh_array_property.sizes();
  EXPECT_TRUE(std::equal(array_property_sizes.begin(), array_property_sizes.end(), sizes.begin()));
  EXPECT_EQ(mesh_array_property.size(), 3);
  EXPECT_EQ(mesh_array_property.cumulatedSize(), 6);
  mesh_array_property.clear();
  EXPECT_EQ(mesh_array_property.size(), 0);
  // add elements: 5 items with one value
  Neo::ItemRange item_range{ Neo::ItemLocalIds{ {}, 0, 5 } };
  std::vector<Neo::utils::Int32> values{ 0, 1, 2, 3, 4 };
  // Check cannot Try to init before resize
#ifndef _MS_REL_ // if constepxr still experiencing problems with MSVC
#ifdef USE_GTEST_DEATH_TEST
  if constexpr (_debug) {
    EXPECT_DEATH(mesh_array_property.init(item_range, values), ".*call resize before init.*");
  }
#endif
#endif
  mesh_array_property.resize({ 1, 1, 1, 1, 1 });
  mesh_array_property.init(item_range, values);
  mesh_array_property.debugPrint();
  EXPECT_EQ(item_range.size(), mesh_array_property.size());
  EXPECT_EQ(values.size(), mesh_array_property.cumulatedSize());
  // check values
  auto index = 0;
  for (auto item : item_range) {
    auto item_array = mesh_array_property[item];
    for (auto item_value : item_array) {
      EXPECT_EQ(item_value, values[index++]);
    }
  }
  // check iterators
  EXPECT_TRUE(std::equal(mesh_array_property.begin(), mesh_array_property.end(), values.begin()));
  // check view
  auto property_1D_view = mesh_array_property.view();
  auto const property_1D_const_view = mesh_array_property.constView();
  EXPECT_TRUE(std::equal(property_1D_view.begin(), property_1D_view.end(), values.begin()));
  EXPECT_TRUE(std::equal(property_1D_const_view.begin(), property_1D_const_view.end(), values.begin()));
  // Add 3 items
  std::vector<int> nb_element_per_item{ 0, 3, 1 };
  Neo::ItemRange item_range2 = { Neo::ItemLocalIds{ { 5, 6, 7 } } };
  std::vector<Neo::utils::Int32> values_added{ 6, 6, 6, 7 };
  mesh_array_property.append(item_range2, values_added, nb_element_per_item);
  mesh_array_property.debugPrint(); // expected result: "0" "1" "2" "3" "4" "6" "6" "6" "7" (check with test framework)
  EXPECT_EQ(item_range.size()+item_range2.size(), mesh_array_property.size());
  EXPECT_EQ(values.size() + values_added.size(), mesh_array_property.cumulatedSize());
  std::vector<int> ref_values = { 0, 1, 2, 3, 4, 6, 6, 6, 7 };
  EXPECT_TRUE(std::equal(ref_values.begin(), ref_values.end(), mesh_array_property.begin()));
  // check values
  index = 0;
  for (auto item : item_range2) {
    auto item_array = mesh_array_property[item];
    for (auto item_value : item_array) {
      EXPECT_EQ(item_value, values_added[index++]);
    }
  }
  // Add three more items
  Neo::ItemRange item_range3 = { Neo::ItemLocalIds{ {}, 8, 3 } };
  std::for_each(values_added.begin(), values_added.end(), [](auto& elt) { return elt += 2; });
  mesh_array_property.append(item_range3, values_added, nb_element_per_item);
  mesh_array_property.debugPrint(); // expected result: "0" "1" "2" "3" "4" "6" "6" "6" "7" "8" "8" "8" "9"
  EXPECT_EQ(item_range.size()+item_range2.size()+item_range3.size(), mesh_array_property.size());
  EXPECT_EQ(values.size() + 2 * values_added.size(), mesh_array_property.cumulatedSize());
  ref_values = { 0, 1, 2, 3, 4, 6, 6, 6, 7, 8, 8, 8, 9 };
  EXPECT_TRUE(std::equal(ref_values.begin(), ref_values.end(), mesh_array_property.begin()));
  // check values
  index = 0;
  for (auto item : item_range3) {
    auto item_array = mesh_array_property[item];
    for (auto item_value : item_array) {
      EXPECT_EQ(item_value, values_added[index++]);
    }
  }
  // Add items and modify existing item
  item_range = { Neo::ItemLocalIds{ { 0, 8, 5 }, 11, 1 } };
  nb_element_per_item = { 3, 3, 2, 1 };
  values_added = { 10, 10, 10, 11, 11, 11, 12, 12, 13 };
  mesh_array_property.append(item_range, values_added, nb_element_per_item); // expected result: "10" "10" "10" "1" "2" "3" "4" "12" "12" "6" "6" "6" "7" "11" "11" "11" "8" "8" "8" "9" "13"
  mesh_array_property.debugPrint();
  EXPECT_EQ(21, mesh_array_property.cumulatedSize());
  ref_values = { 10, 10, 10, 1, 2, 3, 4, 12, 12, 6, 6, 6, 7, 11, 11, 11, 8, 8, 8, 9, 13 };
  EXPECT_TRUE(std::equal(ref_values.begin(), ref_values.end(), mesh_array_property.begin()));
  // check values
  index = 0;
  for (auto item : item_range) {
    auto item_array = mesh_array_property[item];
    for (auto item_value : item_array) {
      EXPECT_EQ(item_value, values_added[index++]);
    }
  }

  // Check add non-0-starting contiguous range in an empty array property
  auto array_property2 = Neo::MeshArrayPropertyT<Neo::utils::Int32>{ "test_array_property2" };
  item_range = { Neo::ItemLocalIds{ {}, 3, 4 } };
  values = { 3, 4, 4, 5, 6, 6 };
  array_property2.append(item_range, values, { 1, 2, 1, 2 });
  array_property2.debugPrint();
  std::vector<int> values_check;
  for (auto item : item_range) {
    for (auto value : array_property2[item])
      values_check.push_back(value);
  }
  Neo::printer() << values_check << Neo::endline;
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));
  item_range = { Neo::ItemLocalIds{ {}, 0, 2 } };
  values = { 0, 1, 1 };
  array_property2.append(item_range, values, { 1, 2 });
  array_property2.debugPrint();
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property2[item])
      values_check.push_back(value);
  }
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));
  // Check for the whole range
  item_range = { Neo::ItemLocalIds{ {}, 0, 7 } };
  values = { 0, 1, 1, 3, 4, 4, 5, 6, 6 };
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property2[item]) {
      values_check.push_back(value);
    }
  }
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));

  // Check with existing property but insertion past the last element
  item_range = { Neo::ItemLocalIds{ {}, 8, 3 } }; // lids {8,9,10}
  values = { 8, 9, 9, 10 };
  array_property2.append(item_range, values, { 1, 2, 1 });
  array_property2.debugPrint();
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property2[item])
      values_check.push_back(value);
  }
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));

  // Same two tests with discontiguous range
  auto array_property3 = Neo::MeshArrayPropertyT<Neo::utils::Int32>{ "test_array_property3" };
  item_range = { Neo::ItemLocalIds{ { 3, 5, 6 } } };
  values = { 3, 3, 5, 6, 6 };
  array_property3.append(item_range, values, { 2, 1, 2 });
  array_property3.debugPrint();
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property3[item])
      values_check.push_back(value);
  }
  Neo::printer() << values_check << Neo::endline;
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));
  // Fill the first items
  item_range = { Neo::ItemLocalIds{ { 0, 2 } } };
  values = { 0, 2, 2 };
  array_property3.append(item_range, values, { 1, 2 });
  array_property3.debugPrint();
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property3[item])
      values_check.push_back(value);
  }
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));
  // Check for the whole range
  item_range = { Neo::ItemLocalIds{ {}, 0, 7 } };
  values = { 0, 2, 2, 3, 3, 5, 6, 6 };
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property3[item]) {
      values_check.push_back(value);
    }
  }
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));
  // Check with existing property but insertion past the last element
  item_range = { Neo::ItemLocalIds{ { 8, 10, 12 } } }; // lids {8,9,10}
  values = { 8, 10, 10, 12 };
  array_property3.append(item_range, values, { 1, 2, 1 });
  array_property3.debugPrint();
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property3[item])
      values_check.push_back(value);
  }
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));
  // Check for the whole range
  item_range = { Neo::ItemLocalIds{ {}, 0, 13 } };
  values = { 0, 2, 2, 3, 3, 5, 6, 6, 8, 10, 10, 12 };
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property3[item]) {
      values_check.push_back(value);
    }
  }
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));

  // Same two tests with mixed range
  auto array_property4 = Neo::MeshArrayPropertyT<Neo::utils::Int32>{ "test_array_property4" };
  item_range = { Neo::ItemLocalIds{ { 4, 6, 7 }, 8, 3 } };
  values = { 4, 4, 6, 7, 7, 8, 9, 10, 10 };
  array_property4.append(item_range, values, { 2, 1, 2, 1, 1, 2 });
  array_property4.debugPrint();
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property4[item])
      values_check.push_back(value);
  }
  Neo::printer() << values_check << Neo::endline;
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));
  // Fill the first items
  item_range = { Neo::ItemLocalIds{ { 2, 3 }, 0, 2 } };
  values = { 2, 2, 3, 0, 0, 1 };
  array_property4.append(item_range, values, { 2, 1, 2, 1 });
  array_property4.debugPrint();
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property4[item])
      values_check.push_back(value);
  }
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));
  // Check for the whole range
  item_range = { Neo::ItemLocalIds{ {}, 0, 11 } };
  values = { 0, 0, 1, 2, 2, 3, 4, 4, 6, 7, 7, 8, 9, 10, 10 };
  values_check.clear();
  for (auto item : item_range) {
    for (auto value : array_property4[item]) {
      values_check.push_back(value);
    }
  }
  EXPECT_TRUE(std::equal(values.begin(), values.end(), values_check.begin()));

  // Check clear method
  mesh_array_property.clear();
  EXPECT_EQ(mesh_array_property.size(), 0);
  // Since property cleared, an init can be called after a resize
  mesh_array_property.resize({ 1, 1, 1, 2, 2 });
  mesh_array_property.init(item_range, values);
  EXPECT_EQ(mesh_array_property.size(), 5);
  EXPECT_EQ(mesh_array_property.cumulatedSize(), 7);
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestArrayProperty, test_mesh_array_property_proxy) {
  Neo::MeshArrayPropertyT<Neo::utils::Int32> mesh_array_property{"mesh_array_property_test"};
  mesh_array_property.resize({1,2,3});
  auto item_range = Neo::ItemRange{ Neo::ItemLocalIds{ {}, 0, 3 } };
  mesh_array_property.init(item_range,{1,2,2,3,3,3});
  Neo::MeshArrayPropertyProxyT<Neo::utils::Int32> mesh_array_property_proxy{mesh_array_property};
  auto mesh_array_property_values = mesh_array_property.view();
  EXPECT_EQ(mesh_array_property_proxy.arrayPropertyData(),mesh_array_property_values.m_ptr);
  EXPECT_EQ(mesh_array_property.view().size(),mesh_array_property_proxy.arrayPropertyDataSize());
  EXPECT_EQ(mesh_array_property.sizes().size(),mesh_array_property_proxy.arrayPropertyOffsetsSize());
  EXPECT_EQ(mesh_array_property.size(),mesh_array_property_proxy.arrayPropertyIndexSize()); // todo must be ok !!
  auto mesh_array_property_sizes = mesh_array_property.sizes();
  EXPECT_TRUE(std::equal(mesh_array_property_sizes.begin(), mesh_array_property_sizes.end(), mesh_array_property_proxy.arrayPropertyOffsets()));
  auto const mesh_array_property_const_proxy{mesh_array_property_proxy};
  EXPECT_EQ(mesh_array_property_const_proxy.arrayPropertyOffsets(),mesh_array_property_sizes.m_ptr);
  [[maybe_unused]] auto property_values = mesh_array_property.view();
  [[maybe_unused]] auto property_data = mesh_array_property_proxy.arrayPropertyData();
  auto property_indexes = mesh_array_property_proxy.arrayPropertyIndex();
  [[maybe_unused]] auto property_offsets = mesh_array_property_proxy.arrayPropertyOffsets();
  [[maybe_unused]] auto property_index = 0;
  auto value_index = 0;
  auto item_index = 0;
  for (auto item : item_range){
    EXPECT_EQ(value_index,property_indexes[item_index]);
    value_index += mesh_array_property[item].size();
    item_index++;
  }
}

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
