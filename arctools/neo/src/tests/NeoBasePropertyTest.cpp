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

TEST(NeoTestProperty, test_scalar_property) {
  Neo::ScalarPropertyT<Neo::utils::Int32> scalar_property{ "test_scalar_property", -1 };
  EXPECT_EQ(scalar_property.name(), "test_scalar_property");
  EXPECT_EQ(scalar_property.get(), -1);
  scalar_property.set(42);
  EXPECT_EQ(42, scalar_property.get());
  auto& const_scalar_property = scalar_property;
  EXPECT_EQ(42, const_scalar_property.get());
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestProperty, test_array_property) {
  Neo::ArrayPropertyT<Neo::utils::Int32> array_property{ "test_array_property" };
  EXPECT_EQ(array_property.name(), "test_array_property");
  array_property.reserve(10);
  EXPECT_EQ(array_property.size(), 0);
  array_property.resize(10);
  EXPECT_EQ(array_property.size(), 10);
  for (auto i = 0; i < 10; ++i) {
    array_property[i] = i;
  }
  std::vector<int> ref_values = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  array_property.debugPrint();
  EXPECT_TRUE(std::equal(array_property.begin(), array_property.end(), ref_values.begin()));
  EXPECT_EQ(array_property.back(), 9);
  auto index = 0;
  for (auto& ref_value : ref_values) {
    EXPECT_EQ(ref_value, array_property[index++]);
  }
  array_property.clear();
  for (auto i = 0; i < 10; ++i) {
    array_property.push_back(i);
  }
  EXPECT_EQ(array_property.size(), 10);
  EXPECT_TRUE(std::equal(array_property.begin(), array_property.end(), ref_values.begin()));
  EXPECT_EQ(array_property.back(), 9);
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestProperty, test_array_property_views) {
  Neo::ArrayPropertyT<Neo::utils::Int32> array_property{ "test_array_property_views" };
  array_property.init({ 0, 1, 2, 3, 4 });
  auto array_property_view = array_property.view();
  EXPECT_TRUE(std::equal(array_property_view.begin(), array_property_view.end(), array_property.begin()));
  auto array_property_const_view = array_property.constView();
  EXPECT_TRUE(std::equal(array_property_const_view.begin(), array_property_const_view.end(), array_property.begin()));
  array_property_view = array_property.subView(2, 3);
  std::vector<Neo::utils::Int32> subview_ref_values{ 2, 3, 4 };
  EXPECT_TRUE(std::equal(array_property_view.begin(), array_property_view.end(), subview_ref_values.begin()));
  // check invalid cases
  array_property_view = array_property.subView(10, 2);
  EXPECT_EQ(array_property_view.size(), 0);
  array_property_view = array_property.subView(0, 10);
  EXPECT_TRUE(std::equal(array_property_view.begin(), array_property_view.end(), array_property.begin()));
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestProperty, test_mesh_scalar_property) {
  Neo::MeshScalarPropertyT<Neo::utils::Int32> property{ "test_mesh_scalar_property" };
  EXPECT_EQ(property.name(), "test_mesh_scalar_property");
  std::vector<Neo::utils::Int32> values{ 1, 2, 3 };
  Neo::ItemRange item_range{ Neo::ItemLocalIds{ {}, 0, 3 } };
  EXPECT_TRUE(property.isInitializableFrom(item_range));
  property.init(item_range, { 1, 2, 3 });
  // Cannot init twice : test assertion
#ifndef _MS_REL_ // if constepxr still experiencing problems with MSVC
#ifdef USE_GTEST_DEATH_TEST
  if constexpr (_debug) {
    EXPECT_DEATH(property.init(item_range, values), ".*Property must be empty.*");
  }
#endif
#endif
  EXPECT_EQ(values.size(), property.size());
  auto prop_values = property.values();
  EXPECT_TRUE(std::equal(prop_values.begin(), prop_values.end(), values.begin()));
  std::vector<Neo::utils::Int32> new_values{ 4, 5, 6 };
  Neo::ItemRange new_item_range{ Neo::ItemLocalIds{ {}, 3, 3 } };
  property.append(new_item_range, new_values);
  property.debugPrint();
  EXPECT_EQ(values.size() + new_values.size(), property.size());
  auto all_values{ values };
  std::copy(new_values.begin(), new_values.end(), std::back_inserter(all_values));
  auto property_values = property.values();
  for (std::size_t i = 0; i < all_values.size(); ++i) {
    EXPECT_EQ(property_values[i], all_values[i]);
  }
  // test for range loop
  auto i = 0;
  for (auto val : property) {
    EXPECT_EQ(val, all_values[i++]);
  }
  // test operator[] (4 versions)
  i = 0;
  for (auto item : item_range) {
    EXPECT_EQ(property[item], values[i++]);
  }
  i = 0;
  [[maybe_unused]] const auto& const_property = property;
  for (const auto& item : new_item_range) {
    EXPECT_EQ(property[item], new_values[i++]);
  }
  // check operator[item_lids] on a lids array. Extract lids and values
  std::vector<int> item_indexes{ 0, 3, 4, 5 };
  auto local_ids = item_range.localIds();
  auto lids_new_range = new_item_range.localIds();
  std::copy(lids_new_range.begin(), lids_new_range.end(),
            std::back_inserter(local_ids));
  std::vector<Neo::utils::Int32> extracted_values_ref;
  std::transform(item_indexes.begin(), item_indexes.end(), std::back_inserter(extracted_values_ref),
                 [&all_values](auto index) { return all_values[index]; });
  std::vector<Neo::utils::Int32> extracted_lids;
  std::transform(item_indexes.begin(), item_indexes.end(), std::back_inserter(extracted_lids),
                 [&local_ids](auto index) { return local_ids[index]; });
  auto extracted_values = property[extracted_lids];
  Neo::printer() << "extracted_values " << extracted_values << Neo::endline;
  Neo::printer() << "extracted_values_ref " << extracted_values_ref << Neo::endline;
  EXPECT_TRUE(std::equal(extracted_values.begin(), extracted_values.end(), extracted_values_ref.begin()));
  // todo check throw if lids out of bound
#ifndef _MS_REL_ // if constepxr still experiencing problems with MSVC
#ifdef USE_GTEST_DEATH_TEST
  if constexpr (_debug) {
    EXPECT_DEATH(property[1000], ".*Item local id must be < max local id.*");
  }
  if constexpr (_debug) {
    EXPECT_DEATH(const_property[1000], ".*Item local id must be < max local id.*");
  }
  extracted_lids = { 100, 1000, 1000000 };
  if constexpr (_debug) {
    EXPECT_DEATH(property[extracted_lids], ".*Max input item lid.*");
  }
#endif
#endif
  // Check append with holes, contiguous range
  item_range = { Neo::ItemLocalIds{ {}, 8, 2 } };
  values = { 8, 9 };
  property.append(item_range, values, Neo::utils::NULL_ITEM_LID);
  property.debugPrint();
  extracted_values = property[{ 8, 9 }];
  EXPECT_TRUE(std::equal(values.begin(), values.end(), extracted_values.begin()));
  std::array null_ids = { Neo::utils::NULL_ITEM_LID, Neo::utils::NULL_ITEM_LID };
  auto extracted_null_ids = property[{ 6, 7 }];
  EXPECT_TRUE(std::equal(null_ids.begin(), null_ids.end(), extracted_null_ids.begin()));
  // Check append in empty property contiguous range not starting at 0
  Neo::MeshScalarPropertyT<Neo::utils::Int32> property2{ "test_property2" };
  item_range = { Neo::ItemLocalIds{ {}, 2, 3 } };
  values = { 2, 3, 4 };
  property2.append(item_range, values, Neo::utils::NULL_ITEM_LID);
  property2.debugPrint();
  extracted_values = property2[{ 2, 3, 4 }];
  EXPECT_TRUE(std::equal(values.begin(), values.end(), extracted_values.begin()));
  extracted_null_ids = property2[std::vector<int>{ 0, 1 }];
  EXPECT_TRUE(std::equal(null_ids.begin(), null_ids.end(), extracted_null_ids.begin()));
  // Check append with holes, discontiguous range
  item_range = { Neo::ItemLocalIds{ {
                                    0,
                                    1,
                                    4,
                                    },
                                    0,
                                    0 } };
  values = { 0, 1, 8 };
  property2.append(item_range, values, Neo::utils::NULL_ITEM_LID);
  property2.debugPrint();
  extracted_values = property2[{ 0, 1, 4 }];
  EXPECT_TRUE(std::equal(values.begin(), values.end(), extracted_values.begin()));
  // Check append in empty property discontiguous range
  Neo::MeshScalarPropertyT<Neo::utils::Int32> property3{ "test_property3" };
  item_range = { Neo::ItemLocalIds{ { 1, 3, 5 }, 0, 0 } };
  values = { 1, 3, 5 };
  property3.append(item_range, values, Neo::utils::NULL_ITEM_LID);
  property3.debugPrint();
  extracted_values = property3[{ 1, 3, 5 }];
  EXPECT_TRUE(std::equal(values.begin(), values.end(), extracted_values.begin()));
  // Check append with holes, mixed range
  item_range = { Neo::ItemLocalIds{ { 4 }, 0, 2 } };
  values = { 10, 11, 18 };
  property2.append(item_range, values, Neo::utils::NULL_ITEM_LID);
  property2.debugPrint();
  extracted_values = property2[{ 4, 0, 1 }];
  EXPECT_TRUE(std::equal(values.begin(), values.end(), extracted_values.begin()));
  // Check append in empty property mixed range
  Neo::MeshScalarPropertyT<Neo::utils::Int32> property4{ "test_property3" };
  item_range = { Neo::ItemLocalIds{ { 1, 3, 5 }, 7, 3 } };
  values = { 1, 3, 5, 7, 8, 9 };
  property4.append(item_range, values, Neo::utils::NULL_ITEM_LID);
  property4.debugPrint();
  extracted_values = property4[{ 1, 3, 5, 7, 8, 9 }];
  EXPECT_TRUE(std::equal(values.begin(), values.end(), extracted_values.begin()));
  // Clear properties
  property.clear();
  EXPECT_EQ(property.size(), 0);
  EXPECT_EQ(property.values().size(), 0);
  // Possible to call init again after a clear
  item_range = { Neo::ItemLocalIds{ {}, 0, 3 } };
  values = { 1, 2, 3 };
  property.init(item_range, { 1, 2, 3 });
  EXPECT_EQ(values.size(), property.size());
  values = { 1, 2, 3 };
  prop_values = property.values();
  EXPECT_TRUE(std::equal(prop_values.begin(), prop_values.end(), values.begin()));
}

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
