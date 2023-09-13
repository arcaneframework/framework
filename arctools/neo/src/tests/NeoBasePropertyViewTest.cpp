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


/*-----------------------------------------------------------------------------*/

TEST(NeoTestPropertyView, test_mesh_scalar_property_view) {
  Neo::MeshScalarPropertyT<Neo::utils::Int32> property{ "name" };
  std::vector<Neo::utils::Int32> values{ 1, 2, 3, 10, 100, 1000 };
  Neo::ItemRange item_range{ Neo::ItemLocalIds{ {}, 0, 6 } };
  property.init(item_range, values);
  auto property_view = property.view();
  EXPECT_EQ(property_view.size(), item_range.size());
  std::vector<Neo::utils::Int32> local_ids{ 1, 3, 5 };
  std::vector<Neo::utils::Int32> partial_values{ 2, 10, 1000 };
  auto partial_item_range = Neo::ItemRange{ Neo::ItemLocalIds{ local_ids } };
  auto partial_property_view = property.view(partial_item_range);
  EXPECT_EQ(partial_property_view.size(), partial_item_range.size());
  for (auto i = 0; i < item_range.size(); ++i) {
    std::cout << "prop values at index " << i << " " << property_view[i] << std::endl;
    EXPECT_EQ(property_view[i], values[i]);
  }
  EXPECT_TRUE(property_view.end() == property_view.end());
  auto beg = property_view.begin();
  for (auto i = 0; i < property_view.size(); ++i) {
    ++beg;
  }
  EXPECT_EQ(beg, property_view.end());
  for (auto value_iter = property_view.begin(); value_iter != property_view.end(); ++value_iter) {
    std::cout << " view value " << *value_iter << " " << std::endl;
  }
  auto index = 0;
  for (auto value : property_view) {
    EXPECT_EQ(value, property_view[index++]);
  }
  EXPECT_TRUE(std::equal(property_view.begin(), property_view.end(), values.begin()));
  for (auto i = 0; i < partial_item_range.size(); ++i) {
    std::cout << "prop values at index " << i << " " << partial_property_view[i] << std::endl;
    EXPECT_EQ(partial_property_view[i], partial_values[i]);
  }
  EXPECT_TRUE(std::equal(partial_property_view.begin(), partial_property_view.end(), partial_values.begin()));
  // Change values
  auto new_val = 50;
  property_view[2] = new_val;
  EXPECT_EQ(property[2], new_val);
  partial_property_view[2] = new_val;
  EXPECT_EQ(property[local_ids[2]], new_val);
  // Check out of bound
#ifndef _MS_REL_ // if constepxr still experiencing problems with MSVC
#ifdef USE_GTEST_DEATH_TEST
  if constexpr (_debug) {
    EXPECT_DEATH(property_view[7], ".*Assertion.*");
  }
  if constexpr (_debug) {
    EXPECT_DEATH(partial_property_view[3], ".*Error, exceeds property view size.*");
  }
#endif
#endif
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestPropertyView, test_mesh_scalar_property_const_view) {
  Neo::MeshScalarPropertyT<Neo::utils::Int32> property{ "name" };
  std::vector<Neo::utils::Int32> values{ 1, 2, 3, 10, 100, 1000 };
  Neo::ItemRange item_range{ Neo::ItemLocalIds{ {}, 0, 6 } };
  property.init(item_range, values);
  auto property_const_view = property.constView();
  EXPECT_EQ(property_const_view.size(), item_range.size());
  auto partial_item_range = Neo::ItemRange{ Neo::ItemLocalIds{ { 1, 3, 5 } } };
  std::vector<Neo::utils::Int32> partial_values{ 2, 10, 1000 };
  auto partial_property_const_view = property.constView(partial_item_range);
  EXPECT_EQ(partial_property_const_view.size(), partial_item_range.size());
  for (auto i = 0; i < item_range.size(); ++i) {
    std::cout << "prop values at index " << i << " " << property_const_view[i] << std::endl;
    EXPECT_EQ(property_const_view[i], values[i]);
  }
  for (auto i = 0; i < partial_item_range.size(); ++i) {
    std::cout << "prop values at index " << i << " " << partial_property_const_view[i] << std::endl;
    EXPECT_EQ(partial_property_const_view[i], partial_values[i]);
  }
#ifndef _MS_REL_ // if constepxr still experiencing problems with MSVC
#ifdef USE_GTEST_DEATH_TEST
  if constexpr (_debug) {
    EXPECT_DEATH(property_const_view[7], ".*Assertion.*");
  }
  if constexpr (_debug) {
    EXPECT_DEATH(partial_property_const_view[3], ".*Error, exceeds property view size.*");
  }
#endif
#endif
  // test const iterator
  EXPECT_TRUE(property_const_view.end() == property_const_view.end());
  auto beg = property_const_view.begin();
  for (auto i = 0; i < property_const_view.size(); ++i) {
    ++beg;
  }
  EXPECT_EQ(beg, property_const_view.end());
  for (auto value_iter = property_const_view.begin(); value_iter != property_const_view.end(); ++value_iter) {
    std::cout << " view value " << *value_iter << " " << std::endl;
  }
  auto index = 0;
  for (auto value : property_const_view) {
    EXPECT_EQ(value, property_const_view[index++]);
  }
  EXPECT_TRUE(std::equal(property_const_view.begin(), property_const_view.end(), values.begin()));
  EXPECT_TRUE(std::equal(partial_property_const_view.begin(), partial_property_const_view.end(), partial_values.begin()));
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestPropertyView, test_mesh_array_property_view) {
  auto mesh_array_property = Neo::MeshArrayPropertyT<Neo::utils::Int32>{ "test_mesh_array_property_view" };
  // add elements: 5 items with one value
  Neo::ItemRange item_range{ Neo::ItemLocalIds{ {}, 0, 5 } };
  std::vector<int> sizes{ 1, 2, 3, 4, 5 };
  mesh_array_property.resize(sizes, true);
  auto i = 0;
  for (auto item : item_range) {
    auto item_values = mesh_array_property[item];
    std::fill(item_values.begin(), item_values.end(), i);
    i++;
  }
  // create a view on the 3 first elements
  std::vector<Neo::utils::Int32> sub_range_ref_values{ 0, 1, 1 };
  Neo::ItemRange item_sub_range{ Neo::ItemLocalIds{ {}, 0, 2 } };
  auto mesh_array_property_view = mesh_array_property.view(item_sub_range);
  // check values
  auto item_index = 0;
  auto value_index = 0;
  for (auto item : item_sub_range) {
    auto item_array = mesh_array_property_view[item];
    EXPECT_EQ(item_array.size(), sizes[item_index]);
    item_index++;
    for (auto item_value : item_array) {
      EXPECT_EQ(item_value, sub_range_ref_values[value_index++]);
    }
  }

  // check assert (debug only)
#ifndef _MS_REL_ // if constepxr still experiencing problems with MSVC
#ifdef USE_GTEST_DEATH_TEST
  if constexpr (_debug) {
    EXPECT_DEATH(mesh_array_property_view[Neo::utils::NULL_ITEM_LID], ".*index must be >0*");
    EXPECT_DEATH(mesh_array_property_view[mesh_array_property_view.size()], ".*Error, exceeds property view*");
  }
#endif
#endif
}

/*-----------------------------------------------------------------------------*/

TEST(NeoTestPropertyView, test_mesh_array_property_const_view) {
  auto mesh_array_property = Neo::MeshArrayPropertyT<Neo::utils::Int32>{ "test_mesh_array_property_view" };
  // add elements: 5 items with one value
  Neo::ItemRange item_range{ Neo::ItemLocalIds{ {}, 0, 5 } };
  mesh_array_property.resize({ 1, 2, 3, 4, 5 }, true);
  auto i = 0;
  for (auto item : item_range) {
    auto item_values = mesh_array_property[item];
    std::fill(item_values.begin(), item_values.end(), i);
    i++;
  }
  // create a view on the 3 first elements
  std::vector<Neo::utils::Int32> sub_range_ref_values{ 0, 1, 1 };
  Neo::ItemRange item_sub_range{ Neo::ItemLocalIds{ {}, 0, 2 } };
  auto mesh_array_property_view = mesh_array_property.constView(item_sub_range);
  // check values
  auto index = 0;
  for (auto item : item_sub_range) {
    auto item_array = mesh_array_property_view[item];
    for (auto item_value : item_array) {
      EXPECT_EQ(item_value, sub_range_ref_values[index++]);
    }
  }

  // check assert (debug only)
#ifndef _MS_REL_ // if constepxr still experiencing problems with MSVC
#ifdef USE_GTEST_DEATH_TEST
  if constexpr (_debug) {
    EXPECT_DEATH(mesh_array_property_view[Neo::utils::NULL_ITEM_LID], ".*index must be >0*");
    EXPECT_DEATH(mesh_array_property_view[mesh_array_property_view.size()], ".*Error, exceeds property view*");
  }
#endif
#endif
}
/*-----------------------------------------------------------------------------*/

TEST(NeoTestPropertyView, test_property_iterator) {
  std::vector data{ 1, 2, 3, 4, 5, 6, 7 };
  std::vector indexes{ 0, 3, 6 };
  Neo::PropertyViewIterator<int> property_view_iterator{ indexes, indexes.begin(), data.data() };
  // right operator
  for (auto index : indexes) {
    EXPECT_EQ(*property_view_iterator, data[index]);
    property_view_iterator++;
  }
  --property_view_iterator;
  for (auto rindex_iterator = indexes.rbegin(); rindex_iterator != indexes.rend(); ++rindex_iterator) {
    EXPECT_EQ(*property_view_iterator, data[*rindex_iterator]);
    std::cout << *property_view_iterator << " ";
    property_view_iterator--;
  }
  // left operator
  for (auto index : indexes) {
    EXPECT_EQ(*property_view_iterator, data[index]);
    ++property_view_iterator;
  }
  property_view_iterator--;
  for (auto rindex_iterator = indexes.rbegin(); rindex_iterator != indexes.rend(); ++rindex_iterator) {
    EXPECT_EQ(*property_view_iterator, data[*rindex_iterator]);
    --property_view_iterator;
  }
  EXPECT_EQ(*(property_view_iterator + 2), data[indexes[2]]);
  EXPECT_EQ(*(property_view_iterator - 2), data[indexes[0]]);
  EXPECT_EQ(*(property_view_iterator += 2), data[indexes[2]]);
  EXPECT_EQ(*(property_view_iterator -= 2), data[indexes[0]]);

  auto property_view_iterator2{ property_view_iterator };
  EXPECT_TRUE(property_view_iterator == property_view_iterator2);
  EXPECT_FALSE(property_view_iterator != property_view_iterator2);

  // check operator->
  std::vector<std::string> data_string{ "hello", "world", "!" };
  indexes = { 0, 1 };
  Neo::PropertyViewIterator<std::string> property_view_iterator3{ indexes, indexes.begin(), data_string.data() };
  EXPECT_EQ(property_view_iterator3->size(), data_string[0].size());
}

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
