// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/core/NodesOfItemReorderer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

namespace Arcane
{
class NodesOfItemReordererTester
{
 public:

  void _doTest(bool expected_reorder, std::initializer_list<Int64> orig_node_list,
               std::initializer_list<Int64> expected_sorted_node_list)
  {
    UniqueArray<Int64> orig_nodes(orig_node_list);
    UniqueArray<Int64> expected_nodes(expected_sorted_node_list);
    UniqueArray<Int64> work_nodes(orig_nodes.size());
    bool is_reorder = NodesOfItemReorderer::_reorderOrder2(orig_nodes, work_nodes);
    ASSERT_EQ(is_reorder, expected_reorder);
    ASSERT_EQ(work_nodes.view(), expected_nodes.view()) << "Orig=" << orig_nodes;
  }
  void doTest()
  {
    _doTest(true, { 13, 12, 11, 14, 7, 6 }, { 11, 12, 13, 7, 14, 6 });
    _doTest(false, { 13, 11, 12, 6, 7, 14 }, { 11, 12, 13, 7, 14, 6 });
    _doTest(true, { 12, 11, 13, 7, 6, 14 }, { 11, 12, 13, 7, 14, 6 });
    _doTest(true, { 11, 13, 12, 6, 14, 7 }, { 11, 12, 13, 7, 14, 6 });
    _doTest(false, { 12, 13, 11, 9, 18, 5 }, { 11, 12, 13, 5, 9, 18 });
#if 0
    UniqueArray<Int64> work_nodes(6);
    UniqueArray<Int64> orig_nodes = { 13, 12, 11, 14, 7, 6 };
    UniqueArray<Int64> expected_nodes = { 11, 12, 13, 7, 14, 6 };
    bool is_reorder = NodesOfItemReorderer::_reorderOrder2(orig_nodes, work_nodes);
    ASSERT_TRUE(is_reorder);
    ASSERT_EQ(work_nodes.view(), expected_nodes.view()) << "Orig=" << orig_nodes;
#endif
  }
};
} // namespace Arcane

TEST(NodesOfItemReorderer, Misc)
{
  NodesOfItemReordererTester tester;
  tester.doTest();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
