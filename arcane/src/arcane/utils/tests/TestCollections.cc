// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/List.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(Collections,Basic)
{
  std::cout << "TEST_Collection Basic\n";

  StringList string_list;
  String str1 = "TotoTiti";
  String str2 = "Tata";
  String str3 = "Hello";
  String str4 = "MyStringToTest";
  
  string_list.add(str1);
  ASSERT_EQ(string_list.count(),1);

  string_list.add(str2);
  ASSERT_EQ(string_list.count(),2);

  string_list.add(str3);
  ASSERT_EQ(string_list.count(),3);

  ASSERT_TRUE(string_list.contains("Tata"));
  ASSERT_FALSE(string_list.contains("NotTata"));
  ASSERT_EQ(string_list[0],str1);
  ASSERT_EQ(string_list[1],"Tata");
  ASSERT_EQ(string_list[2],str3);

  string_list.remove("Tata");
  ASSERT_EQ(string_list.count(),2);
  ASSERT_EQ(string_list[0],str1);
  ASSERT_EQ(string_list[1],str3);

  string_list.clear();
  ASSERT_EQ(string_list.count(),0);

  string_list.add(str4);
  ASSERT_EQ(string_list.count(),1);
  string_list.add(str2);
  ASSERT_EQ(string_list.count(),2);
  string_list.add(str1);
  ASSERT_EQ(string_list.count(),3);

  ASSERT_TRUE(string_list.contains("Tata"));
  ASSERT_FALSE(string_list.contains("NotTata"));
  ASSERT_TRUE(string_list.contains(str2));
  ASSERT_FALSE(string_list.contains(str3));
  ASSERT_TRUE(string_list.contains(str1));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
template class List<String>;
template class ListImplBase<String>;
template class ListImplT<String>;
template class Collection<String>;
template class CollectionImplT<String>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
