﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TestCompositeSpace.cc                                       (C) 2000-2023 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include <alien/data/Space.h>
#include <alien/kernels/composite/CompositeSpace.h>

TEST(TestCompositeSpace, DefaultConstructor)
{
  const Alien::CompositeKernel::Space s;
  ASSERT_EQ(0, s.size());
  ASSERT_EQ(0, s.subSpaceSize());
}

TEST(TestCompositeSpace, SpaceEquality)
{
  const Alien::CompositeKernel::Space s1;
  ASSERT_TRUE(s1 == s1);
  const Alien::CompositeKernel::Space s2;
  ASSERT_TRUE(s1 == s2);

  // En C++20 et plus, on considére ces comparaisons comme ambigües.
  // TODO : À supprimer lors du passage en C++20.
  #if defined(ARCCORE_CXX_STANDARD) && ARCCORE_CXX_STANDARD == 17
  const Alien::Space s3;
  ASSERT_TRUE(s1 == s3);
  const Alien::Space s4(1);
  ASSERT_FALSE(s1 == s4);
  const Alien::Space s5(0, "Named");
  ASSERT_FALSE(s1 == s5);
  #endif
}

TEST(TestCompositeSpace, SubSpaceResize)
{
  Alien::CompositeKernel::Space s;
  s.resizeSubSpace(3);
  ASSERT_EQ(0, s.size());
  ASSERT_EQ(3, s.subSpaceSize());
  s[0].reset(new Alien::Space(1));
  ASSERT_EQ(1, s.size());
  s[0].reset(new Alien::Space(3));
  ASSERT_EQ(3, s.size());
  s[1].reset(new Alien::Space(4, "Named"));
  ASSERT_EQ(7, s.size());
  s[2].reset(new Alien::Space(3));
  ASSERT_EQ(10, s.size());

  // TODO : À supprimer lors du passage en C++20.
  #if defined(ARCCORE_CXX_STANDARD) && ARCCORE_CXX_STANDARD == 17
  const Alien::Space s1(10);
  ASSERT_TRUE(s1 == s);
  #endif
}

TEST(TestCompositeSpace, SubSpaceMultipleResize)
{
  Alien::CompositeKernel::Space s;
  s.resizeSubSpace(1);
  ASSERT_EQ(0, s.size());
  ASSERT_EQ(1, s.subSpaceSize());
  s[0].reset(new Alien::Space(3));
  ASSERT_EQ(3, s.size());
  s.resizeSubSpace(2);
  ASSERT_EQ(0, s.size());
  ASSERT_EQ(2, s.subSpaceSize());
  s[0].reset(new Alien::Space(3));
  ASSERT_EQ(3, s.size());
  s[1].reset(new Alien::Space(4));
  ASSERT_EQ(7, s.size());
}

TEST(TestCompositeSpace, RValueConstructor)
{
  Alien::CompositeKernel::Space s1;
  s1.resizeSubSpace(1);
  s1[0].reset(new Alien::Space(3));
  auto f = []() -> Alien::CompositeKernel::Space {
    Alien::CompositeKernel::Space s;
    s.resizeSubSpace(1);
    s[0].reset(new Alien::Space(3));
    return s;
  };
  const Alien::CompositeKernel::Space s2 =
  f(); // Attention, const déclenche de optimisations...
  ASSERT_TRUE(s1 == s2);
  Alien::CompositeKernel::Space s3;
  s3 = std::move(s2);
  ASSERT_TRUE(s1 == s3);
}
