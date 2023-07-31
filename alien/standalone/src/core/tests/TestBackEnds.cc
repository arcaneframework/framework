// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TestBackEnds.cc                                             (C) 2000-2023 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include <Environment.h>

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/data/Space.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/kernels/composite/CompositeMultiVectorImpl.h>

TEST(TestBackEnds, Constructor)
{
  Alien::VectorDistribution dist(3, AlienTest::Environment::parallelMng());
  Alien::Space sp(3);
  Alien::MultiVectorImpl impl(std::make_shared<Alien::Space>(sp), dist.clone());
  ASSERT_EQ(nullptr, impl.block());
  ASSERT_EQ(sp, impl.space());
  ASSERT_EQ(dist, impl.distribution());
}

TEST(TestBackEnds, CompositeVector)
{
  Alien::CompositeKernel::MultiVectorImpl impl;
  ASSERT_EQ(nullptr, impl.block());

  // En C++20 et plus, on considére cette comparaison comme ambigüe.
  // TODO : À supprimer lors du passage en C++20.
  #if defined(ARCCORE_CXX_STANDARD) && ARCCORE_CXX_STANDARD == 17
  ASSERT_EQ(Alien::Space(), impl.space());
  #endif

  ASSERT_EQ(Alien::VectorDistribution(), impl.distribution());
}
