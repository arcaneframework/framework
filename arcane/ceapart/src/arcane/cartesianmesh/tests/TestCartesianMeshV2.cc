// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/tests/CartesianMeshV2TestUtils.h"

#include "arcane/cartesianmesh/v2/CartesianTypes.h"
#include "arcane/cartesianmesh/v2/CartesianGrid.h"
#include "arcane/cartesianmesh/v2/CartesianNumbering.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(CartesianMeshV2,Test1)
{
  std::cout << "TEST_CARTESIANMESHV2 Test1\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Effecttue des instantiations explicites pour tester la compilation.
template class Arcane::CartesianMesh::V2::CartesianGrid<Arcane::Int32>;
template class Arcane::CartesianMesh::V2::CartesianGrid<Arcane::Int64>;

template class Arcane::CartesianMesh::V2::CartesianNumbering<Arcane::Int32>;
template class Arcane::CartesianMesh::V2::CartesianNumbering<Arcane::Int64>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
