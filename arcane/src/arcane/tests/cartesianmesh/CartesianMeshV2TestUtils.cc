// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshTestUtils.cc                                   (C) 2000-2021 */
/*                                                                           */
/* Fonctions utilitaires pour les tests de 'CartesianMesh'.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/cartesianmesh/CartesianMeshV2TestUtils.h"

#include "arcane/cartesianmesh/v2/CartesianTypes.h"
#include "arcane/cartesianmesh/v2/CartesianGrid.h"
#include "arcane/cartesianmesh/v2/CartesianNumbering.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshV2TestUtils::
CartesianMeshV2TestUtils(ICartesianMesh* cm)
: TraceAccessor(cm->traceMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshV2TestUtils::
~CartesianMeshV2TestUtils()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshV2TestUtils::
testAll()
{
  // Ajouter les tests
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
