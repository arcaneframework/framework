// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshTestUtils.cc                                   (C) 2000-2026 */
/*                                                                           */
/* Fonctions utilitaires pour les tests de 'CartesianMesh' V2.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CEA_TESTS_CARTESIANMESHV2TESTUTILS_H
#define ARCANE_CEA_TESTS_CARTESIANMESHV2TESTUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ICartesianMesh;
}

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe utilitaire pour tester les variantes de 'CartesianMesh' V2
 */
class CartesianMeshV2TestUtils
: public TraceAccessor
{
 public:

  explicit CartesianMeshV2TestUtils(ICartesianMesh* cm);
  ~CartesianMeshV2TestUtils();

 public:

  void testAll();

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

