// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicUnitTest.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Service basique de test unitaire.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_BASICUNITTEST_H
#define ARCANE_CORE_BASICUNITTEST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IUnitTest.h"
#include "arcane/core/BasicService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Service basique de test unitaire.
 */
class ARCANE_CORE_EXPORT BasicUnitTest
: public BasicService
, public IUnitTest
{
 public:

  explicit BasicUnitTest(const ServiceBuildInfo& sbi);
  ~BasicUnitTest() override;

 public:

  void initializeTest() override;
  void executeTest() override;
  void finalizeTest() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
