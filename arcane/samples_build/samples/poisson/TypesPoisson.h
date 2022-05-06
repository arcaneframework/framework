// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TypesPoisson.h                                              (C) 2000-2022 */
/*                                                                           */
/* Types de Poisson (dans axl)                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef TYPESPOISSON_H
#define TYPESPOISSON_H

#include <arcane/ItemGroup.h>

struct TypesPoisson
{
  enum eBoundaryCondition
  {
    Temperature,
    Unknown
  };
};

#endif  

