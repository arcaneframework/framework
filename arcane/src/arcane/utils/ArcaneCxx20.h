// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCxx20.h                                               (C) 2000-2023 */
/*                                                                           */
/* Vérification des fonctionnalités minimale requises du C++20.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARCANECXX20_H
#define ARCANE_UTILS_ARCANECXX20_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// std::atomic_ref
#if __cpp_lib_atomic_ref < 201806L
#error "The compiler does not support C++20 std::atomic_ref"
#endif

// concepts
#if __cpp_concepts < 201907L
#error "The compiler does not support C++20 concepts"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
