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
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneCxx20.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// std::atomic_ref
#include <atomic>
#if __cpp_lib_atomic_ref < 201806L
#error "The compiler does not support C++20 std::atomic_ref"
#endif

// concepts
#include <concepts>
#if __cpp_concepts < 201907L
#error "The compiler does not support C++20 concepts"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
