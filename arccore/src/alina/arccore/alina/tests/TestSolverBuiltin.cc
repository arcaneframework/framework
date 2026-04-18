// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/accelerator/internal/Initializer.h"

#include "TestSolverCommon.h"

void _doTestSolverBuiltinDefault(bool use_accelerator, Int32 max_allowed_thread)
{
  Accelerator::Initializer x(use_accelerator, max_allowed_thread);
  test_backend< Alina::BuiltinBackend<double> >();
}

void _doTestSolverBuiltinInt32Int64(bool use_accelerator, Int32 max_allowed_thread)
{
  Accelerator::Initializer x(use_accelerator, max_allowed_thread);
  test_backend< Alina::BuiltinBackend<double, Int32, Int64> >();
}

void _doTestSolverBuiltinInt32Int32(bool use_accelerator, Int32 max_allowed_thread)
{
  Accelerator::Initializer x(use_accelerator, max_allowed_thread);
  test_backend< Alina::BuiltinBackend<double, Int32, Int32> >();
}

void _doTestSolverBuiltinUInt32SizeT(bool use_accelerator, Int32 max_allowed_thread)
{
  Accelerator::Initializer x(use_accelerator, max_allowed_thread);
  test_backend< Alina::BuiltinBackend<double, uint32_t, size_t> >();
}

ARCCORE_ALINA_TEST_DO_TEST_ACCELERATOR(alina_test_solvers, test_builtin_backend_default, _doTestSolverBuiltinDefault);
ARCCORE_ALINA_TEST_DO_TEST_ACCELERATOR(alina_test_solvers, test_builtin_backend_int32_int64, _doTestSolverBuiltinInt32Int64);
ARCCORE_ALINA_TEST_DO_TEST_ACCELERATOR(alina_test_solvers, test_builtin_backend_int32_int32, _doTestSolverBuiltinInt32Int32);
ARCCORE_ALINA_TEST_DO_TEST_ACCELERATOR(alina_test_solvers, test_builtin_backend_uint32_sizet, _doTestSolverBuiltinUInt32SizeT);
