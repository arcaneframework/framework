// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TestCommon.h                                                (C) 2000-2026 */
/*                                                                           */
/* Déclarations communes pour les tests.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_TESTS_TESTCOMMON_H
#define ARCCORE_ACCELERATOR_TESTS_TESTCOMMON_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Les macros suivantes permettent de définir des tests en fonctions
 * du support accélérateur disponible.
 */
#if defined(ARCCORE_HAS_CUDA)
#define ARCCORE_TEST_DO_TEST_CUDA(name1, name2, func) \
  TEST(name1, name2##_cuda) \
  { \
    func(true, 0); \
  }
#else
#define ARCCORE_TEST_DO_TEST_CUDA(name1, name2, func)
#endif

#if defined(ARCCORE_HAS_HIP)
#define ARCCORE_TEST_DO_TEST_HIP(name1, name2, func) \
  TEST(name1, name2##_hip) \
  { \
    func(true, 0); \
  }
#else
#define ARCCORE_TEST_DO_TEST_HIP(name1, name2, func)
#endif

#if defined(ARCCORE_HAS_SYCL)
#define ARCCORE_TEST_DO_TEST_SYCL(name1, name2, func) \
  TEST(name1, name2##_sycl) \
  { \
    func(true, 0); \
  }
#else
#define ARCCORE_TEST_DO_TEST_SYCL(name1, name2, func)
#endif

#define ARCCORE_TEST_DO_TEST_TASK(name1, name2, func) \
  TEST(name1, name2##_task4) \
  { \
    func(false, 4); \
  }

#define ARCCORE_TEST_DO_TEST_SEQUENTIAL(name1, name2, func) \
  TEST(name1, name2) \
  { \
    func(false, 0); \
  }

//! Macro pour définir les tests en fonction de l'accélérateur
#define ARCCORE_TEST_DO_TEST_ACCELERATOR(name1, name2, func) \
  ARCCORE_TEST_DO_TEST_CUDA(name1, name2, func); \
  ARCCORE_TEST_DO_TEST_HIP(name1, name2, func); \
  ARCCORE_TEST_DO_TEST_SYCL(name1, name2, func); \
  ARCCORE_TEST_DO_TEST_TASK(name1, name2, func); \
  ARCCORE_TEST_DO_TEST_SEQUENTIAL(name1, name2, func);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
