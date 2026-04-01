// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/utils/Precomp.h>

#ifndef USE_SYCL_USM
//#define USE_SYCL_USM
#endif
#ifndef USE_HIPSYCL
#ifndef USE_ACPPSYCL
#define USE_ONEAPI
#endif
#endif
#ifdef USE_ACPPSYCL
#ifndef USE_SYCL2020
#define USE_SYCL2020
#endif
#endif

// Sélection à la compilation selon la cible
#if defined(__HIP_PLATFORM_AMD__) || defined(__AMDGCN__)
  static constexpr int PKSIZE      = 1024; // wavefront MI300 should be 64
  static constexpr int WG_SIZE     = 256;  // 4 WF/bloc
#else
  static constexpr int PKSIZE      = 1024;   // warp H100 should be 32
  static constexpr int WG_SIZE     = 256;  // 8 warps/bloc
#endif

// Grid dynamique (remplace m_total_threads figé)
static constexpr int TARGET_WAVES = 4;
