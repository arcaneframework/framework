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
  // Constantes tuned pour MI300 (gfx942)
  // - CU count  : 228 CUs
  // - Wavefront : 64 threads (RDNA/CDNA)
  // - LDS/CU    : 64 KB  → 1024 threads × 8 B (double) = 8 KB/workgroup, safe

  static constexpr int PKSIZE      = 1024; // wavefront MI300 should be 64
  //static constexpr int WG_SIZE     = 256;  // 4 WF/bloc
  static constexpr int TARGET_WAVES = 4;

  static constexpr int WG_SIZE       = 256;   // 16 wavefronts/WG
  static constexpr int ITEMS_PER_WI  = 8;    // unroll : chaque WI traite 8 éléments


#else
  static constexpr int PKSIZE       = 1024;   // warp H100 should be 32
  //static constexpr int WG_SIZE      = 256;    // 8 warps/bloc
  static constexpr int WG_SIZE      = 512;   // 16 warps/bloc
  static constexpr int ITEMS_PER_WI = 8;     // unroll ILP
  static constexpr int WARP_SIZE    = 32;    // natif CUDA/H100
  static constexpr int TARGET_WAVES = 4;     // waves/SM pour masquer latence
#endif

// Grid dynamique (remplace m_total_threads figé)
