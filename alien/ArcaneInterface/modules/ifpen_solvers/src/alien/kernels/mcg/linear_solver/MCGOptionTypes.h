// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once

struct MCGOptionTypes
{
  enum eKernelType
  {
    CPU_CBLAS_BCSR,
    CPU_AVX_BCSR,
    CPU_AVX2_BCSP,
    CPU_AVX512_BCSP,
    GPU_CUBLAS_BELL,
    GPU_CUBLAS_BCSP,
    GPU_CUBLAS_BCSR,
  };
};
