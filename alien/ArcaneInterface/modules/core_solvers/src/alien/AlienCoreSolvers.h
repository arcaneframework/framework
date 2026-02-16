// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/AlienCoreSolversPrecomp.h>
#include <alien/kernels/simple_csr/linear_solver/AlienCoreLinearSolver.h>
#include <alien/kernels/common/linear_solver/arcane/AlienLinearSolver.h>

#ifdef ALIEN_USE_SYCL
#include <alien/AlienCoreSYCLSolversPrecomp.h>
#include <alien/kernels/sycl/linear_solver/AlienCoreSYCLLinearSolver.h>
#include <alien/kernels/common/linear_solver/arcane/AlienSYCLLinearSolver.h>
#endif

