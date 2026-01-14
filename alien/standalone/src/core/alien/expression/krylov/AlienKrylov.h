// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/expression/krylov/Iteration.h>
#include <alien/expression/krylov/CG.h>
#include <alien/expression/krylov/BiCGStab.h>
#include <alien/expression/krylov/AMGSolverT.h>
#include <alien/expression/krylov/DiagPreconditioner.h>
#include <alien/expression/krylov/ChebyshevPreconditioner.h>
#include <alien/expression/krylov/NeumannPolyPreconditioner.h>
#include <alien/expression/krylov/ILU0Preconditioner.h>
#include <alien/expression/krylov/FILU0Preconditioner.h>
#include <alien/expression/krylov/AMGPreconditioner.h>
#include <alien/expression/krylov/CxrOperator.h>
#include <alien/expression/krylov/CxrPreconditioner.h>
