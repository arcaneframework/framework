// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "alien/utils/Precomp.h"
#include "alien/core/backend/LinearAlgebra.h"

#include "alien/kernels/mcg/MCGBackEnd.h"
#include "alien/kernels/mcg/algebra/MCGInternalLinearAlgebra.h"

namespace Alien {
using MCGLinearAlgebra = MCGInternalLinearAlgebra;
}
