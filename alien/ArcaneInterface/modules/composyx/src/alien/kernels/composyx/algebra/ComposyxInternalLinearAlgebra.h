// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#include <alien/AlienComposyxPrecomp.h>

#include <alien/kernels/composyx/ComposyxBackEnd.h>
#include <alien/core/backend/IInternalLinearAlgebraT.h>

#include <alien/expression/solver/ILinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRInternalLinearAlgebra.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef SimpleCSRInternalLinearAlgebra ComposyxSolverInternalLinearAlgebra;

//typedef AlgebraTraits<BackEnd::tag::composyx>::matrix_type CSRMatrix;
//typedef AlgebraTraits<BackEnd::tag::composyx>::vector_type CSRVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
