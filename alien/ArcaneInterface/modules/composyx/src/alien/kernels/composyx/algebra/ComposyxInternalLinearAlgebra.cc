// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "ComposyxInternalLinearAlgebra.h"

#include <alien/kernels/composyx/ComposyxBackEnd.h>

#include <alien/core/backend/LinearAlgebraT.h>

#include <alien/data/Space.h>

#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRInternalLinearAlgebra.h>

#include <arccore/base/NotImplementedException.h>
//#include <alien/kernels/composyx/data_structure/ComposyxMatrix.h>
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

//template class ALIEN_COMPOSYX_EXPORT LinearAlgebra<BackEnd::tag::composyx>;
// template class ALIEN_COMPOSYX_EXPORT
// LinearAlgebra<BackEnd::tag::composyx,BackEnd::tag::simplecsr> ;

/*---------------------------------------------------------------------------*/
IInternalLinearAlgebra<SimpleCSRMatrix<Real>, SimpleCSRVector<Real>>*
ComposyxSolverInternalLinearAlgebraFactory()
{
  return new ComposyxSolverInternalLinearAlgebra();
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
