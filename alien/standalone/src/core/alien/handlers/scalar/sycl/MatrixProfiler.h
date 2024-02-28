// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#pragma once

#include <alien/handlers/profiler/BaseMatrixProfiler.h>


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  template <typename Scalar>
  class HCSRMatrix;


  namespace SYCL
  {
    class MatrixProfiler
        : public Common::MatrixProfilerT<Arccore::Real,HCSRMatrix<Arccore::Real>>
    {
     public:
      MatrixProfiler(IMatrix& matrix)
      : Common::MatrixProfilerT<Arccore::Real,HCSRMatrix<Arccore::Real>>(matrix)
      {}

    };
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
