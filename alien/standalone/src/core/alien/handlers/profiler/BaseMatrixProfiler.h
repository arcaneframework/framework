// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#pragma once

#include <vector>

#include <arccore/message_passing/MessagePassingGlobal.h>
#include <alien/data/IMatrix.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Scalar>
class SimpleCSRMatrix;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Common
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename ValueT = Real, typename MatrixImplT = SimpleCSRMatrix<ValueT>>
  class MatrixProfilerT
  {
   public:
    typedef MatrixImplT MatrixImplType ;
    explicit MatrixProfilerT(IMatrix& matrix);

    virtual ~MatrixProfilerT();

    void addMatrixEntry(Integer iIndex, Integer jIndex);

    void allocate();

   private:
    IMatrix& m_matrix;

    MatrixImplType* m_matrix_impl = nullptr;

    //! @internal data structure for a vector.
    typedef std::vector<Integer> VectorDefinition;

    //! @internal data structure for matrix adjency graph.
    typedef UniqueArray<VectorDefinition> MatrixDefinition;

    //! @internal data structure for matrix values (CSR)
    MatrixDefinition m_def_matrix;

    //! Global matrix informations.
    Integer m_local_offset = 0;
    Integer m_global_size = 0;
    Integer m_local_size = 0;

    bool m_square_matrix = false;
    Integer m_col_local_offset = 0;
    Integer m_col_global_size = 0;
    Integer m_col_local_size = 0;

    Integer m_nproc = 1;
    IMessagePassingMng* m_parallel_mng = nullptr;
    ITraceMng* m_trace = nullptr;

    bool m_allocated = false;

   private:
    void _startTimer() {}
    void computeProfile();
    void _stopTimer() {}
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "MatrixProfilerT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
