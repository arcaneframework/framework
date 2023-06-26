/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include <alien/data/IMatrix.h>

namespace Arccore
{
class ITraceMng;
namespace MessagePassing
{
  class IMessagePassingMng;
}
} // namespace Arccore

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

  template <typename ValueT = Real>
  class MatrixProfilerT
  {
   public:
    explicit MatrixProfilerT(IMatrix& matrix);

    virtual ~MatrixProfilerT();

    void addMatrixEntry(Integer iIndex, Integer jIndex);

    void allocate();

   private:
    IMatrix& m_matrix;

    SimpleCSRMatrix<ValueT>* m_matrix_impl;

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
