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

#include <alien/data/IMatrix.h>
#include <alien/utils/Precomp.h>

#include <arccore/collections/Array2.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Scalar>
class SimpleCSRMatrix;

class VBlock;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct ProfiledVBlockMatrixBuilderOptions
{
  enum ResetFlag
  {
    eKeepValues,
    eResetValues
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Common
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class ALIEN_EXPORT ProfiledVBlockMatrixBuilder
  {
   public:
    class MatrixElement
    {
     public:
      MatrixElement(const Integer iIndex, const Integer jIndex,
                    ProfiledVBlockMatrixBuilder& parent)
      : m_iIndex(iIndex)
      , m_jIndex(jIndex)
      , m_parent(parent)
      {}

      inline void operator+=(ConstArray2View<Real> value)
      {
        m_parent.addData(m_iIndex, m_jIndex, value);
      }

      inline void operator-=(ConstArray2View<Real> value)
      {
        m_parent.addData(m_iIndex, m_jIndex, -1., value);
      }

      inline void operator=(ConstArray2View<Real> value)
      {
        m_parent.setData(m_iIndex, m_jIndex, value);
      }

     private:
      const Integer m_iIndex;
      const Integer m_jIndex;
      ProfiledVBlockMatrixBuilder& m_parent;
    };

   public:
    using ResetFlag = ProfiledVBlockMatrixBuilderOptions::ResetFlag;

    ProfiledVBlockMatrixBuilder(IMatrix& matrix, const ResetFlag reset_values);

    virtual ~ProfiledVBlockMatrixBuilder();

   private:
    ProfiledVBlockMatrixBuilder(const ProfiledVBlockMatrixBuilder&);

   public:
    inline MatrixElement operator()(const Integer iIndex, const Integer jIndex)
    {
      return MatrixElement(iIndex, jIndex, *this);
    }

    void addData(const Integer iIndex, const Integer jIndex, ConstArray2View<Real> value);

    void addData(const Integer iIndex, const Integer jIndex, const Real factor,
                 ConstArray2View<Real> value);

    void setData(const Integer iIndex, const Integer jIndex, ConstArray2View<Real> value);

    void finalize();

   private:
    bool isLocal(Integer jIndex)
    {
      return (jIndex >= m_local_offset) && (jIndex < m_next_offset);
    }

    IMatrix& m_matrix;
    SimpleCSRMatrix<Real>* m_matrix_impl = nullptr;
    const VBlock* m_vblock = nullptr;

    Integer m_local_offset;
    Integer m_next_offset;
    Integer m_local_size;
    ConstArrayView<Integer> m_row_starts;
    ConstArrayView<Integer> m_cols;
    ConstArrayView<Integer> m_cols_offset;
    ConstArrayView<Integer> m_block_cols;
    ConstArrayView<Integer> m_local_row_size;
    ArrayView<Real> m_values;
    bool m_finalized;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
