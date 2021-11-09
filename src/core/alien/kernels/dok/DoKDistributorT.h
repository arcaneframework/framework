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

#include <alien/utils/Precomp.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/kernels/dok/DoKLocalMatrixT.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename NNZValue>
class DoKDistributorT
{
 public:
  typedef DoKLocalMatrixT<NNZValue> Matrix;

 public:
  DoKDistributorT(const MatrixDistribution& dist)
  : m_dist(dist)
  {}

  virtual ~DoKDistributorT() {}

  Matrix&& operator()(Matrix& src)
  {
    Matrix dst;
    distribute(src, dst);
    return std::move(dst);
  }

  void distribute(Matrix& src, Matrix& dst)
  {
    /* Distribution algorithm:
     * 1 - Compact local data
     * 2 - Compute communication plan
     * 3 - Perform communication
     * 4 - Return new local DoKMatrix
     *
     * Note: this function can almost be used to redistribute data accross
     * different IMessagePassingMng.
     */
    src.compact();

    // Copy values to a send buffer, in case src and dst are the same matrix.
    UniqueArray<NNZValue> snd_values = src.getValues(); // TODO: avoid this copy
    IReverseIndexer* reverse_index = src.getReverseIndexer();
    Arccore::MessagePassing::IMessagePassingMng* pm = m_dist.parallelMng();

    Int32 comm_size = pm->commSize();

    // Prepare communication buffer
    UniqueArray<Int32> snd_rows(snd_values.size());
    UniqueArray<Int32> snd_cols(snd_values.size());

    for (IReverseIndexer::Offset offset = 0; offset < snd_values.size(); ++offset) {
      IReverseIndexer::Index ij = (*reverse_index)[offset];
      snd_rows[offset] = ij.first;
      snd_cols[offset] = ij.second;
    }

    UniqueArray<Int32> snd_offset(comm_size + 1);
    int p = 0;
    Arccore::Int64 p_offset = m_dist.rowOffset(p);
    snd_offset[0] = 0;
    for (IReverseIndexer::Offset i = 0; i < snd_rows.size(); i++) {
      Int32 row_id = snd_rows[i];
      while ((p < comm_size) && (row_id >= p_offset)) {
        snd_offset[p + 1] = i;
        p++;
        p_offset = m_dist.rowOffset(p);
      }
    }
    snd_offset[comm_size] = snd_rows.size();

    UniqueArray<Int32> snd_count(comm_size);
    for (p = 0; p < comm_size; ++p) {
      snd_count[p] = snd_offset[p + 1] - snd_offset[p];
    }

    UniqueArray<Int32> rcv_offset(comm_size + 1);
    UniqueArray<Int32> rcv_count(comm_size);

    pm->allToAll(snd_count, rcv_count, 1);
    rcv_offset[0] = 0;
    for (p = 0; p < comm_size; ++p) {
      rcv_offset[p + 1] = rcv_offset[p] + rcv_count[p];
    }

    UniqueArray<Int32> rcv_rows(rcv_offset[comm_size]);
    UniqueArray<Int32> rcv_cols(rcv_offset[comm_size]);
    UniqueArray<Real> rcv_values(rcv_offset[comm_size]);

    pm->allToAllVariable(
    snd_rows, snd_count, snd_offset, rcv_rows, rcv_count, rcv_offset);
    pm->allToAllVariable(
    snd_cols, snd_count, snd_offset, rcv_cols, rcv_count, rcv_offset);
    pm->allToAllVariable(
    snd_values, snd_count, snd_offset, rcv_values, rcv_count, rcv_offset);

    dst.setMaxNnz(rcv_values.size());

    for (int offset = 0; offset < rcv_values.size(); ++offset) {
      dst.set(rcv_rows[offset], rcv_cols[offset], rcv_values[offset]);
    }
  }

 private:
  const MatrixDistribution& m_dist;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
