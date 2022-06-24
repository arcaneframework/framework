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

#include <alien/ref/import_export/MatrixMarketSystemWriter.h>

#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

#include <alien/core/impl/MultiVectorImpl.h>

#include <alien/ref/data/block/BlockMatrix.h>
#include <alien/ref/data/block/BlockVector.h>
#include <alien/ref/data/scalar/Matrix.h>
#include <alien/ref/data/scalar/Vector.h>

namespace Alien
{

MatrixMarketSystemWriter::MatrixMarketSystemWriter(std::string const& filename,
                                                   IMessagePassingMng* parallel_mng)
: m_filename(filename)
, m_parallel_mng(parallel_mng)
{
  m_rank = 0;
  m_nproc = 1;
  if (m_parallel_mng) {
    m_rank = m_parallel_mng->commRank();
    m_nproc = m_parallel_mng->commSize();
    std::stringstream suf;
    suf << "-R" << m_rank << "P" << m_nproc;
    m_filename = filename + suf.str();
  }
  else {
    m_filename = filename;
  }
}

MatrixMarketSystemWriter::~MatrixMarketSystemWriter() {}

void MatrixMarketSystemWriter::dump(Matrix const& A, std::string const& description)
{
  const SimpleCSRMatrix<Real>& csr = A.impl()->get<BackEnd::tag::simplecsr>();
  const SimpleCSRMatrix<Real>::ProfileType& profile = csr.getProfile();
  int nrows = profile.getNRows();
  int nnz = profile.getNnz();

  const int* cols = profile.cols();
  const int* kcol = profile.kcol();
  const double* values = csr.getAddressData();
  if (m_nproc == 1) {
    std::ofstream fout(m_filename);
    fout << "%%MatrixMarket matrix coordinate real general" << std::endl;
    fout << "%" << std::endl;
    fout << "% " << description << std::endl;
    fout << "%" << std::endl;
    fout << "% #rows #cols #nonzeros" << std::endl;
    fout << "% #vertices #hyperedges #pins" << std::endl;
    fout << nrows << " " << nrows << " " << nnz << std::endl;
    for (int irow = 0; irow < nrows; ++irow) {
      for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {
        fout << irow + 1 << " " << cols[k] + 1 << " " << values[k] << std::endl;
      }
    }
  }
  else {
    int global_nrows = Arccore::MessagePassing::mpAllReduce(m_parallel_mng,
                                                            Arccore::MessagePassing::ReduceSum,
                                                            nrows);

    int global_nnz = Arccore::MessagePassing::mpAllReduce(m_parallel_mng,
                                                          Arccore::MessagePassing::ReduceSum,
                                                          nnz);
    if (m_rank == 0) {
      std::ofstream fout(m_filename);
      fout << "%%MatrixMarket matrix coordinate real general" << std::endl;
      fout << "%" << std::endl;
      fout << "% " << description << std::endl;
      fout << "%" << std::endl;
      fout << "% #rows #cols #nonzeros" << std::endl;
      fout << "% #vertices #hyperedges #pins" << std::endl;
      fout << global_nrows << " " << global_nrows << " " << global_nnz << std::endl;
      for (int irow = 0; irow < nrows; ++irow) {
        for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {
          fout << irow + 1 << " " << cols[k] + 1 << " " << values[k] << std::endl;
        }
      }
      for (int ip = 1; ip < m_nproc; ++ip) {
        Integer local_nnz = 0;
        Arccore::MessagePassing::mpReceive(m_parallel_mng, ArrayView<Integer>(1, &local_nnz), ip);
        if (local_nnz > 0) {
          UniqueArray<Integer> indexes(2 * local_nnz);
          UniqueArray<Real> local_values(2 * local_nnz);
          Arccore::MessagePassing::mpReceive(m_parallel_mng, indexes, ip);
          Arccore::MessagePassing::mpReceive(m_parallel_mng, local_values, ip);
          for (int k = 0; k < local_nnz; ++k) {
            fout << indexes[2 * k] << " " << indexes[2 * k + 1] << " " << local_values[k] << std::endl;
          }
        }
      }
    }
    else {
      Arccore::MessagePassing::mpSend(m_parallel_mng, ArrayView<Integer>(1, &nnz), 0);
      UniqueArray<Integer> indexes(2 * nnz);
      UniqueArray<Real> local_values(nnz);
      Integer domain_offset = csr.distribution().rowOffset();
      for (int irow = 0; irow < nrows; ++irow) {
        for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {
          indexes[2 * k] = domain_offset + irow + 1;
          indexes[2 * k + 1] = cols[k] + 1;
          local_values[k] = values[k];
        }
      }
      Arccore::MessagePassing::mpSend(m_parallel_mng, indexes, 0);
      Arccore::MessagePassing::mpSend(m_parallel_mng, local_values, 0);
    }
  }
}

void MatrixMarketSystemWriter::dump(Vector const& rhs, std::string const& description)
{
  const SimpleCSRVector<Real>& v = rhs.impl()->get<BackEnd::tag::simplecsr>();
  auto local_size = v.distribution().localSize();
  const double* values = v.getAddressData();
  if (m_nproc == 1) {
    std::ofstream fout(m_filename);
    fout << "%%MatrixMarket matrix array real general" << std::endl;
    fout << "%" << std::endl;
    fout << "% " << description << std::endl;
    fout << "%" << std::endl;
    fout << local_size << " 1" << std::endl;
    for (int irow = 0; irow < local_size; ++irow) {
      fout << irow + 1 << " " << values[irow] << std::endl;
    }
  }
  else {
    int global_size = Arccore::MessagePassing::mpAllReduce(m_parallel_mng,
                                                           Arccore::MessagePassing::ReduceSum,
                                                           local_size);
    if (m_rank == 0) {
      std::ofstream fout(m_filename);
      fout << "%%MatrixMarket matrix array real general" << std::endl;
      fout << "%" << std::endl;
      fout << "% " << description << std::endl;
      fout << "%" << std::endl;
      fout << global_size << " 1" << std::endl;
      for (int irow = 0; irow < local_size; ++irow) {
        fout << irow + 1 << " " << values[irow] << std::endl;
      }
      for (int ip = 1; ip < m_nproc; ++ip) {
        Integer local_nrows = 0;
        Integer domain_offset = v.distribution().offset(ip);
        Arccore::MessagePassing::mpReceive(m_parallel_mng, ArrayView<Integer>(1, &local_nrows), ip);
        if (local_nrows > 0) {
          UniqueArray<Real> local_values(local_nrows);
          Arccore::MessagePassing::mpReceive(m_parallel_mng, local_values, ip);
          for (int k = 0; k < local_nrows; ++k) {
            fout << domain_offset + k + 1 << " " << local_values[k] << std::endl;
          }
        }
      }
    }
    else {
      Arccore::MessagePassing::mpSend(m_parallel_mng, ConstArrayView<Integer>(1, &local_size), 0);
      Arccore::MessagePassing::mpSend(m_parallel_mng, ConstArrayView<Real>(local_size, values), 0);
    }
  }
}

} /* namespace Alien */
