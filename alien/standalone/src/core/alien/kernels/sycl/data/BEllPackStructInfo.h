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

#include <algorithm>
#include <alien/kernels/sycl/SYCLPrecomp.h>

/*---------------------------------------------------------------------------*/

namespace Alien
{
namespace SYCLInternal
{
  template <int BlockSize, typename IndexT>
  struct StructInfoInternal;

}

class BaseBEllPackStructInfo
{
 public:
  BaseBEllPackStructInfo(std::size_t nrows,
                         std::size_t nnz)
  : m_nrows(nrows)
  , m_nnz(nnz)
  {}

  std::size_t getNRows() const { return m_nrows; }

  std::size_t getNnz() const { return m_nnz; }

  Arccore::Int64 timestamp() const { return m_timestamp; }

  void setTimestamp(Arccore::Int64 value) { m_timestamp = value; }

 protected:
  // clang-format off
  std::size_t m_nrows       = 0 ;
  std::size_t m_nnz         = 0 ;
  Arccore::Int64 m_timestamp = -1;
  // clang-format on
};
/*---------------------------------------------------------------------------*/

template <int BlockSize, typename IndexT = int>
class ALIEN_EXPORT BEllPackStructInfo
: public BaseBEllPackStructInfo
{
 public:
  // clang-format off
  typedef IndexT                                             index_type ;
  static const int                                           block_size = BlockSize ;

  typedef SYCLInternal::StructInfoInternal<BlockSize,IndexT> InternalType ;
  typedef IndexT                                             IndexType ;
  // clang-format on

  static std::size_t nbBlocks(std::size_t nrows)
  {
    return (nrows + block_size - 1) / block_size;
  }

  static std::size_t roundUp(std::size_t nrows)
  {
    return nbBlocks(nrows) * block_size;
  }

  static void computeBlockRowOffset(std::vector<int>& block_row_offset,
                                    std::size_t nrows,
                                    int const* kcol);

  BEllPackStructInfo(std::size_t nrows,
                     int const* kcol,
                     int const* cols,
                     int const* h_block_row_offset,
                     int const* h_local_row_size);

  const BaseBEllPackStructInfo& base() const
  {
    return *this;
  }

  InternalType const* internal() const
  {
    return m_internal;
  }

  std::size_t getBlockNnz() const { return m_block_nnz; }

  Arccore::ConstArrayView<Integer> getRowOffset() const
  {
    return Arccore::ConstArrayView<Integer>((Integer)m_nrows + 1, kcol());
  }

  IndexType const* kcol() const;

  IndexType const* cols() const;

  IndexType const* dcol() const;

  int const* localRowSize() const
  {
    return m_h_local_row_size;
  }

 protected:
  // clang-format off
  std::size_t m_block_nrows         = 0 ;
  std::size_t m_block_nnz           = 0 ;

  InternalType*  m_internal         = nullptr;
  int const*     m_h_local_row_size = nullptr ;
  // clang-format on
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
