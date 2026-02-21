// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#pragma once

#include <vector>

#include <algorithm>
#include <alien/kernels/sycl/SYCLPrecomp.h>

/*---------------------------------------------------------------------------*/

namespace Alien
{
namespace SYCLInternal
{
  template <int EllPackSize, typename IndexT>
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

  virtual ~BaseBEllPackStructInfo() {}

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

template <int EllPackSize, typename IndexT = int>
class ALIEN_EXPORT BEllPackStructInfo
: public BaseBEllPackStructInfo
{
 public:
  // clang-format off
  using index_type              = IndexT;
  using IndexType               = IndexT;
  using InternalType            = SYCLInternal::StructInfoInternal<EllPackSize,IndexT>;
  static const int ellpack_size = EllPackSize ;

  // clang-format on

  static std::size_t nbBlocks(std::size_t nrows)
  {
    return (nrows + ellpack_size - 1) / ellpack_size;
  }

  static std::size_t roundUp(std::size_t nrows)
  {
    return nbBlocks(nrows) * ellpack_size;
  }

  static void computeBlockRowOffset(std::vector<int>& block_row_offset,
                                    std::size_t nrows,
                                    int const* kcol);

  BEllPackStructInfo(std::size_t nrows,
                     int const* kcol,
                     int const* cols,
                     int const* h_block_row_offset,
                     int const* h_local_row_size);

  virtual ~BEllPackStructInfo() ;

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

  void computeUpperDiagOffset() const
  {
  }

  Integer computeBandeSize() const
  {
    return 0 ;
  }

  Integer computeUpperBandeSize() const
  {
    return 0;
  }

  Integer computeLowerBandeSize() const
  {
    return 0;
  }

  Integer computeMaxRowSize() const
  {
    m_max_row_size = 0;
    return m_max_row_size;
  }

  Integer getMaxRowSize() const
  {
    if (m_max_row_size == -1)
      computeMaxRowSize();
    return m_max_row_size;
  }


 protected:
  // clang-format off
  std::size_t m_block_nrows         = 0 ;
  std::size_t m_block_nnz           = 0 ;
  mutable Integer m_max_row_size    = -1 ;

  InternalType*  m_internal         = nullptr;
  int const*     m_h_local_row_size = nullptr ;
  // clang-format on
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
