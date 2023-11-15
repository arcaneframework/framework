// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include "alien/distribution/MatrixDistribution.h"
#include "SendRecvOp.h" // FIXME: remove
#include "SimpleCSRPrecomp.h"
#include <cstdlib>
#include <unordered_set>

namespace Alien
{

class VBlock;

}

/*---------------------------------------------------------------------------*/

namespace Alien::SimpleCSRInternal
{

class CSRStructInfo;

/*---------------------------------------------------------------------------*/

class ALIEN_EXPORT DistStructInfo
{
 public:
  DistStructInfo() {}

  ~DistStructInfo() {}

  DistStructInfo(const DistStructInfo& src) { copy(src); }

  DistStructInfo& operator=(const DistStructInfo& src)
  {
    copy(src);
    return *this;
  }

  void compute(Integer nproc, ConstArrayView<Integer> offset, Integer my_rank,
               IMessagePassingMng* parallel_mng, const CSRStructInfo& profile,
               ITraceMng* trace = NULL);

  void compute(Integer nproc, ConstArrayView<Integer> offset, Integer my_rank,
               IMessagePassingMng* parallel_mng, const CSRStructInfo& profile,
               const VBlock* block_sizes, const MatrixDistribution& dist, ITraceMng* trace = NULL);

  Integer domainId(Integer nproc, ConstArrayView<Integer> offset, Integer id)
  {
    for (Integer ip = 0; ip < nproc; ++ip) {
      if (id < offset[ip + 1])
        return ip;
    }
    return -1;
  }

  bool isInterfaceRow(Arccore::Integer row_id) const
  {
    return m_interface_row_set.find(row_id) != m_interface_row_set.end();
  }

  void copy(const DistStructInfo& distStructInfo);

  void computeUpperDiagOffset(const CSRStructInfo& profile) const
  {
    auto nrows = profile.getNRows();
    auto row_offset = profile.getRowOffset();
    m_upper_diag_offset.resize(nrows);
    for (int irow = 0; irow < nrows; ++irow) {
      int index = row_offset[irow];
      for (int k = row_offset[irow]; k < row_offset[irow] + m_local_row_size[irow]; ++k) {
        if (m_cols[k] < irow)
          ++index;
        else
          break;
      }
      m_upper_diag_offset[irow] = index;
    }
  }

  ConstArrayView<Integer> getUpperDiagOffset(const CSRStructInfo& profile) const
  {
    if (m_upper_diag_offset.size() == 0)
      computeUpperDiagOffset(profile);
    return m_upper_diag_offset.constView();
  }

  int const* dcol(const CSRStructInfo& profile) const
  {
    getUpperDiagOffset(profile);
    return m_upper_diag_offset.data();
  }

  void computeBlock2DSizesAndOffsets(Integer const* kcol, Integer const* dcol, Integer const* bcol) const;

  // clang-format off
  Arccore::UniqueArray<Arccore::Integer> m_local_row_size;
  Arccore::Integer                       m_ghost_nrow = 0;
  Arccore::Integer                       m_interface_nrow = 0;
  Arccore::Integer                       m_first_upper_ghost_index = 0;
  Arccore::UniqueArray<Arccore::Integer> m_interface_rows;
  std::unordered_set<int>                m_interface_row_set;
  Arccore::UniqueArray<Arccore::Integer> m_cols;
  mutable
  Arccore::UniqueArray<Arccore::Integer> m_upper_diag_offset;
  mutable
  CommInfo                               m_recv_info;
  mutable
  CommInfo                               m_send_info;
  Arccore::UniqueArray<Arccore::Integer> m_block_sizes;
  Arccore::UniqueArray<Arccore::Integer> m_block_offsets;
  mutable
  Arccore::UniqueArray<Arccore::Integer> m_block2d_sizes;
  mutable
  Arccore::UniqueArray<Arccore::Integer> m_block2d_offsets;
  // clang-format on
};

/*---------------------------------------------------------------------------*/

} // namespace Alien::SimpleCSRInternal

/*---------------------------------------------------------------------------*/
