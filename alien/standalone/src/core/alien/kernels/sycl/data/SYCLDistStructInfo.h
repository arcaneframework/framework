// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <cstdlib>
#include <unordered_set>

#include "alien/distribution/MatrixDistribution.h"
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/DistStructInfo.h>


#include <alien/kernels/sycl/SYCLPrecomp.h>

namespace Alien {

/*---------------------------------------------------------------------------*/
namespace SYCLInternal {


class ALIEN_EXPORT SYCLDistStructInfo
: public SimpleCSRInternal::DistStructInfo
{
 public:
  //static constexpr int PKSIZE   = 256 ;
  using BaseType                = SimpleCSRInternal::DistStructInfo;
  using ProfileType             = BEllPackStructInfo<PKSIZE,int>;

  SYCLDistStructInfo()
  : BaseType()
  {}

  SYCLDistStructInfo(const SYCLDistStructInfo& src)
  {
    BaseType::copy(src);
  }


  SYCLDistStructInfo& operator=(const SYCLDistStructInfo& src)
  {
    BaseType::copy(src);
    return *this;
  }

  void computeUpperDiagOffset(const ProfileType& profile) const
  {
    auto nrows            = profile.getNRows();
    auto const* kcol      = profile.kcol();
    auto const* lrow_size = profile.localRowSize();
    m_upper_diag_offset.resize(nrows);
    for (int irow = 0; irow < nrows; ++irow)
    {
      int begin = kcol[irow] ;
      int end   = kcol[irow] + lrow_size[irow] ;
      int index = begin;
      for (int k = begin; k < end; ++k)
      {
        if (m_cols[k] < irow)
          ++index;
        else
          break;
      }
      m_upper_diag_offset[irow] = index;
    }
  }

  ConstArrayView<Integer> getUpperDiagOffset(const ProfileType& profile) const
  {
    if (m_upper_diag_offset.size() == 0)
      computeUpperDiagOffset(profile);
    return m_upper_diag_offset.constView();
  }

  int const* dcol(const ProfileType& profile) const
  {
    getUpperDiagOffset(profile);
    return this->m_upper_diag_offset.data();
  }

};

} // end namespace SYCLInternal

} // end namespace Alien
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
