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

namespace Alien
{


/*---------------------------------------------------------------------------*/
namespace SYCLInternal {


class ALIEN_EXPORT SYCLDistStructInfo : public SimpleCSRInternal::DistStructInfo
{
 public:
  typedef SimpleCSRInternal::DistStructInfo BaseType ;

  typedef BEllPackStructInfo<1024,int>      ProfileType;

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


  int const* dcol([[maybe_unused]] const ProfileType& profile) const
  {
    //getUpperDiagOffset(profile);
    //return m_upper_diag_offset.data();
    return nullptr ;
  }


};

}

}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
