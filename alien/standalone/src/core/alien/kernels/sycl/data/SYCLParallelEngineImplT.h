// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#pragma once

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/sycl/SYCLPrecomp.h>

#include <alien/kernels/sycl/data/SYCLEnv.h>
#include <alien/kernels/sycl/data/SYCLEnvInternal.h>
#include <alien/kernels/sycl/data/SYCLParallelEngineInternal.h>
namespace Alien
{

  class SYCLControlGroupHandler
  {
  public:
    SYCLControlGroupHandler(sycl::handler& cgh)
    : m_internal(cgh)
    {}

    template<typename LambdaT>
    void parallel_for(std::size_t range, LambdaT lambda)
    {
      m_internal.parallel_for<LambdaT>(sycl::range<1>{range},lambda) ;
    }

    template<typename LambdaT>
    void parallel_for(std::size_t range1, std::size_t range2, LambdaT lambda)
    {
      m_internal.parallel_for<LambdaT>(sycl::range<2>{range1,range2},lambda) ;
    }

    sycl::handler& m_internal ;

  } ;

  template<typename LambdaT>
  void SYCLParallelEngine::submit(LambdaT lambda)
  {
    m_internal->m_env->internal()->queue().submit( [&](sycl::handler& cgh)
                                       {
                                          SYCLControlGroupHandler handler(cgh) ;
                                          lambda(handler) ;
                                       }
                                     ) ;
  }

}
