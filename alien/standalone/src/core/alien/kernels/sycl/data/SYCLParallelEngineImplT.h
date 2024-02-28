/*
 * SYCLParallelEngine.h
 *
 *  Created on: 15 févr. 2024
 *      Author: gratienj
 */

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
      m_internal.parallel_for<LambdaT>(sycl::range<1>{range},
                                            lambda) ;
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
