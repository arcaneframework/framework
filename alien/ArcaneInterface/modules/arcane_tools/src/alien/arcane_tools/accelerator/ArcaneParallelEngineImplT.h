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
#include <alien/kernels/sycl/data/SYCLParallelEngine.h>
#include <alien/kernels/sycl/data/SYCLParallelEngineInternal.h>

#include <alien/arcane_tools/accelerator/ArcaneParallelEngine.h>
#include <alien/arcane_tools/accelerator/ArcaneParallelEngineInternal.h>
namespace Alien
{

  class ControlGroupHandler : public SYCLControlGroupHandler
  {
  public:
    ControlGroupHandler(sycl::handler& cgh, Arcane::Accelerator::RunQueue& queue)
    : SYCLControlGroupHandler(cgh)
    , m_run_command(Arcane::Accelerator::makeCommand(queue))
    {}

    Arcane::Accelerator::RunCommand& command() {
      return m_run_command ;
    }

    Arcane::Accelerator::RunCommand m_run_command ;
  } ;

  template<typename LambdaT>
  void ParallelEngine::submit(LambdaT lambda)
  {
    m_internal->m_env->internal()->queue().submit( [&](sycl::handler& cgh)
                                       {
                                          ControlGroupHandler handler(cgh,m_queue) ;
                                          lambda(handler) ;
                                       }
                                     ) ;
  }

}
