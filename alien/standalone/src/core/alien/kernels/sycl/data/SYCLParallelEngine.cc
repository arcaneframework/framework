// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
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

#include <cassert>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/sycl/SYCLPrecomp.h>

#include "SYCLEnv.h"
#include "SYCLParallelEngine.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

namespace Alien
{

  namespace SYCLInternal
  {
    struct EngineInternal
    {
      EngineInternal()
      {
        m_env = SYCLEnv::instance() ;
      }
      SYCLEnv* m_env = nullptr ;
    };
  }

  SYCLParallelEngine::SYCLParallelEngine()
  {
    m_internal.reset(new SYCLInternal::EngineInternal()) ;
  }

  SYCLParallelEngine::~SYCLParallelEngine()
  {
    //delete m_internal ;
  }

  std::size_t  SYCLParallelEngine::maxNumThreads() const {
    return m_internal->m_env->maxNumThreads() ;
  }

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
