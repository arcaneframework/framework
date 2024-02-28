// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/sycl/SYCLPrecomp.h>

namespace Alien
{
namespace SYCLInternal
{
  struct EngineInternal;
}

class ALIEN_EXPORT SYCLControlGroupHandler ;

class ALIEN_EXPORT SYCLParallelEngine
{
 public:
  typedef SYCLInternal::EngineInternal InternalType ;

  template<int dim> struct Item ;


  SYCLParallelEngine() ;

  virtual ~SYCLParallelEngine() ;

  std::size_t maxNumThreads() const ;

  template<typename LambdaT>
  void submit(LambdaT lambda) ;

 private :
  //std::unique_ptr<InternalType> m_internal ;
  InternalType* m_internal = nullptr ;

} ;

}
