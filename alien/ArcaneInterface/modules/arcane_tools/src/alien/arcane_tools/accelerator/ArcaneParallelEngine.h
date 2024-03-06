// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

namespace Alien
{
namespace Internal
{
  struct EngineInternal;
}

class ALIEN_EXPORT ControlGroupHandler ;

class ALIEN_EXPORT ParallelEngine
{
 public:
  typedef Internal::EngineInternal InternalType ;

  template<int dim> struct Item ;


  ParallelEngine(Arcane::Accelerator::RunQueue& queue) ;

  virtual ~ParallelEngine() ;

  std::size_t maxNumThreads() const ;

  template<typename LambdaT>
  void submit(LambdaT lambda) ;

 private :
  std::unique_ptr<InternalType> m_internal ;
  Arcane::Accelerator::RunQueue& m_queue ;

} ;

}
