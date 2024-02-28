/*
 * SYCLParallelEngine.h
 *
 *  Created on: 15 févr. 2024
 *      Author: gratienj
 */

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
