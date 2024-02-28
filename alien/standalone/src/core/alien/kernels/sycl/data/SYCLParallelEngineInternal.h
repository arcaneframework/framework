/*
 * SYCLParallelEngine.h
 *
 *  Created on: 15 févr. 2024
 *      Author: gratienj
 */

#pragma once

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

      SYCLEnv* m_env ;
    };
  }

  template<int dim>
  struct SYCLParallelEngine::Item : public sycl::item<dim>
  {
    Item(sycl::item<1> const& item)
    : sycl::item<dim>(item)
    {}

    Item(Item const& item)
    : sycl::item<dim>(item.base())
    {}

    sycl::item<1> const& base() const {
      return *this ;
    }
  } ;

}
