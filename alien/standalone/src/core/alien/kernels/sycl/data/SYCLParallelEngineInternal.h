// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


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
    using type = sycl::item<dim>;

    Item(sycl::item<dim> const& item)
    : sycl::item<dim>(item)
    {}

    Item(Item const& item)
    : sycl::item<dim>(item.base())
    {}

    sycl::item<dim> const& base() const {
      return *this ;
    }
  } ;

}
