// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#pragma once

#include <vector>
#include <span>

#include <alien/data/IVector.h>

namespace Alien
{
  class Timestamp ;

  class SYCLParallelEngine ;

  class SYCLControlGroupHandler ;

  namespace SYCL
  {

    template <typename ValueT>
    class VectorAccessorT
    {
     public:
      typedef ValueT ValueType;

      class Impl ;

      class View ;

      class ConstView ;

      class HostView ;

     public:
      VectorAccessorT(IVector& vector, bool update = true);

      virtual ~VectorAccessorT() { end(); }

      void end();

      View view(SYCLControlGroupHandler& cgh) ;

      ConstView constView(SYCLControlGroupHandler& cgh) const ;

      HostView hostView() const ;

      Impl* impl() ;

     protected:
      Timestamp* m_time_stamp;
      Integer m_local_offset;
      std::unique_ptr<Impl> m_impl ;
      bool m_finalized;
    };

  }
}
