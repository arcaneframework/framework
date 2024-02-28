/*
 * VectorAccessor.h
 *
 *  Created on: 16 févr. 2024
 *      Author: gratienj
 */

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




     protected:
      Timestamp* m_time_stamp;
      Integer m_local_offset;
      Impl* m_impl = nullptr;
      bool m_finalized;
    };

  }
}
