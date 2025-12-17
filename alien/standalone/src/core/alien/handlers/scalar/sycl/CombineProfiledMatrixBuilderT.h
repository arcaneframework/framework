// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <arccore/base/Span.h>

#include <alien/handlers/scalar/sycl/ProfiledMatrixBuilderT.h>

namespace Alien
{

  template <typename Scalar>
  class HCSRMatrix;

  namespace SYCL
  {
    template<typename ValueT>
    class CombineAdd
    {
    public:
      static ValueT apply(ValueT a, ValueT b) {
        return a+b ;
      }

      static ValueT init_value()  {
        return ValueT() ;
      }
    };

    template<typename ValueT>
    class CombineMult
    {
    public:
      static ValueT apply(ValueT a, ValueT b) {
        return a*b ;
      }

      static ValueT init_value() {
        return  ValueT(1) ;
      }
    };


    template <typename ValueT, typename IndexT, typename CombineOpT=CombineAdd<ValueT> >
    class ALIEN_EXPORT CombineProfiledMatrixBuilderT
    : public ProfiledMatrixBuilderT<ValueT,IndexT>
    {

     public:

      class Impl ;


      typedef ProfiledMatrixBuilderT<ValueT,IndexT> BaseType ;

      CombineProfiledMatrixBuilderT(IMatrix& matrix, ProfiledMatrixOptions::ResetFlag reset_values);

      virtual ~CombineProfiledMatrixBuilderT();

      CombineProfiledMatrixBuilderT(const CombineProfiledMatrixBuilderT&) = delete;
      CombineProfiledMatrixBuilderT(CombineProfiledMatrixBuilderT&&) = delete;
      CombineProfiledMatrixBuilderT& operator=(const CombineProfiledMatrixBuilderT&) = delete;
      CombineProfiledMatrixBuilderT& operator=(CombineProfiledMatrixBuilderT&&) = delete;

     public:
      void setParallelAssembleStencil(std::size_t max_nb_contributors,
                                      Arccore::ConstArrayView<IndexT> stencil_offsets,
                                      Arccore::ConstArrayView<IndexT> stencil_indexes) ;

      std::size_t combineSize() const {
        return m_combine_size ;
      }

      class View ;

      View view(SYCLControlGroupHandler& cgh) ;

      class HostView ;

      HostView hostView() ;

      void combine();

     private:
      std::unique_ptr<Impl> m_impl;
      std::size_t m_nnz = 0 ;
      std::size_t m_combine_size = 0 ;

      std::size_t m_max_nb_contributors = 0 ;
      std::vector<IndexT> m_contributor_indexes ;

    };


    /*---------------------------------------------------------------------------*/
    /*---------------------------------------------------------------------------*/

    typedef CombineProfiledMatrixBuilderT<Real,Integer,CombineAdd<Real>> CombineAddProfiledMatrixBuilder ;
    typedef CombineProfiledMatrixBuilderT<Real,Integer,CombineMult<Real>> CombineMultProfiledMatrixBuilder ;

  } // namespace SYCL

}

