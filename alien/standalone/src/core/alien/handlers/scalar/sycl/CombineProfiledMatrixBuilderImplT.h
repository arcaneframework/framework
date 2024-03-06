// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#pragma once

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/sycl/data/HCSRMatrix.h>
#include <alien/kernels/sycl/data/HCSRMatrixInternal.h>

#include <alien/kernels/sycl/data/SYCLParallelEngine.h>
#include <alien/kernels/sycl/data/SYCLParallelEngineImplT.h>

#include <alien/handlers/scalar/sycl/CombineProfiledMatrixBuilderT.h>

#include <span>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace SYCL
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  template <typename ValueT,typename IndexT, typename CombineOpT>
  class CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::Impl
  {
  public :
    typedef HCSRMatrix<ValueT>::InternalType MatrixInternalType ;
    typedef MatrixInternalType::ValueBufferType ValueBufferType ;
    typedef MatrixInternalType::IndexBufferType IndexBufferType ;

    Impl(std::size_t size,
         IndexT const* ptr)
    : m_combine_values_buffer(sycl::range(size))
    , m_contributor_indexes_buffer(ptr,sycl::range(size))
    {
      m_contributor_indexes_buffer.set_final_data(nullptr) ;

      auto env = SYCLEnv::instance() ;
      env->internal()->queue().submit([&](sycl::handler& cgh)
                                       {
                                         auto init_value = CombineOpT::init_value() ;
                                         auto access_x = m_combine_values_buffer.template get_access<sycl::access::mode::read_write>(cgh);
                                         cgh.fill(access_x,ValueT(init_value)) ;
                                       }) ;
    }

    ValueBufferType m_combine_values_buffer ;
    IndexBufferType m_contributor_indexes_buffer ;
  };

  template <typename ValueT,typename IndexT, typename CombineOpT>
  CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::CombineProfiledMatrixBuilderT(IMatrix& matrix, ProfiledMatrixOptions::ResetFlag reset_values)
  : BaseType(matrix,reset_values)
  {}

  template <typename ValueT,typename IndexT, typename CombineOpT>
  CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::~CombineProfiledMatrixBuilderT()
  {
  }


  template <typename ValueT,typename IndexT, typename CombineOpT>
  void CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::
  setParallelAssembleStencil(std::size_t max_nb_contributors,
                             Arccore::ConstArrayView<IndexT> stencil_offsets,
                             Arccore::ConstArrayView<IndexT> stencil_indexes)
  {
    m_max_nb_contributors = max_nb_contributors ;
    std::size_t nnz = this->m_row_starts[this->m_local_size] ;
    std::size_t size = m_max_nb_contributors*nnz ;
    m_contributor_indexes.resize(size) ;
    m_contributor_indexes.assign(size,-1) ;
    for(int irow=0;irow<this->m_local_size;++irow)
    {
      for(auto k=stencil_indexes[irow];k<stencil_indexes[irow+1];++k)
      {
        auto col = stencil_indexes[k] ;
        auto eij = this->entryIndex(col,irow) ;
        auto offset = eij*m_max_nb_contributors ;
        for(int c=0;c<m_max_nb_contributors;++c)
        {
          if(m_contributor_indexes[offset+c]==irow)
            break ;
          if(m_contributor_indexes[offset+c]==-1)
          {
            m_contributor_indexes[offset+c] = irow ;
            break ;
          }
        }
      }
    }
    m_impl.reset(new Impl{size,m_contributor_indexes.data()}) ;
  }

  /*---------------------------------------------------------------------------*/


  template <typename ValueT, typename IndexT, typename CombineOpT>
  class CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::View
  : public ProfiledMatrixBuilderT<ValueT,IndexT>::View
  {
    sycl::handler* m_h = nullptr ;
    sycl::buffer<ValueT,1>* m_vb = nullptr ;
    sycl::buffer<IndexT,1>* m_ib = nullptr ;
    using ValueAccessorType = decltype(m_vb->template get_access<sycl::access::mode::read_write>(*m_h));
    using IndexAccessorType = decltype(m_ib->template get_access<sycl::access::mode::read>(*m_h));

    typedef ProfiledMatrixBuilderT<ValueT,IndexT>::View BaseType ;
  public :
    explicit View(ValueAccessorType values_accessor,
                  IndexAccessorType cols_accessor,
                  IndexAccessorType kcol_accessor,
                  ValueAccessorType combine_values_accessor,
                  IndexAccessorType prow_cols_accessor)
    : BaseType(values_accessor,cols_accessor,kcol_accessor)
    , m_combine_values_accessor(combine_values_accessor)
    , m_prow_cols_accessor(prow_cols_accessor)
    {}

    IndexT combineEntryIndex(IndexT prow, IndexT row, IndexT col) const {
      for(auto k=m_kcol_accessor[row];k<m_kcol_accessor[row+1];++k)
        if(m_cols_accessor[k]==col)
        {
          for(int j=0;j<m_nb_contributor;++j)
          {
            if(m_prow_cols_accessor[m_nb_contributor*k+j]==prow)
              return m_nb_contributor*k+j ;
          }
        }
      return -1 ;
    }


    void combine(IndexT index,ValueT value) const {
      m_combine_values_accessor[index] = value ;
    }

  private :
    ValueAccessorType m_values_accessor ;
    IndexAccessorType m_cols_accessor ;
    IndexAccessorType m_kcol_accessor ;

    ValueAccessorType m_combine_values_accessor ;
    IndexAccessorType m_prow_cols_accessor ;
    int m_nb_contributor = 0 ;

  } ;

  /*---------------------------------------------------------------------------*/
  template <typename ValueT,typename IndexT, typename CombineOpT>
  CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::View CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::view(SYCLControlGroupHandler& cgh)
  {
    return View(BaseType::m_impl->m_values_buffer.template get_access<sycl::access::mode::read_write>(cgh.m_internal),
                BaseType::m_impl->m_cols_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal),
                BaseType::m_impl->m_kcol_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal),
                m_impl->m_combine_values_buffer.template get_access<sycl::access::mode::read_write>(cgh.m_internal),
                m_impl->m_contributor_indexes_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal)) ;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  template <typename ValueT,typename IndexT, typename CombineOpT>
  void CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::
  combine()
  {

  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace SYCL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
