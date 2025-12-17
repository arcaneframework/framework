// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
    typedef typename HCSRMatrix<ValueT>::InternalType MatrixInternalType ;
    typedef typename MatrixInternalType::ValueBufferType ValueBufferType ;
    typedef typename MatrixInternalType::IndexBufferType IndexBufferType ;

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
    m_nnz = this->m_row_starts[this->m_local_size] ;
    m_combine_size = m_max_nb_contributors*m_nnz ;
    m_contributor_indexes.resize(m_combine_size) ;
    m_contributor_indexes.assign(m_combine_size,-1) ;

    auto f =  [&](int contrib_index, int row_index, int col_index)
              {
                auto eij = this->entryIndex(row_index,col_index) ;
                auto offset = eij*m_max_nb_contributors ;
                for(std::size_t c=0;c<m_max_nb_contributors;++c)
                {
                  if(m_contributor_indexes[offset+c]==contrib_index)
                    break ;
                  if(m_contributor_indexes[offset+c]==-1)
                  {
                    m_contributor_indexes[offset+c] = contrib_index ;
                    break ;
                  }
                }
              } ;
    for(int irow=0;irow<this->m_local_size;++irow)
    {
      for(auto k=stencil_offsets[irow];k<stencil_offsets[irow+1];++k)
      {
        auto col = stencil_indexes[k] ;
        f(col,irow,irow) ;
        f(col,irow,col) ;
      }
    }
    m_impl.reset(new Impl{m_combine_size,m_contributor_indexes.data()}) ;
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

    typedef typename ProfiledMatrixBuilderT<ValueT,IndexT>::View BaseType ;
  public :
    explicit View(ValueAccessorType values_accessor,
                  IndexAccessorType cols_accessor,
                  IndexAccessorType kcol_accessor,
                  ValueAccessorType combine_values_accessor,
                  IndexAccessorType prow_cols_accessor,
                  std::size_t nb_contributor)
    : BaseType(values_accessor,cols_accessor,kcol_accessor)
    , m_combine_values_accessor(combine_values_accessor)
    , m_prow_cols_accessor(prow_cols_accessor)
    , m_nb_contributor(nb_contributor)
    {

    }

    IndexT combineEntryIndex(IndexT prow, IndexT row, IndexT col) const {
      for(auto k=this->m_kcol_accessor[row];k<this->m_kcol_accessor[row+1];++k)
        if(this->m_cols_accessor[k]==col)
        {
          for(std::size_t j=0;j<m_nb_contributor;++j)
          {
            if(m_prow_cols_accessor[m_nb_contributor*k+j]==prow)
              return (IndexT)m_nb_contributor*k+j ;
          }
        }
      return -1 ;
    }


    void combine(IndexT index,ValueT value) const {
      m_combine_values_accessor[index] = value ;
    }

  private :
    ValueAccessorType m_combine_values_accessor ;
    IndexAccessorType m_prow_cols_accessor ;
    std::size_t m_nb_contributor = 0 ;
  } ;

  /*---------------------------------------------------------------------------*/
  template <typename ValueT,typename IndexT, typename CombineOpT>
  typename CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::View CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::view(SYCLControlGroupHandler& cgh)
  {
    return View(BaseType::m_impl->m_values_buffer.template get_access<sycl::access::mode::read_write>(cgh.m_internal),
                BaseType::m_impl->m_cols_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal),
                BaseType::m_impl->m_kcol_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal),
                m_impl->m_combine_values_buffer.template get_access<sycl::access::mode::read_write>(cgh.m_internal),
                m_impl->m_contributor_indexes_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal),
                m_max_nb_contributors) ;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  template <typename ValueT,typename IndexT, typename CombineOpT>
  void CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::
  combine()
  {
    auto env = SYCLEnv::instance() ;
    auto nnz = m_nnz ;
    auto nb_contributors = m_max_nb_contributors ;
    auto total_threads = env->maxNumThreads() ;
    env->internal()->queue().submit([&](sycl::handler& cgh)
                                    {
                                       auto values_acc = BaseType::m_impl->m_values_buffer.template get_access<sycl::access::mode::read_write>(cgh);
                                       auto combine_values_acc = m_impl->m_combine_values_buffer.template get_access<sycl::access::mode::read>(cgh);
                                       cgh.parallel_for<class class_combine>( sycl::range<1>{total_threads},
                                                                              [=] (sycl::item<1> itemId)
                                                                              {
                                                                                  auto id = itemId.get_id(0);
                                                                                  for (auto k = id; k < nnz; k += itemId.get_range()[0])
                                                                                  {
                                                                                     auto value = values_acc[k] ;
                                                                                     for(std::size_t c=0;c<nb_contributors;++c)
                                                                                     {
                                                                                       value = CombineOpT::apply(value,combine_values_acc[nb_contributors*k+c]) ;
                                                                                     }
                                                                                     values_acc[k] = value ;
                                                                                  }
                                                                              }) ;
                                     }) ;
  }
  /*---------------------------------------------------------------------------*/


  template <typename ValueT,typename IndexT, typename CombineOpT>
  class CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::HostView
  : public ProfiledMatrixBuilderT<ValueT,IndexT>::HostView
  {
  public :
    sycl::buffer<ValueT,1>* m_b = nullptr ;
    using ValueAccessorType = decltype(m_b->get_host_access());

    sycl::buffer<IndexT,1>* m_ib = nullptr ;
    using IndexAccessorType = decltype(m_ib->get_host_access());


    typedef typename ProfiledMatrixBuilderT<ValueT,IndexT>::HostView BaseType ;

    HostView(ValueAccessorType values,
             IndexAccessorType cols,
             IndexAccessorType kcol,
             ValueAccessorType combine_values,
             IndexAccessorType prow_cols,
             std::size_t nb_contributor)
    : BaseType(values,cols,kcol)
    , m_combine_values(combine_values)
    , m_prow_cols(prow_cols)
    , m_nb_contributor(nb_contributor)
    {}


    IndexT combineEntryIndex(IndexT prow, IndexT row, IndexT col) const {
      for(auto k=this->m_kcol[row];k<this->m_kcol[row+1];++k)
        if(this->m_cols[k]==col)
        {
          for(std::size_t j=0;j<m_nb_contributor;++j)
          {
            if(m_prow_cols[m_nb_contributor*k+j]==prow)
              return m_nb_contributor*k+j ;
          }
        }
      return -1 ;
    }

  private:
    ValueAccessorType m_combine_values ;
    IndexAccessorType m_prow_cols;
    std::size_t m_nb_contributor = 0 ;
  };


  template <typename ValueT,typename IndexT, typename CombineOpT>
  typename CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::HostView
  CombineProfiledMatrixBuilderT<ValueT,IndexT,CombineOpT>::hostView()
  {
    return HostView(BaseType::m_impl->m_values_buffer.get_host_access(),
                    BaseType::m_impl->m_cols_buffer.get_host_access(),
                    BaseType::m_impl->m_kcol_buffer.get_host_access(),
                    m_impl->m_combine_values_buffer.get_host_access(),
                    m_impl->m_contributor_indexes_buffer.get_host_access(),
                    m_max_nb_contributors) ;
  }
  /*---------------------------------------------------------------------------*/

} // namespace SYCL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
