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

#include <alien/handlers/scalar/sycl/ProfiledMatrixBuilderT.h>

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
  template <typename ValueT,typename IndexT>
  class ProfiledMatrixBuilderT<ValueT,IndexT>::Impl
  {
  public :
    typedef typename HCSRMatrix<ValueT>::InternalType MatrixInternalType ;
    typedef typename MatrixInternalType::ValueBufferType ValueBufferType ;
    typedef typename MatrixInternalType::IndexBufferType IndexBufferType ;

    Impl(ValueBufferType& values_buffer,
         IndexBufferType& cols_buffer,
         IndexBufferType& kcol_buffer)
    : m_values_buffer(values_buffer)
    , m_cols_buffer(cols_buffer)
    , m_kcol_buffer(kcol_buffer)
    {}

    ValueBufferType& m_values_buffer ;
    IndexBufferType& m_cols_buffer ;
    IndexBufferType& m_kcol_buffer ;
  };

  template <typename ValueT,typename IndexT>
  ProfiledMatrixBuilderT<ValueT,IndexT>::ProfiledMatrixBuilderT(IMatrix& matrix, ResetFlag reset_values)
  : m_matrix(matrix)
  , m_finalized(false)
  {
    m_matrix.impl()->lock();
    m_matrix_impl = &m_matrix.impl()->get<BackEnd::tag::hcsr>(true);

    const MatrixDistribution& dist = m_matrix_impl->distribution();

    m_local_size = dist.localRowSize();
    m_local_offset = dist.rowOffset();
    m_next_offset = m_local_offset + m_local_size;

    SimpleCSRInternal::CSRStructInfo const& profile =
    m_matrix_impl->getCSRProfile();
    m_row_starts = profile.getRowOffset();
    m_local_row_size = m_matrix_impl->getDistStructInfo().m_local_row_size;
    m_cols = profile.getCols();
    m_impl.reset(new Impl(m_matrix_impl->internal()->values(),
                          m_matrix_impl->internal()->cols(),
                          m_matrix_impl->internal()->kcol())) ;
  }

  template <typename ValueT,typename IndexT>
  ProfiledMatrixBuilderT<ValueT,IndexT>::~ProfiledMatrixBuilderT()
  {
    if (!m_finalized) {
      finalize();
    }
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT,typename IndexT>
  void ProfiledMatrixBuilderT<ValueT,IndexT>::finalize()
  {
    if (m_finalized)
      return;
    m_matrix.impl()->unlock();
    m_finalized = true;
  }

  template <typename ValueT, typename IndexT>
  class ProfiledMatrixBuilderT<ValueT,IndexT>::View
  {
    sycl::handler* m_h = nullptr ;
    sycl::buffer<ValueT,1>* m_vb = nullptr ;
    sycl::buffer<IndexT,1>* m_ib = nullptr ;
    using ValueAccessorType = decltype(m_vb->template get_access<sycl::access::mode::read_write>(*m_h));
    using IndexAccessorType = decltype(m_ib->template get_access<sycl::access::mode::read>(*m_h));

  public :
    explicit View(ValueAccessorType values_accessor,
                  IndexAccessorType cols_accessor,
                  IndexAccessorType kcol_accessor)
    : m_values_accessor(values_accessor)
    , m_cols_accessor(cols_accessor)
    , m_kcol_accessor(kcol_accessor)
    {}

    ValueT& operator[](IndexT index) const {
      return m_values_accessor[index] ;
    }

    IndexT entryIndex(IndexT row, IndexT col) const {
      for(auto k=m_kcol_accessor[row];k<m_kcol_accessor[row+1];++k)
        if(m_cols_accessor[k]==col)
          return k ;
      return -1 ;
    }

  protected :
    ValueAccessorType m_values_accessor ;
    IndexAccessorType m_cols_accessor ;
    IndexAccessorType m_kcol_accessor ;

  } ;

  template <typename ValueT,typename IndexT>
  class ProfiledMatrixBuilderT<ValueT,IndexT>::ConstView
  {
    sycl::handler* m_h = nullptr ;
    sycl::buffer<ValueT,1>* m_vb = nullptr ;
    sycl::buffer<IndexT,1>* m_ib = nullptr ;
    using ValueAccessorType = decltype(m_vb->template get_access<sycl::access::mode::read>(*m_h));
    using IndexAccessorType = decltype(m_ib->template get_access<sycl::access::mode::read>(*m_h));

  public :
    explicit ConstView(ValueAccessorType values_accessor,
                       IndexAccessorType cols_accessor,
                       IndexAccessorType kcol_accessor)
    : m_values_accessor(values_accessor)
    , m_cols_accessor(cols_accessor)
    , m_kcol_accessor(kcol_accessor)
    {}

    ValueT operator[](IndexT index) const {
      return m_values_accessor[index] ;
    }

    IndexT entryIndex(IndexT row,IndexT col) const {
      for(auto k=m_kcol_accessor[row];k<m_kcol_accessor[row+1];++k)
        if(m_cols_accessor[k]==col)
          return k ;
      return -1 ;
    }

  protected :
    ValueAccessorType m_values_accessor ;
    IndexAccessorType m_cols_accessor ;
    IndexAccessorType m_kcol_accessor ;

  } ;


  template <typename ValueT,typename IndexT>
  class ProfiledMatrixBuilderT<ValueT,IndexT>::HostView
  {
  public :
    sycl::buffer<ValueT,1>* m_b = nullptr ;
    using ValueAccessorType = decltype(m_b->get_host_access());

    sycl::buffer<IndexT,1>* m_ib = nullptr ;
    using IndexAccessorType = decltype(m_ib->get_host_access());

    HostView(ValueAccessorType values,
             IndexAccessorType cols,
             IndexAccessorType kcol)
    : m_values(values)
    , m_cols(cols)
    , m_kcol(kcol)
    {}

    ValueType operator[](IndexT index) const {
      return m_values[index] ;
    }

    IndexT entryIndex(IndexT row,IndexT col) const {
      for(auto k=m_kcol[row];k<m_kcol[row+1];++k)
        if(m_cols[k]==col)
          return k ;
      return -1 ;
    }

    IndexT kcol(IndexT row) const {
      return m_kcol[row] ;
    }

    IndexT col(IndexT index) const {
      return m_cols[index] ;
    }


  protected:
    ValueAccessorType m_values ;
    IndexAccessorType m_cols;
    IndexAccessorType m_kcol;
    //std::span<IndexT> m_kcol ;
    //std::span<IndexT> m_cols ;
  };

  /*---------------------------------------------------------------------------*/
  template <typename ValueT,typename IndexT>
  typename ProfiledMatrixBuilderT<ValueT,IndexT>::View ProfiledMatrixBuilderT<ValueT,IndexT>::view(SYCLControlGroupHandler& cgh)
  {
    return View(m_impl->m_values_buffer.template get_access<sycl::access::mode::read_write>(cgh.m_internal),
                m_impl->m_cols_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal),
                m_impl->m_kcol_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal)) ;
  }

  template <typename ValueT,typename IndexT>
  typename ProfiledMatrixBuilderT<ValueT,IndexT>::ConstView ProfiledMatrixBuilderT<ValueT,IndexT>::constView(SYCLControlGroupHandler& cgh) const
  {
    return ProfiledMatrixBuilderT<ValueT,IndexT>::ConstView(m_impl->m_values_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal),
                                                     m_impl->m_cols_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal),
                                                     m_impl->m_kcol_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal)) ;
  }

  template <typename ValueT,typename IndexT>
  typename ProfiledMatrixBuilderT<ValueT,IndexT>::HostView ProfiledMatrixBuilderT<ValueT,IndexT>::hostView() const
  {
    return HostView(m_impl->m_values_buffer.get_host_access(),
                    m_impl->m_cols_buffer.get_host_access(),
                    m_impl->m_kcol_buffer.get_host_access()) ;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace SYCL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
