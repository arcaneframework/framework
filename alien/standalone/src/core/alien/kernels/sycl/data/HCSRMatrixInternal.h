// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/kernels/sycl/SYCLPrecomp.h>

#ifdef USE_SYCL2020
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include "alien/kernels/sycl/data/HCSRMatrix.h"

#include "SYCLEnv.h"
#include "SYCLEnvInternal.h"
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

namespace Alien
{

  namespace HCSRInternal
  {

  /*---------------------------------------------------------------------------*/

  #ifndef USE_SYCL2020
    using namespace cl ;
  #endif

  template <typename ValueT = Real>
  class MatrixInternal
  {
   public:
    // clang-format off
    typedef ValueT                           ValueType;
    typedef MatrixInternal<ValueType>        ThisType;
    typedef SimpleCSRInternal::CSRStructInfo ProfileType;
    typedef typename ProfileType::IndexType  IndexType ;
    typedef sycl::buffer<ValueType, 1>       ValueBufferType;
    typedef sycl::buffer<IndexType, 1>       IndexBufferType;


    class HypreProfile
    {
    public:
      HypreProfile(std::size_t nrows, IndexType* rows, IndexType* ncols)
      : m_rows(rows,nrows)
      , m_ncols(ncols,nrows)
      {}
      mutable IndexBufferType m_rows;
      mutable IndexBufferType m_ncols;
    };

    class COOProfile
    {
    public:
      COOProfile(std::size_t nrows,
                 std::size_t nnz,
                 std::size_t nghost,
                 IndexType* rows,
                 IndexType* ncols,
                 IndexType* ghost_uids)
      : m_rows(rows,nnz)
      , m_ncols(ncols,nrows)
      //, m_ghost_uids(ghost_uids,nghost)
      {}
      mutable IndexBufferType m_rows;
      mutable IndexBufferType m_ncols;
      //mutable IndexBufferType m_ghost_uids;
    };
    // clang-format on

   public:
    MatrixInternal(ProfileType* profile)
    : m_profile(profile)
    , m_values(sycl::range<1>(profile->getNnz()+1))
    , m_kcol(profile->kcol(),profile->getNRows()+1)
    , m_cols(profile->cols(),profile->getNnz())
    {
      m_values.set_final_data(nullptr);
    }

    virtual ~MatrixInternal() {}


    ProfileType& getCSRProfile() {
      assert(m_profile) ;
      return *m_profile ;
    }

    ProfileType const& getCSRProfile() const {
      assert(m_profile) ;
      return *m_profile ;
    }

    void setValues(ValueT value)
    {
        auto env = SYCLEnv::instance() ;
        env->internal()->queue().submit([&](sycl::handler& cgh)
                                         {
                                           auto access_x = m_values.template get_access<sycl::access::mode::read_write>(cgh);
                                           cgh.fill(access_x,value) ;
                                         }) ;
    }

    /*
    ConstArrayView<ValueType> getValues() const
    {
      return ConstArrayView<ValueType>(m_h_values.size(),m_h_values.data());
    }

    ArrayView<ValueType> getValues() {
      return ArrayView<ValueType>(m_h_values.size(),m_h_values.data());
    }
    */

    ValueBufferType& values()
    {
      return m_values;
    }

    ValueBufferType& values() const
    {
      return m_values;
    }

    IndexBufferType& kcol() {
      return m_kcol;
    }

    IndexBufferType const& kcol() const {
      return m_kcol;
    }

    IndexBufferType& cols() {
      return m_cols;
    }

    IndexBufferType const& cols() const {
      return m_cols;
    }

    HypreProfile& getHypreProfile(IndexType local_offset) const
    {
      if(m_hypre_profile.get()==nullptr)
      {
          assert(m_profile) ;
          auto nrows = m_profile->getNRow();
          auto kcol = m_profile->kcol() ;
          m_global_rows_ids.resize(nrows) ;
          m_row_sizes.resize(nrows) ;
          for(IndexType irow=0;irow<nrows;++irow)
            {
              m_global_rows_ids[irow] = local_offset + irow;
              m_row_sizes[irow] = kcol[irow+1] - kcol[irow];
            }
          m_hypre_profile.reset(new HypreProfile(nrows,m_global_rows_ids.data(),m_row_sizes.data()));
      }
      return *m_hypre_profile ;
    }

    COOProfile& getCOOProfile(IndexType local_offset,
                              std::size_t ghost_size=0,
                              const int* ghost_uids=nullptr) const
    {
      alien_debug([&] {
        cout() << "create COOProfile "<<m_coo_profile.get()<<" "<<m_profile->getNRow()<<" "<< m_profile->getNnz()<< " "<<ghost_size<<" "<<ghost_uids;
      });
      if(m_coo_profile.get()==nullptr)
      {
          assert(m_profile) ;
          auto nrows = m_profile->getNRow();
          auto nnz = m_profile->getNnz();
          auto kcol = m_profile->kcol() ;
          m_global_rows_ids.resize(nnz) ;
          m_row_sizes.resize(nrows) ;
          for(IndexType irow=0;irow<nrows;++irow)
          {
            auto row_uid = local_offset + irow;
            m_row_sizes[irow] = kcol[irow+1] - kcol[irow];
            for(std::size_t k=kcol[irow];k<kcol[irow+1];++k)
              m_global_rows_ids[k] = row_uid ;
          }
          IndexType* ghost_uids_ptr = ghost_uids ? (IndexType*)ghost_uids : (IndexType*)&m_empty_ghost_uids ;

          m_coo_profile.reset(new COOProfile(nrows,
                                             nnz,
                                             ghost_size,
                                             m_global_rows_ids.data(),
                                             m_row_sizes.data(),
                                             ghost_uids_ptr));
      }
      alien_debug([&] {
        cout() << "create COOProfile OK";
      });
      return *m_coo_profile ;
    }



    void copyValuesToHost(std::size_t nnz, ValueT* ptr)
    {
      auto h_values = m_values.get_host_access();
      for (std::size_t i = 0; i < nnz; ++i)
        ptr[i] = h_values[i];
    }
    ProfileType* m_profile = nullptr ;

    mutable ValueBufferType                m_values;
    mutable IndexBufferType                m_kcol;
    mutable IndexBufferType                m_cols;
    mutable std::vector<IndexType>         m_global_rows_ids;
    mutable std::vector<IndexType>         m_row_sizes;
    mutable std::unique_ptr<HypreProfile>  m_hypre_profile ;
    mutable std::unique_ptr<COOProfile>    m_coo_profile ;
    Integer m_empty_ghost_uids = 0 ;
  };

  /*---------------------------------------------------------------------------*/

  } // namespace Alien::SYCLInternal

  template <typename ValueT>
  HCSRMatrix<ValueT>::HCSRMatrix()
  : IMatrixImpl(nullptr, AlgebraTraits<BackEnd::tag::hcsr>::name())
  , m_local_size(0)
  {
    m_profile.reset(new ProfileType()) ;
  }

  //! Constructeur avec association ? un MultiImpl
  template <typename ValueT>
  HCSRMatrix<ValueT>::HCSRMatrix(const MultiMatrixImpl* multi_impl)
  : IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::hcsr>::name())
  , m_local_size(0)
  {
    m_profile.reset(new ProfileType()) ;
  }


  template <typename ValueT>
  HCSRMatrix<ValueT>::~HCSRMatrix()
  {

  }
/*---------------------------------------------------------------------------*/
template <typename ValueT>
void HCSRMatrix<ValueT>::allocate()
{
  assert(m_profile.get()) ;
  m_internal.reset(new InternalType(m_profile.get())) ;
}


/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
