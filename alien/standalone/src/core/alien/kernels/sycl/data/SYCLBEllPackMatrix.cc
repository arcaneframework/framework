// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>

#include <alien/kernels/sycl/data/SYCLEnvInternal.h>
#include <alien/kernels/sycl/data/SYCLEnv.h>

#include <alien/kernels/sycl/data/SYCLVectorInternal.h>
#include <alien/kernels/sycl/data/SYCLVector.h>

#include <alien/kernels/sycl/data/SYCLBEllPackInternal.h>
#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>

/*---------------------------------------------------------------------------*/

namespace Alien
{

SYCLEnv::SYCLEnv()
{
  m_internal = new SYCLInternal::EnvInternal;
}

SYCLEnv::~SYCLEnv()
{
  delete m_internal;
}

SYCLEnv* SYCLEnv::m_instance = nullptr;

SYCLEnv* SYCLEnv::instance()
{
  if (!m_instance)
    m_instance = new SYCLEnv;
  return m_instance;
}

std::size_t SYCLEnv::maxNumGroups()
{
  return m_internal->maxNumGroups();
}

std::size_t SYCLEnv::maxWorkGroupSize()
{
  return m_internal->maxWorkGroupSize();
}

std::size_t SYCLEnv::maxNumThreads()
{
  return m_internal->maxNumThreads();
}

template <int EllpackSize, typename IndexT>
void BEllPackStructInfo<EllpackSize, IndexT>::computeBlockRowOffset(std::vector<int>& block_row_offset,
                                                                   std::size_t nrows,
                                                                   int const* kcol)
{
  std::size_t block_nrows = BEllPackStructInfo<EllpackSize, IndexT>::nbBlocks(nrows);
  block_row_offset.resize(block_nrows + 1);
  int offset = 0;
  for (std::size_t ib = 0; ib < block_nrows; ++ib) {
    block_row_offset[ib] = offset;
    int max_row_size = 0;
    for (int i = 0; i < std::min(EllpackSize, int(nrows - ib * EllpackSize)); ++i) {
      auto irow = ib * EllpackSize + i;
      int row_size = kcol[irow + 1] - kcol[irow];
      max_row_size = std::max(max_row_size, row_size);
    }
    offset += max_row_size;
  }
  block_row_offset[block_nrows] = offset;
}

template <int EllpackSize, typename IndexT>
SYCLInternal::StructInfoInternal<EllpackSize, IndexT>::
StructInfoInternal(std::size_t nrows,
                   std::size_t nnz,
                   std::size_t block_nrows,
                   std::size_t block_nnz,
                   int const* h_kcol,
                   int const* h_cols,
                   int const* h_block_row_offset,
                   int const* h_local_row_size)
: m_nrows(nrows)
, m_nnz(nnz)
, m_block_nrows(block_nrows)
, m_block_nnz(block_nnz)
, m_h_kcol(h_kcol, h_kcol + nrows + 1)
, m_h_cols(h_cols, h_cols + nnz)
, m_h_block_cols(block_nnz * ellpack_size)
, m_block_row_offset(h_block_row_offset, sycl::range<1>(m_block_nrows + 1))
, m_block_cols(m_h_block_cols.data(), sycl::range<1>(m_block_nnz * ellpack_size))
, m_kcol(m_h_kcol.data(), sycl::range<1>(m_nrows + 1))
{
  auto env = SYCLEnv::instance();

  auto& queue = env->internal()->queue();

  auto num_groups = env->internal()->maxNumGroups();

  auto max_work_group_size = env->internal()->maxWorkGroupSize();

  // building the best number of global thread
  auto total_threads = num_groups * ellpack_size;

  // clang-format off
  if(h_local_row_size==nullptr)
  {
    IndexBufferType cols_buffer(m_h_cols.data(), sycl::range<1>(m_nnz));

    queue.submit([&](sycl::handler& cgh)
                 {
                   auto access_kcol_buffer      = m_kcol.template get_access<sycl::access::mode::read>(cgh);
                   auto access_block_row_offset = m_block_row_offset.template get_access<sycl::access::mode::read>(cgh);

                   auto access_cols_buffer      = cols_buffer.template get_access<sycl::access::mode::read>(cgh);
                   auto access_block_cols       = m_block_cols.template get_access<sycl::access::mode::discard_write>(cgh);

                   cgh.parallel_for<class struct_info_sycl>(sycl::range<1>{total_threads},
                                                     [=] (sycl::item<1> itemId)
                                                     {
                                                         auto id = itemId.get_id(0);

                                                         for (auto i = id; i < nrows; i += itemId.get_range()[0])
                                                         {
                                                           auto block_id = i/ellpack_size ;
                                                           auto local_id = i%ellpack_size ;

                                                           int begin              = access_kcol_buffer[i] ;
                                                           int end                = access_kcol_buffer[i+1] ;
                                                           int row_size           = end - begin ;

                                                           int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                           auto block_row_size = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                           //access_rx[i] = row_size;
                                                            /*
                                                            access_brs[i] = block_row_size ;
                                                            access_begin[i] = begin ;
                                                            access_end[i] = end ;
                                                            access_first_cols[i] = access_cols_buffer[begin] ;
                                                            */
                                                            for(int k=begin;k<end;++k)
                                                            {
                                                              access_block_cols[block_row_offset+(k-begin)*ellpack_size+local_id] = access_cols_buffer[k] ;
                                                            }
                                                            for(int k=row_size;k<block_row_size;++k)
                                                            {
                                                              access_block_cols[block_row_offset+k*ellpack_size+local_id] = -1 ;
                                                            }

                                                         }
                                                      });
                 });
  }
  else
  {
    IndexBufferType cols_buffer(m_h_cols.data(), sycl::range<1>(m_nnz));
    IndexBufferType lrowsize_buffer(h_local_row_size, sycl::range<1>(nrows));
    // clang-format off
    queue.submit([&](sycl::handler& cgh)
                 {
                   auto access_kcol_buffer      = m_kcol.template get_access<sycl::access::mode::read>(cgh);
                   auto access_block_row_offset = m_block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                   auto access_lrowsize_buffer  = lrowsize_buffer.template get_access<sycl::access::mode::read>(cgh);

                   auto access_cols_buffer      = cols_buffer.template get_access<sycl::access::mode::read>(cgh);
                   auto access_block_cols       = m_block_cols.template get_access<sycl::access::mode::discard_write>(cgh);

                   cgh.parallel_for<class struct_info_sycl2>(sycl::range<1>{total_threads},
                                                     [=] (sycl::item<1> itemId)
                                                     {
                                                         auto id = itemId.get_id(0);

                                                         for (auto i = id; i < nrows; i += itemId.get_range()[0])
                                                         {
                                                           auto block_id = i/ellpack_size ;
                                                           auto local_id = i%ellpack_size ;

                                                           int begin              = access_kcol_buffer[i] ;
                                                           int lrow_size          = access_lrowsize_buffer[i] ;

                                                           int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                           auto block_row_size = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                            for(int k=begin;k<begin+lrow_size;++k)
                                                            {
                                                              access_block_cols[block_row_offset+(k-begin)*ellpack_size+local_id] = access_cols_buffer[k] ;
                                                            }
                                                            for(int k=lrow_size;k<block_row_size;++k)
                                                            {
                                                              access_block_cols[block_row_offset+k*ellpack_size+local_id] = -1 ;
                                                            }
                                                         }
                                                      });
                 });
  }
  // clang-format on
}

template <int EllpackSize, typename IndexT>
void SYCLInternal::StructInfoInternal<EllpackSize, IndexT>::getUpperDiagOffset() const
{
  if (m_h_dcol.size() == 0) {
    m_h_dcol.resize(m_nrows);
    for (std::size_t irow = 0; irow < m_nrows; ++irow) {
      int index = m_h_kcol[irow];
      for (int k = m_h_kcol[irow]; k < m_h_kcol[irow + 1]; ++k) {
        if ((std::size_t)m_h_cols[k] < irow)
          ++index;
        else
          break;
      }
      m_h_dcol[irow] = index;
    }
  }
}

template <int EllpackSize, typename IndexT>
void SYCLInternal::StructInfoInternal<EllpackSize, IndexT>::computeLowerUpperMask() const
{

  if (not m_lower_upper_mask_ready) {
    m_lower_mask.reset(new MaskBufferType(sycl::range<1>(m_block_nnz * ellpack_size)));
    m_upper_mask.reset(new MaskBufferType(sycl::range<1>(m_block_nnz * ellpack_size)));

    auto env = SYCLEnv::instance();

    auto& queue = env->internal()->queue();

    auto num_groups = env->internal()->maxNumGroups();

    auto max_work_group_size = env->internal()->maxWorkGroupSize();

    // building the best number of global thread
    auto total_threads = num_groups * ellpack_size;

    {
      IndexBufferType dcol_buffer(m_h_dcol.data(), sycl::range<1>(m_nrows));

      auto nrows = m_nrows;

      // clang-format off
      queue.submit([&](sycl::handler& cgh)
                   {
                     auto access_dcol_buffer      = dcol_buffer.template get_access<sycl::access::mode::read>(cgh);

                     auto access_kcol_buffer      = m_kcol.template get_access<sycl::access::mode::read>(cgh);
                     auto access_block_row_offset = m_block_row_offset.template get_access<sycl::access::mode::read>(cgh);

                     auto access_lower_mask       = sycl::accessor { *m_lower_mask, cgh, sycl::write_only, sycl::property::no_init{}};
                     auto access_upper_mask       = sycl::accessor { *m_upper_mask, cgh, sycl::write_only, sycl::property::no_init{}};


                     cgh.parallel_for<class lower_upper_mask>(sycl::range<1>{total_threads},
                                                       [=] (sycl::item<1> itemId)
                                                       {
                                                           auto id = itemId.get_id(0);

                                                           for (auto i = id; i < nrows; i += itemId.get_range()[0])
                                                           {
                                                             auto block_id = i/ellpack_size ;
                                                             auto local_id = i%ellpack_size ;

                                                             int begin              = access_kcol_buffer[i] ;
                                                             int end                = access_kcol_buffer[i+1] ;
                                                             int diag               = access_dcol_buffer[i] ;
                                                             int row_size           = end - begin ;

                                                             int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                             auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                             access_lower_mask[block_row_offset+(diag-begin)*ellpack_size+local_id] = 0 ;
                                                             access_upper_mask[block_row_offset+(diag-begin)*ellpack_size+local_id] = 0 ;

                                                             for(int k=begin;k<diag;++k)
                                                             {
                                                                access_lower_mask[block_row_offset+(k-begin)*ellpack_size+local_id] = 1 ;
                                                                access_upper_mask[block_row_offset+(k-begin)*ellpack_size+local_id] = 0 ;
                                                             }
                                                             for(int k=diag+1;k<end;++k)
                                                             {
                                                                access_lower_mask[block_row_offset+(k-begin)*ellpack_size+local_id] = 0 ;
                                                                access_upper_mask[block_row_offset+(k-begin)*ellpack_size+local_id] = 1 ;
                                                             }
                                                             for(int k=row_size;k<block_row_size;++k)
                                                             {
                                                                access_lower_mask[block_row_offset+k*ellpack_size+local_id] = 0 ;
                                                                access_upper_mask[block_row_offset+k*ellpack_size+local_id] = 0 ;
                                                             }

                                                           }
                                                        });
                   });
      // clang-format on
    }
    m_lower_upper_mask_ready = true;
  }
}

template <int EllpackSize, typename IndexT>
typename SYCLInternal::StructInfoInternal<EllpackSize, IndexT>::MaskBufferType&
SYCLInternal::StructInfoInternal<EllpackSize, IndexT>::getLowerMask() const
{
  computeLowerUpperMask();
  return *m_lower_mask;
}

template <int EllpackSize, typename IndexT>
typename SYCLInternal::StructInfoInternal<EllpackSize, IndexT>::MaskBufferType&
SYCLInternal::StructInfoInternal<EllpackSize, IndexT>::getUpperMask() const
{
  computeLowerUpperMask();
  return *m_upper_mask;
}

template <int EllpackSize, typename IndexT>
BEllPackStructInfo<EllpackSize, IndexT>::BEllPackStructInfo(std::size_t nrows,
                                                          int const* kcol,
                                                          int const* cols,
                                                          int const* h_block_row_offset,
                                                          int const* h_local_row_size)

: BaseBEllPackStructInfo(nrows, kcol[nrows])
, m_block_nrows(BEllPackStructInfo<EllpackSize, IndexT>::nbBlocks(nrows))
, m_block_nnz(h_block_row_offset[m_block_nrows])
, m_h_local_row_size(h_local_row_size)
{
  // clang-format off
  alien_debug([&] {
                       cout() << "COMPUTE BELLPACK PROFILE";
                       cout() << "       NROWS : "<<this->m_nrows;
                       cout() << "       NNZ   : "<<this->m_nnz;
                       cout() << " BLOCK NROWS : "<<m_block_nrows;
                       cout()  <<" BLOCK NNZ   : "<<m_block_nnz;
                      });
  // clang-format on

  //m_h_block_cols.assign(m_block_nnz*ellpack_size,-1) ;
  m_internal = new InternalType{ this->m_nrows,
                                 this->m_nnz,
                                 m_block_nrows,
                                 m_block_nnz,
                                 kcol,
                                 cols,
                                 h_block_row_offset,
                                 h_local_row_size };
}

template <int EllpackSize, typename IndexT>
typename BEllPackStructInfo<EllpackSize, IndexT>::IndexType const*
BEllPackStructInfo<EllpackSize, IndexT>::kcol() const
{
  return m_internal->kcol();
}

template <int EllpackSize, typename IndexT>
typename BEllPackStructInfo<EllpackSize, IndexT>::IndexType const*
BEllPackStructInfo<EllpackSize, IndexT>::cols() const
{
  return m_internal->cols();
}

template <int EllpackSize, typename IndexT>
typename BEllPackStructInfo<EllpackSize, IndexT>::IndexType const*
BEllPackStructInfo<EllpackSize, IndexT>::dcol() const
{
  return m_internal->dcol();
}

namespace SYCLInternal
{
  template <typename ValueT, int EllpackSize>
  MatrixInternal<ValueT, EllpackSize>::MatrixInternal(ProfileType const* profile, int N)
  : m_profile(profile)
  , m_N(N)
  , m_NxN(N*N)
  , m_h_values(profile->getBlockNnz() * ellpack_size * N*N)
  , m_values(m_h_values.data(), sycl::range<1>(profile->getBlockNnz() * ellpack_size * N*N))
  {
    //m_values.set_final_data(nullptr);
    alien_debug([&] { cout() << "SYCL InternalMATRIX" << profile->getBlockNnz() * ellpack_size<< "N="<<m_N; });
  }

  template <typename ValueT, int EllpackSize>
  bool MatrixInternal<ValueT, EllpackSize>::setMatrixValuesFromHost()
  {
    alien_debug([&] { cout() << "SYCLMatrix setMatrixValuesFromHost "; });
    auto env = SYCLEnv::instance();
    auto& queue = env->internal()->queue();
    auto num_groups = env->internal()->maxNumGroups();
    auto max_work_group_size = env->internal()->maxWorkGroupSize();
    auto total_threads = num_groups * ellpack_size;

    int N          = this->m_N;
    int NxN        = this->m_NxN;
    int pack_size  = ellpack_size ;
    auto nrows     = m_profile->getNRows();
    auto nnz       = m_profile->getNnz();
    auto block_nnz = m_profile->getBlockNnz();

    auto internal_profile = m_profile->internal();
    auto& kcol = internal_profile->getKCol();
    auto& block_row_offset = internal_profile->getBlockRowOffset();
    auto& block_cols       = internal_profile->getBlockCols();
    auto local_row_size = m_profile->localRowSize();
    if(N==1)
    {
      if (local_row_size == nullptr)
      {
        assert(m_h_csr_values.size()>=nnz) ;
        ValueBufferType values_buffer(m_h_csr_values.data(), sycl::range<1>(nnz));
        // COMPUTE COLS
        // clang-format off
          queue.submit([&](sycl::handler& cgh)
                       {
                         auto access_kcol_buffer      = kcol.template get_access<sycl::access::mode::read>(cgh);
                         auto access_block_row_offset = block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                         auto access_values_buffer    = values_buffer.template get_access<sycl::access::mode::read>(cgh);
                         auto access_block_values     = m_values.template get_access<sycl::access::mode::read_write>(cgh);

                         //cgh.parallel_for<class vector_axpy>(sycl::nd_range<1>{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}},[=](sycl::nd_item<1> item_id)
                         cgh.parallel_for<class set_matrix_values>(sycl::range<1>{total_threads},
                                                               [=] (sycl::item<1> item_id)
                                                               {
                                                                 auto id = item_id.get_id(0);
                                                                 //auto local_id  = item_id.get_local_id(0);
                                                                 //auto block_id  = item_id.get_group(0) ;
                                                                 //auto global_id = item_id.get_global_id(0);

                                                                 for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                                 {
                                                                    auto block_id = i/ellpack_size ;
                                                                    auto local_id = i%ellpack_size ;

                                                                    auto begin              = access_kcol_buffer[i] ;
                                                                    auto end                = access_kcol_buffer[i+1] ;
                                                                    auto row_size           = end - begin ;

                                                                    int block_row_offset   = access_block_row_offset[block_id];
                                                                    auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                    for(int k=0;k<row_size;++k)
                                                                    {
                                                                       access_block_values[(block_row_offset+k)*ellpack_size+local_id] = access_values_buffer[begin+k] ;
                                                                    }
                                                                    for(int k=row_size;k<block_row_size;++k)
                                                                    {
                                                                       access_block_values[(block_row_offset+k)*ellpack_size+local_id] = 0 ;
                                                                    }
                                                                 }
                                                              });
                       }) ;
        // clang-format on
        m_values_is_update = true;
      }
      else
      {
        ValueBufferType values_buffer(m_h_csr_values.data(), sycl::range<1>(nnz));
        IndexBufferType lrowsize_buffer(local_row_size, sycl::range<1>(nrows));
        // COMPUTE COLS
        // clang-format off
        queue.submit([&](sycl::handler& cgh)
                     {
                       auto access_kcol_buffer      = internal_profile->getKCol().template get_access<sycl::access::mode::read>(cgh);
                       auto access_lrowsize_buffer  = lrowsize_buffer.template get_access<sycl::access::mode::read>(cgh);

                       auto access_block_row_offset = internal_profile->getBlockRowOffset().template get_access<sycl::access::mode::read>(cgh);
                       auto access_values_buffer    = values_buffer.template get_access<sycl::access::mode::read>(cgh);
                       auto access_block_values     = m_values.template get_access<sycl::access::mode::read_write>(cgh);

                       //cgh.parallel_for<class vector_axpy>(sycl::nd_range<1>{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}},[=](sycl::nd_item<1> item_id)
                       cgh.parallel_for<class set_matrix_values2>(sycl::range<1>{total_threads},
                                                             [=] (sycl::item<1> item_id)
                                                             {
                                                               auto id = item_id.get_id(0);
                                                               //auto local_id  = item_id.get_local_id(0);
                                                               //auto block_id  = item_id.get_group(0) ;
                                                               //auto global_id = item_id.get_global_id(0);

                                                               for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                               {
                                                                  auto block_id = i/pack_size ;
                                                                  auto local_id = i%pack_size ;

                                                                  auto begin              = access_kcol_buffer[i] ;
                                                                  auto lrow_size          = access_lrowsize_buffer[i] ;
                                                                  auto end                = begin + lrow_size ;

                                                                  int block_row_offset   = access_block_row_offset[block_id];
                                                                  auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                  for(int k=0;k<lrow_size;++k)
                                                                  {
                                                                     access_block_values[(block_row_offset+k)*pack_size+local_id] = access_values_buffer[begin+k] ;
                                                                  }
                                                                  for(int k=lrow_size;k<block_row_size;++k)
                                                                  {
                                                                     access_block_values[(block_row_offset+k)*pack_size+local_id] = 0. ;
                                                                  }
                                                               }
                                                            });
                     }) ;

        auto interface_nrows = m_ext_profile->getNRows();
        auto ext_nnz         = m_ext_profile->getNnz();
        auto ext_block_nnz   = m_ext_profile->getBlockNnz();

        auto ext_internal_profile  = m_ext_profile->internal();
        auto& ext_kcol             = ext_internal_profile->getKCol();
        auto& ext_block_row_offset = ext_internal_profile->getBlockRowOffset();
        //m_h_ext_values.resize(ext_block_nnz) ;
        m_ext_values.reset(new ValueBufferType(ext_block_nnz * ellpack_size * NxN)) ;
        {
          ValueBufferType ext_csr_values_buffer(m_h_csr_ext_values.data(), sycl::range<1>(ext_nnz));
          queue.submit([&](sycl::handler& cgh)
                       {
                         auto access_kcol_buffer      = ext_kcol.template get_access<sycl::access::mode::read>(cgh);
                         auto access_block_row_offset = ext_block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                         auto access_values_buffer    = ext_csr_values_buffer.template get_access<sycl::access::mode::read>(cgh);
                         //auto access_block_values     = m_ext_values->template get_access<sycl::access::mode::read_write>(cgh);
                         auto access_block_values     = sycl::accessor { *m_ext_values, cgh, sycl::write_only, sycl::property::no_init{}};

                         cgh.parallel_for<class set_matrix_values3>(sycl::range<1>{total_threads},
                                                                     [=] (sycl::item<1> item_id)
                                                                     {
                                                                       auto id = item_id.get_id(0);

                                                                       for (auto i = id; i < interface_nrows; i += item_id.get_range()[0])
                                                                       {
                                                                          auto block_id = i/ellpack_size ;
                                                                          auto local_id = i%ellpack_size ;
                                                                          auto begin              = access_kcol_buffer[i] ;
                                                                          auto end                = access_kcol_buffer[i+1] ;
                                                                          auto row_size           = end - begin ;

                                                                          int block_row_offset   = access_block_row_offset[block_id]*ellpack_size*NxN ;
                                                                          auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                          for(int k=begin*NxN;k<end*NxN;++k)
                                                                          {
                                                                            access_block_values[block_row_offset+(k-begin*NxN)*ellpack_size+local_id] = access_values_buffer[k] ;
                                                                          }
                                                                          for(int k=row_size*NxN;k<block_row_size*NxN;++k)
                                                                          {
                                                                            access_block_values[block_row_offset+k*ellpack_size+local_id] = 0 ;
                                                                          }
                                                                       }
                                                                    });
                         }).wait() ;
          // clang-format on
        }
        m_values_is_update = true;
      }
    }
    else
    {
      if (local_row_size == nullptr)
      {
        assert(m_h_csr_values.size()>=nnz*NxN) ;
        ValueBufferType values_buffer(m_h_csr_values.data(), sycl::range<1>(nnz*NxN));
        // COMPUTE COLS
        // clang-format off
          queue.submit([&](sycl::handler& cgh)
                       {
                         auto access_kcol_buffer      = kcol.template get_access<sycl::access::mode::read>(cgh);
                         auto access_cols_buffer      = block_cols.template get_access<sycl::access::mode::read>(cgh);
                         auto access_block_row_offset = block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                         auto access_values_buffer    = values_buffer.template get_access<sycl::access::mode::read>(cgh);
                         auto access_block_values     = m_values.template get_access<sycl::access::mode::read_write>(cgh);
                         std::cout<<"BUFFER SIZES : "<<block_row_offset.size()<<" "<<kcol.size()<<" "<<values_buffer.size()<<" "<<m_values.size()<<std::endl ;
                         //cgh.parallel_for<class vector_axpy>(sycl::nd_range<1>{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}},[=](sycl::nd_item<1> item_id)
                         cgh.parallel_for<class set_block_matrix_values>(sycl::range<1>{total_threads},
                                                               [=] (sycl::item<1> item_id)
                                                               {
                                                                 auto id = item_id.get_id(0);
                                                                 //auto local_id  = item_id.get_local_id(0);
                                                                 //auto block_id  = item_id.get_group(0) ;
                                                                 //auto global_id = item_id.get_global_id(0);

                                                                 for (std::size_t i = id; i < nrows; i += item_id.get_range()[0])
                                                                 {
                                                                    int block_id = i/pack_size ;
                                                                    int local_id = i%pack_size ;

                                                                    int begin               = access_kcol_buffer[i] ;
                                                                    int end                 = access_kcol_buffer[i+1] ;
                                                                    int row_size            = end - begin ;

                                                                    int block_row_offset    = access_block_row_offset[block_id] ;
                                                                    int block_row_size      = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                    for(int k=0;k<row_size;++k)
                                                                    {
                                                                      for(int i=0;i<N;++i)
                                                                        for(int j=0;j<N;++j)
                                                                        {
                                                                          access_block_values[((block_row_offset+k)*NxN+i*N+j)*pack_size+local_id] = access_values_buffer[(begin+k)*NxN + i*N+j] ;
                                                                        }
                                                                      /*
                                                                      printf("MATRIX[%zu,%d] (%d,%d) = (%f,%f,%f,%f)\n",i,k,block_row_offset,begin, access_values_buffer[(begin+k)*NxN], access_values_buffer[(begin+k)*NxN+1], access_values_buffer[(begin+k)*NxN+2], access_values_buffer[(begin+k)*NxN +3]);*/

                                                                    }
                                                                    for(int k=row_size;k<block_row_size;++k)
                                                                    {
                                                                      for(int i=0;i<N;++i)
                                                                        for(int j=0;j<N;++j)
                                                                        {
                                                                          access_block_values[((block_row_offset+k)*NxN + i*N+j)*pack_size+local_id] = 0. ;
                                                                        }
                                                                    }
                                                                 }
                                                              });
                       }).wait() ;
        // clang-format on
        m_values_is_update = true;
        /*
        sycl::host_accessor<ValueT, 1, sycl::access::mode::read> h_acc(m_values);
        sycl::host_accessor<int, 1, sycl::access::mode::read> kcol_acc(block_row_offset);
        for(int irow=0;irow<nrows;++irow)
        {
          auto ib=irow/ellpack_size ;
          auto il=irow%ellpack_size ;
          std::cout<<"MAT["<<irow<<","<<ib<<","<<il<<"]:";
          for(int k=kcol_acc[ib];k<kcol_acc[ib+1];++k)
          {
              for(int i=0;i<N;++i)
                for(int j=0;j<N;++j)
                  std::cout<<h_acc[(k*NxN+i*N+j)*ellpack_size+il]<<",";
              std::cout<<std::endl;
          }
        }*/
      }
      else
      {
        ValueBufferType values_buffer(m_h_csr_values.data(), sycl::range<1>(nnz));
        IndexBufferType lrowsize_buffer(local_row_size, sycl::range<1>(nrows));
        // COMPUTE COLS
        // clang-format off
        queue.submit([&](sycl::handler& cgh)
                     {
                       auto access_kcol_buffer      = internal_profile->getKCol().template get_access<sycl::access::mode::read>(cgh);
                       auto access_lrowsize_buffer  = lrowsize_buffer.template get_access<sycl::access::mode::read>(cgh);

                       auto access_block_row_offset = internal_profile->getBlockRowOffset().template get_access<sycl::access::mode::read>(cgh);
                       auto access_values_buffer    = values_buffer.template get_access<sycl::access::mode::read>(cgh);
                       auto access_block_values     = m_values.template get_access<sycl::access::mode::read_write>(cgh);

                       //cgh.parallel_for<class vector_axpy>(sycl::nd_range<1>{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}},[=](sycl::nd_item<1> item_id)
                       cgh.parallel_for<class set_block_matrix_values2>(sycl::range<1>{total_threads},
                                                             [=] (sycl::item<1> item_id)
                                                             {
                                                               auto id = item_id.get_id(0);
                                                               //auto local_id  = item_id.get_local_id(0);
                                                               //auto block_id  = item_id.get_group(0) ;
                                                               //auto global_id = item_id.get_global_id(0);

                                                               for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                               {
                                                                  auto block_id = i/ellpack_size ;
                                                                  auto local_id = i%ellpack_size ;

                                                                  auto begin              = access_kcol_buffer[i] ;
                                                                  auto lrow_size          = access_lrowsize_buffer[i] ;
                                                                  auto end                = begin + lrow_size ;

                                                                  int block_row_offset   = access_block_row_offset[block_id];
                                                                  auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                  for(int k=0;k<lrow_size;++k)
                                                                  {
                                                                    for(int i=0;i<N;++i)
                                                                      for(int j=0;j<N;++j)
                                                                      {
                                                                        access_block_values[((block_row_offset+k)*NxN+i*N+j)*ellpack_size+local_id] = access_values_buffer[(begin+k)*NxN+i*N+j] ;
                                                                      }
                                                                  }
                                                                  for(int k=lrow_size;k<block_row_size;++k)
                                                                  {
                                                                    for(int i=0;i<N;++i)
                                                                      for(int j=0;j<N;++j)
                                                                      {
                                                                        access_block_values[((block_row_offset+k)*NxN+i*N+j)*ellpack_size+local_id] = 0. ;
                                                                      }
                                                                  }
                                                               }
                                                            });
                     }) ;


        auto interface_nrows = m_ext_profile->getNRows();
        auto ext_nnz         = m_ext_profile->getNnz();
        auto ext_block_nnz   = m_ext_profile->getBlockNnz();

        auto ext_internal_profile  = m_ext_profile->internal();
        auto& ext_kcol             = ext_internal_profile->getKCol();
        auto& ext_block_row_offset = ext_internal_profile->getBlockRowOffset();
        //m_h_ext_values.resize(ext_block_nnz) ;
        m_ext_values.reset(new ValueBufferType(ext_block_nnz * ellpack_size * NxN)) ;
        {
          ValueBufferType ext_csr_values_buffer(m_h_csr_ext_values.data(), sycl::range<1>(ext_nnz));
          queue.submit([&](sycl::handler& cgh)
                       {
                         auto access_kcol_buffer      = ext_kcol.template get_access<sycl::access::mode::read>(cgh);
                         auto access_block_row_offset = ext_block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                         auto access_values_buffer    = ext_csr_values_buffer.template get_access<sycl::access::mode::read>(cgh);
                         //auto access_block_values     = m_ext_values->template get_access<sycl::access::mode::read_write>(cgh);
                         auto access_block_values     = sycl::accessor { *m_ext_values, cgh, sycl::write_only, sycl::property::no_init{}};

                         cgh.parallel_for<class set_matrix_values3>(sycl::range<1>{total_threads},
                                                                     [=] (sycl::item<1> item_id)
                                                                     {
                                                                       auto id = item_id.get_id(0);

                                                                       for (auto i = id; i < interface_nrows; i += item_id.get_range()[0])
                                                                       {
                                                                          auto block_id = i/ellpack_size ;
                                                                          auto local_id = i%ellpack_size ;
                                                                          auto begin              = access_kcol_buffer[i] ;
                                                                          auto end                = access_kcol_buffer[i+1] ;
                                                                          auto row_size           = end - begin ;

                                                                          int block_row_offset   = access_block_row_offset[block_id]*ellpack_size*NxN ;
                                                                          auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                          for(int k=begin*NxN;k<end*NxN;++k)
                                                                          {
                                                                            access_block_values[block_row_offset+(k-begin*NxN)*ellpack_size+local_id] = access_values_buffer[k] ;
                                                                          }
                                                                          for(int k=row_size*NxN;k<block_row_size*NxN;++k)
                                                                          {
                                                                            access_block_values[block_row_offset+k*ellpack_size+local_id] = 0 ;
                                                                          }
                                                                       }
                                                                    });
                         }).wait() ;
          // clang-format on
        }
        m_values_is_update = true;
      }
    }

    return true;
  }

  template <typename ValueT, int EllpackSize>
  bool MatrixInternal<ValueT, EllpackSize>::setMatrixValues(ValueBufferType& values_buffer)
  {
    alien_debug([&] { cout() << "SYCLMatrix setMatrixValues "; });
    auto env = SYCLEnv::instance();
    auto& queue = env->internal()->queue();
    auto num_groups = env->internal()->maxNumGroups();
    auto max_work_group_size = env->internal()->maxWorkGroupSize();
    auto total_threads = num_groups * ellpack_size;

    int NxN = this->m_NxN;

    auto nrows = m_profile->getNRows();
    auto nnz = m_profile->getNnz();
    auto block_nnz = m_profile->getBlockNnz();

    auto internal_profile = m_profile->internal();
    auto& kcol = internal_profile->getKCol();
    auto& block_row_offset = internal_profile->getBlockRowOffset();

    auto local_row_size = m_profile->localRowSize();
    if (local_row_size == nullptr) {
      //ValueBufferType values_buffer(m_h_csr_values.data(), sycl::range<1>(nnz));
      // COMPUTE COLS
      // clang-format off
        queue.submit([&](sycl::handler& cgh)
                     {
                       auto access_kcol_buffer      = internal_profile->getKCol().template get_access<sycl::access::mode::read>(cgh);
                       auto access_block_row_offset = internal_profile->getBlockRowOffset().template get_access<sycl::access::mode::read>(cgh);
                       auto access_values_buffer    = values_buffer.template get_access<sycl::access::mode::read>(cgh);
                       auto access_block_values     = m_values.template get_access<sycl::access::mode::read_write>(cgh);

                       //cgh.parallel_for<class vector_axpy>(sycl::nd_range<1>{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}},[=](sycl::nd_item<1> item_id)
                       cgh.parallel_for<class set_matrix_values4>(sycl::range<1>{total_threads},
                                                             [=] (sycl::item<1> item_id)
                                                             {
                                                               auto id = item_id.get_id(0);
                                                               //auto local_id  = item_id.get_local_id(0);
                                                               //auto block_id  = item_id.get_group(0) ;
                                                               //auto global_id = item_id.get_global_id(0);

                                                               for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                               {
                                                                  auto block_id = i/ellpack_size ;
                                                                  auto local_id = i%ellpack_size ;

                                                                  auto begin              = access_kcol_buffer[i] ;
                                                                  auto end                = access_kcol_buffer[i+1] ;
                                                                  auto row_size           = end - begin ;

                                                                  int block_row_offset   = access_block_row_offset[block_id]*ellpack_size*NxN ;
                                                                  auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                  for(int k=begin*NxN;k<end*NxN;++k)
                                                                  {
                                                                    access_block_values[block_row_offset+(k-begin*NxN)*ellpack_size+local_id] = access_values_buffer[k] ;
                                                                  }
                                                                  for(int k=row_size*NxN;k<block_row_size*NxN;++k)
                                                                  {
                                                                    access_block_values[block_row_offset+k*ellpack_size+local_id] = 0 ;
                                                                  }
                                                               }
                                                            });
                     }) ;
      // clang-format on
      m_values_is_update = true;
    }
    else {
      //ValueBufferType values_buffer(m_h_csr_values.data(), sycl::range<1>(nnz));
      IndexBufferType lrowsize_buffer(local_row_size, sycl::range<1>(nrows));

      // COMPUTE COLS
      // clang-format off
      queue.submit([&](sycl::handler& cgh)
                   {
                     auto access_kcol_buffer      = internal_profile->getKCol().template get_access<sycl::access::mode::read>(cgh);
                     auto access_lrowsize_buffer  = lrowsize_buffer.template get_access<sycl::access::mode::read>(cgh);

                     auto access_block_row_offset = internal_profile->getBlockRowOffset().template get_access<sycl::access::mode::read>(cgh);
                     auto access_values_buffer    = values_buffer.template get_access<sycl::access::mode::read>(cgh);
                     auto access_block_values     = m_values.template get_access<sycl::access::mode::read_write>(cgh);

                     //cgh.parallel_for<class vector_axpy>(sycl::nd_range<1>{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}},[=](sycl::nd_item<1> item_id)
                     cgh.parallel_for<class set_matrix_values5>(sycl::range<1>{total_threads},
                                                           [=] (sycl::item<1> item_id)
                                                           {
                                                             auto id = item_id.get_id(0);
                                                             //auto local_id  = item_id.get_local_id(0);
                                                             //auto block_id  = item_id.get_group(0) ;
                                                             //auto global_id = item_id.get_global_id(0);

                                                             for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                             {
                                                                auto block_id = i/ellpack_size ;
                                                                auto local_id = i%ellpack_size ;

                                                                auto begin              = access_kcol_buffer[i] ;
                                                                auto lrow_size          = access_lrowsize_buffer[i] ;
                                                                auto end                = begin + lrow_size ;

                                                                int block_row_offset   = access_block_row_offset[block_id]*ellpack_size*NxN ;
                                                                auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                for(int k=begin*NxN;k<end*NxN;++k)
                                                                {
                                                                  access_block_values[block_row_offset+(k-begin*NxN)*ellpack_size+local_id] = access_values_buffer[k] ;
                                                                }
                                                                for(int k=lrow_size*NxN;k<block_row_size*NxN;++k)
                                                                {
                                                                  access_block_values[block_row_offset+k*ellpack_size+local_id] = 0 ;
                                                                }
                                                             }
                                                          });
                   }) ;
      auto kcol            = m_profile->kcol() ;
      auto local_row_size  = m_profile->localRowSize() ;

      auto interface_nrows = m_ext_profile->getNRows();
      auto ext_nnz         = m_ext_profile->getNnz();
      auto ext_block_nnz   = m_ext_profile->getBlockNnz();

      auto ext_internal_profile  = m_ext_profile->internal();
      auto& ext_kcol             = ext_internal_profile->getKCol();
      auto& ext_block_row_offset = ext_internal_profile->getBlockRowOffset();
      //m_h_ext_values.resize(ext_block_nnz) ;
      m_ext_values.reset(new ValueBufferType(ext_block_nnz * ellpack_size * NxN)) ;

      // EXTRACT EXTERNAL PROFILE
      {
        ValueBufferType ext_csr_values_buffer(m_h_csr_ext_values.data(), sycl::range<1>(ext_nnz * NxN));

        std::vector<Integer> h_local_row_offset(interface_nrows+1) ;
        {
          Integer offset = 0 ;
          for(std::size_t i=0;i<interface_nrows;++i)
          {
            h_local_row_offset[i] = offset ;
            offset += local_row_size[i] ;
          }
          h_local_row_offset[interface_nrows] = offset ;
        }
        IndexBufferType local_row_offset(h_local_row_offset.data(),sycl::range<1>(interface_nrows+1)) ;

        queue.submit([&](sycl::handler& cgh)
                     {
                        auto access_kcol              = internal_profile->getKCol().template get_access<sycl::access::mode::read>(cgh);
                        auto access_interface_row_ids = m_interface_row_ids->template get_access<sycl::access::mode::read>(cgh);
                        auto access_local_row_size    = lrowsize_buffer.template get_access<sycl::access::mode::read>(cgh);
                        auto access_local_row_offset  = local_row_offset.template get_access<sycl::access::mode::read>(cgh);
                        auto access_csr_values        = values_buffer.template get_access<sycl::access::mode::read>(cgh);
                        auto access_ext_csr_values    = ext_csr_values_buffer.template get_access<sycl::access::mode::read_write>(cgh);
                        cgh.parallel_for<class set_matrix_values6>(sycl::range<1>{total_threads},
                                                                     [=] (sycl::item<1> item_id)
                                                                     {
                                                                        auto id = item_id.get_id(0);
                                                                        for (auto i = id; i < interface_nrows; i += item_id.get_range()[0])
                                                                        {
                                                                            Integer jcol = access_local_row_offset[i] ;
                                                                            for (int k = (access_kcol[i] + access_local_row_size[i])*NxN; k < access_kcol[i + 1] * NxN; ++k)
                                                                                access_ext_csr_values[jcol++] = access_csr_values[k];
                                                                        }
                                                                     });
                     }).wait() ;

        queue.submit([&](sycl::handler& cgh)
                     {
                       auto access_kcol_buffer      = ext_kcol.template get_access<sycl::access::mode::read>(cgh);
                       auto access_block_row_offset = ext_block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                       auto access_values_buffer    = ext_csr_values_buffer.template get_access<sycl::access::mode::read>(cgh);
                       //auto access_block_values     = m_ext_values->template get_access<sycl::access::mode::read_write>(cgh);
                       auto access_block_values     = sycl::accessor { *m_ext_values, cgh, sycl::write_only, sycl::property::no_init{}};

                       cgh.parallel_for<class set_matrix_values7>(sycl::range<1>{total_threads},
                                                                   [=] (sycl::item<1> item_id)
                                                                   {
                                                                     auto id = item_id.get_id(0);

                                                                     for (auto i = id; i < interface_nrows; i += item_id.get_range()[0])
                                                                     {
                                                                        auto block_id = i/ellpack_size ;
                                                                        auto local_id = i%ellpack_size ;
                                                                        auto begin              = access_kcol_buffer[i] ;
                                                                        auto end                = access_kcol_buffer[i+1] ;
                                                                        auto row_size           = end - begin ;

                                                                        int block_row_offset   = access_block_row_offset[block_id]*ellpack_size*NxN ;
                                                                        auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                        for(int k=begin*NxN;k<end*NxN;++k)
                                                                        {
                                                                          access_block_values[block_row_offset+(k-begin*NxN)*ellpack_size+local_id] = access_values_buffer[k] ;
                                                                        }
                                                                        for(int k=row_size*NxN;k<block_row_size*NxN;++k)
                                                                        {
                                                                          access_block_values[block_row_offset+k*ellpack_size+local_id] = 0 ;
                                                                        }
                                                                     }
                                                                  });
                       }).wait() ;
        // clang-format on
      }
      m_values_is_update = true;
    }
    return true;
  }

  template <typename ValueT, int EllpackSize>
  bool MatrixInternal<ValueT, EllpackSize>::setMatrixValues(ValueT const* values, bool only_host)
  {
    alien_debug([&] { cout() << "SYCLMatrix setMatrixValues " << only_host; });

    int NxN = this->m_NxN;

    auto nnz = m_profile->getNnz();
    m_h_csr_values.resize(nnz*NxN);
    std::copy(values, values + nnz*NxN, m_h_csr_values.begin());
    if (m_ext_profile) {
      // clang-format off
      auto kcol            = m_profile->kcol() ;
      auto local_row_size  = m_profile->localRowSize() ;

      auto interface_nrows = m_ext_profile->getNRows() ;
      auto ext_nnz         = m_ext_profile->getNnz();
      // clang-format on

      m_h_csr_ext_values.resize(ext_nnz*NxN);

      // EXTRACT EXTERNAL PROFILE
      {
        int jcol = 0;
        for (std::size_t i = 0; i < interface_nrows; ++i) {
          auto id = m_h_interface_row_ids[i];
          for (int k = (kcol[id] + local_row_size[id])*NxN; k < kcol[id + 1]*NxN; ++k) {
            m_h_csr_ext_values[jcol++] = values[k];
          }
        }
      }
    }
    if (only_host) {
      m_values_is_update = false;
    }
    else {
      setMatrixValuesFromHost();
      m_values_is_update = true;
    }
    return true;
  }

  template <typename ValueT, int EllpackSize>
  bool MatrixInternal<ValueT, EllpackSize>::copy(std::size_t nb_blocks,
                                               int block_size,
                                               ValueBufferType& values_buffer,
                                               int rhs_block_size)
  {
    alien_debug([&] { cout() << "SYCLMatrix copy "<<nb_blocks<<" "<<block_size<<" RHS BLK-SIZE="<<rhs_block_size; });
    auto env = SYCLEnv::instance();
    auto& queue = env->internal()->queue();
    auto num_groups = env->internal()->maxNumGroups();
    auto max_work_group_size = env->internal()->maxWorkGroupSize();
    auto total_threads = num_groups * ellpack_size;

    auto nrows = m_profile->getNRows();
    auto nnz = m_profile->getNnz();
    auto block_nnz = m_profile->getBlockNnz();

    auto internal_profile = m_profile->internal();
    auto& kcol = internal_profile->getKCol();
    auto& block_row_offset = internal_profile->getBlockRowOffset();

    auto local_row_size = m_profile->localRowSize();
    if (local_row_size == nullptr) {
      //ValueBufferType values_buffer(m_h_csr_values.data(), sycl::range<1>(nnz));
      // COMPUTE COLS
      // clang-format off
        queue.submit([&](sycl::handler& cgh)
                     {
                       auto access_kcol_buffer      = internal_profile->getKCol().template get_access<sycl::access::mode::read>(cgh);
                       auto access_block_row_offset = internal_profile->getBlockRowOffset().template get_access<sycl::access::mode::read>(cgh);
                       auto access_values_buffer    = values_buffer.template get_access<sycl::access::mode::read>(cgh);
                       auto access_block_values     = m_values.template get_access<sycl::access::mode::read_write>(cgh);

                       //cgh.parallel_for<class vector_axpy>(sycl::nd_range<1>{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}},[=](sycl::nd_item<1> item_id)
                       cgh.parallel_for<class set_matrix_values4>(sycl::range<1>{total_threads},
                                                             [=] (sycl::item<1> item_id)
                                                             {
                                                               auto id = item_id.get_id(0);
                                                               //auto local_id  = item_id.get_local_id(0);
                                                               //auto block_id  = item_id.get_group(0) ;
                                                               //auto global_id = item_id.get_global_id(0);

                                                               for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                               {
                                                                  auto block_id = i/ellpack_size ;
                                                                  auto local_id = i%ellpack_size ;

                                                                  auto begin              = access_kcol_buffer[i] ;
                                                                  auto end                = access_kcol_buffer[i+1] ;
                                                                  auto row_size           = end - begin ;

                                                                  int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                                  auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                  for(int k=begin;k<end;++k)
                                                                  {
                                                                    access_block_values[block_row_offset+(k-begin)*ellpack_size+local_id] = access_values_buffer[k] ;
                                                                  }
                                                                  for(int k=row_size;k<block_row_size;++k)
                                                                  {
                                                                    access_block_values[block_row_offset+k*ellpack_size+local_id] = 0 ;
                                                                  }
                                                               }
                                                            });
                     }) ;
      // clang-format on
      m_values_is_update = true;
    }
    else
    {
      //ValueBufferType values_buffer(m_h_csr_values.data(), sycl::range<1>(nnz));
      IndexBufferType lrowsize_buffer(local_row_size, sycl::range<1>(nrows));

      // COMPUTE COLS
      // clang-format off
      queue.submit([&](sycl::handler& cgh)
                   {
                     auto access_kcol_buffer      = internal_profile->getKCol().template get_access<sycl::access::mode::read>(cgh);
                     auto access_lrowsize_buffer  = lrowsize_buffer.template get_access<sycl::access::mode::read>(cgh);

                     auto access_block_row_offset = internal_profile->getBlockRowOffset().template get_access<sycl::access::mode::read>(cgh);
                     auto access_values_buffer    = values_buffer.template get_access<sycl::access::mode::read>(cgh);
                     auto access_block_values     = m_values.template get_access<sycl::access::mode::read_write>(cgh);

                     //cgh.parallel_for<class vector_axpy>(sycl::nd_range<1>{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}},[=](sycl::nd_item<1> item_id)
                     cgh.parallel_for<class set_matrix_values5>(sycl::range<1>{total_threads},
                                                           [=] (sycl::item<1> item_id)
                                                           {
                                                             auto id = item_id.get_id(0);
                                                             //auto local_id  = item_id.get_local_id(0);
                                                             //auto block_id  = item_id.get_group(0) ;
                                                             //auto global_id = item_id.get_global_id(0);

                                                             for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                             {
                                                                auto block_id = i/ellpack_size ;
                                                                auto local_id = i%ellpack_size ;

                                                                auto begin              = access_kcol_buffer[i] ;
                                                                auto lrow_size          = access_lrowsize_buffer[i] ;
                                                                auto end                = begin + lrow_size ;

                                                                int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                                auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                for(int k=begin;k<end;++k)
                                                                {
                                                                  access_block_values[block_row_offset+(k-begin)*ellpack_size+local_id] = access_values_buffer[k] ;
                                                                }
                                                                for(int k=lrow_size;k<block_row_size;++k)
                                                                {
                                                                  access_block_values[block_row_offset+k*ellpack_size+local_id] = 0 ;
                                                                }
                                                             }
                                                          });
                   }) ;
      auto kcol            = m_profile->kcol() ;
      auto local_row_size  = m_profile->localRowSize() ;

      auto interface_nrows = m_ext_profile->getNRows();
      auto ext_nnz         = m_ext_profile->getNnz();
      auto ext_block_nnz   = m_ext_profile->getBlockNnz();

      auto ext_internal_profile  = m_ext_profile->internal();
      auto& ext_kcol             = ext_internal_profile->getKCol();
      auto& ext_block_row_offset = ext_internal_profile->getBlockRowOffset();
      //m_h_ext_values.resize(ext_block_nnz) ;
      m_ext_values.reset(new ValueBufferType(ext_block_nnz * ellpack_size)) ;

      // EXTRACT EXTERNAL PROFILE
      {
        ValueBufferType ext_csr_values_buffer(m_h_csr_ext_values.data(), sycl::range<1>(ext_nnz));

        std::vector<Integer> h_local_row_offset(interface_nrows+1) ;
        {
          Integer offset = 0 ;
          for(std::size_t i=0;i<interface_nrows;++i)
          {
            h_local_row_offset[i] = offset ;
            offset += local_row_size[i] ;
          }
          h_local_row_offset[interface_nrows] = offset ;
        }
        IndexBufferType local_row_offset(h_local_row_offset.data(),sycl::range<1>(interface_nrows+1)) ;

        queue.submit([&](sycl::handler& cgh)
                     {
                        auto access_kcol              = internal_profile->getKCol().template get_access<sycl::access::mode::read>(cgh);
                        auto access_interface_row_ids = m_interface_row_ids->template get_access<sycl::access::mode::read>(cgh);
                        auto access_local_row_size    = lrowsize_buffer.template get_access<sycl::access::mode::read>(cgh);
                        auto access_local_row_offset  = local_row_offset.template get_access<sycl::access::mode::read>(cgh);
                        auto access_csr_values        = values_buffer.template get_access<sycl::access::mode::read>(cgh);
                        auto access_ext_csr_values    = ext_csr_values_buffer.template get_access<sycl::access::mode::read_write>(cgh);
                        cgh.parallel_for<class set_matrix_values6>(sycl::range<1>{total_threads},
                                                                     [=] (sycl::item<1> item_id)
                                                                     {
                                                                        auto id = item_id.get_id(0);
                                                                        for (auto i = id; i < interface_nrows; i += item_id.get_range()[0])
                                                                        {
                                                                            Integer jcol = access_local_row_offset[i] ;
                                                                            for (int k = access_kcol[i] + access_local_row_size[i]; k < access_kcol[i + 1]; ++k)
                                                                                access_ext_csr_values[jcol++] = access_csr_values[k];
                                                                        }
                                                                     });
                     }).wait() ;

        queue.submit([&](sycl::handler& cgh)
                     {
                       auto access_kcol_buffer      = ext_kcol.template get_access<sycl::access::mode::read>(cgh);
                       auto access_block_row_offset = ext_block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                       auto access_values_buffer    = ext_csr_values_buffer.template get_access<sycl::access::mode::read>(cgh);
                       //auto access_block_values     = m_ext_values->template get_access<sycl::access::mode::read_write>(cgh);
                       auto access_block_values     = sycl::accessor { *m_ext_values, cgh, sycl::write_only, sycl::property::no_init{}};

                       cgh.parallel_for<class set_matrix_values7>(sycl::range<1>{total_threads},
                                                                   [=] (sycl::item<1> item_id)
                                                                   {
                                                                     auto id = item_id.get_id(0);

                                                                     for (auto i = id; i < interface_nrows; i += item_id.get_range()[0])
                                                                     {
                                                                        auto block_id = i/ellpack_size ;
                                                                        auto local_id = i%ellpack_size ;
                                                                        auto begin              = access_kcol_buffer[i] ;
                                                                        auto end                = access_kcol_buffer[i+1] ;
                                                                        auto row_size           = end - begin ;

                                                                        int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                                        auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                        for(int k=begin;k<end;++k)
                                                                        {
                                                                          access_block_values[block_row_offset+(k-begin)*ellpack_size+local_id] = access_values_buffer[k] ;
                                                                        }
                                                                        for(int k=row_size;k<block_row_size;++k)
                                                                        {
                                                                          access_block_values[block_row_offset+k*ellpack_size+local_id] = 0 ;
                                                                        }
                                                                     }
                                                                  });
                       }).wait() ;
        // clang-format on
      }
      m_values_is_update = true;
    }
    return true;
  }


  template <typename ValueT, int EllpackSize>
  bool MatrixInternal<ValueT, EllpackSize>::needUpdate()
  {
    return m_values_is_update != true;
  }

  template <typename ValueT, int EllpackSize>
  void MatrixInternal<ValueT, EllpackSize>::notifyChanges()
  {
    m_values_is_update = false;
  }

  template <typename ValueT, int EllpackSize>
  void MatrixInternal<ValueT, EllpackSize>::endUpdate()
  {
    if (not m_values_is_update) {
      setMatrixValuesFromHost();
    }
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::mult(ValueBufferType& x, ValueBufferType& y, sycl::queue& queue) const
  {

    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * ellpack_size;

    // clang-format off
    int N                  = this->m_N;
    int NxN                = this->m_NxN;
    int pack_size          = ellpack_size;
    auto nrows             = m_profile->getNRows();
    auto nnz               = m_profile->getNnz();

    auto internal_profile  = m_profile->internal();
    auto& kcol             = internal_profile->getKCol();
    auto& block_row_offset = internal_profile->getBlockRowOffset();
    auto& block_cols       = internal_profile->getBlockCols();
    // clang-format on
    if(N==1)
    {
      // COMPUTE VALUES
      // clang-format off
        queue.submit([&](sycl::handler& cgh)
                 {
                   auto access_block_row_offset = block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                   auto access_cols             = block_cols.template get_access<sycl::access::mode::read>(cgh);
                   auto access_values           = m_values.template get_access<sycl::access::mode::read>(cgh);


                   auto access_x                = x.template get_access<sycl::access::mode::read>(cgh);
                   auto access_y                = y.template get_access<sycl::access::mode::read_write>(cgh);


                   //sycl::nd_range<1> r{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}};
                   //cgh.parallel_for<class compute_mult>(r, [&](sycl::nd_item<1> item_id)
                   cgh.parallel_for<class compute_mult>(sycl::range<1>{total_threads},
                                                        [=] (sycl::item<1> item_id)
                                                        {
                                                          auto id = item_id.get_id(0);
                                                          //auto local_id  = item_id.get_local_id(0);
                                                          //auto block_id  = item_id.get_group(0) ;
                                                          //auto global_id = item_id.get_global_id(0);

                                                          for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                          {
                                                             auto block_id = i/ellpack_size ;
                                                             auto local_id = i%ellpack_size ;

                                                             int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                             auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                             ValueType value = 0. ;
                                                             for(int j=0;j<block_row_size;++j)
                                                             {
                                                               auto k = block_row_offset+j*ellpack_size+local_id ;
                                                                value += access_values[k]* access_x[access_cols[k]] ;
                                                             }
                                                             access_y[i] = value ;
                                                          }
                                                      });
                 });
      // clang-format on
    }
    else
    {
      //size_t max_local_mem = device.get_info<sycl::info::device::local_mem_size>();
      //std::cout<<"MAX LOCAL MEMORY SIZE       : "<<max_local_mem<<" Bytes"<<std::endl ;
      //std::cout<<"MAX NB DOUBLES IN LOCAL MEM : "<<max_local_mem/sizeof(ValueT)<<std::endl ;
      queue.submit(
         [&](sycl::handler& cgh)
         {
           auto access_block_row_offset = block_row_offset.template get_access<sycl::access::mode::read>(cgh);
           auto access_cols             = block_cols.template get_access<sycl::access::mode::read>(cgh);
           auto access_values           = m_values.template get_access<sycl::access::mode::read>(cgh);


           auto access_x                = x.template get_access<sycl::access::mode::read>(cgh);
           auto access_y                = y.template get_access<sycl::access::mode::discard_write>(cgh);

           //sycl::local_accessor<ValueT,1> local_mem(sycl::range<1>(ellpack_size*N), cgh);
           //sycl::nd_range<1> range{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}};
           sycl::range<1> range{total_threads} ;
           auto tile = Tile(N) ;
           //cgh.parallel_for<class compute_mult>(r, [&](sycl::nd_item<1> item_id)
           cgh.parallel_for<class compute_block_mult>(range,
                [=] (sycl::item<1> item_id /*sycl::nd_item<1> item_id*/)
                {
                  auto id        = item_id.get_id(0);
                  //auto local_id  = item_id.get_local_id(0);
                  //auto block_id  = item_id.get_group(0) ;
                  //auto global_id = item_id.get_global_id(0);



                  for (auto i = id; i < nrows; i += item_id.get_range()[0] /*item_id.get_local_range(0)*/)
                  {
                     auto block_id = i/pack_size ;
                     auto local_id = i%pack_size ;

                     std::size_t block_row_offset   = access_block_row_offset[block_id] ;
                     auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                     for(int ieq=0;ieq<N;++ieq)
                     {
                       ValueType value = 0. ;
                       for(std::size_t k=block_row_offset;k<block_row_offset+block_row_size;++k)
                       {
                         // value += access_values[k]* access_x[access_cols[k]] ;
                         value += tile.mult(ieq,
                                            local_id,
                                            k,
                                            access_cols,
                                            access_values,
                                            access_x) ;
                       }
                       access_y[i*N+ieq] = value;
                     }
                  }
              });
           });
      /*
      {
        sycl::host_accessor<ValueT, 1, sycl::access::mode::read> x_acc(x);
        for(int il=0;il<nrows;++il)
        {
          std::cout<<"X["<<il<<"]:";
          for(int i=0;i<N;++i)
              std::cout<<x_acc[il*N+i]<<",";
          std::cout<<std::endl;
        }
        sycl::host_accessor<ValueT, 1, sycl::access::mode::read> y_acc(y);
        for(int il=0;il<nrows;++il)
        {
          std::cout<<"Y["<<il<<"]:";
          for(int i=0;i<N;++i)
              std::cout<<y_acc[il*N+i]<<",";
          std::cout<<std::endl;
        }

        auto tile = Tile(N) ;
        sycl::host_accessor<ValueT, 1, sycl::access::mode::read> matrix_acc(m_values);
        sycl::host_accessor<int, 1, sycl::access::mode::read> cols_acc(block_cols);
        sycl::host_accessor<int, 1, sycl::access::mode::read> brow_offset_acc(block_row_offset);
        for(std::size_t irow=0;irow<nrows;++irow)
        {
          auto ib = irow/ellpack_size ;
          auto il = irow%ellpack_size ;
          for(int ieq=0;ieq<N;++ieq)
          {
            std::cout<<"LINE["<<il<<","<<ieq<<"] : ";
            ValueType value = 0. ;
            for(std::size_t k=brow_offset_acc[ib];k<brow_offset_acc[ib+1];++k)
            {
              // value += access_values[k]* access_x[access_cols[k]] ;
              value += tile.mult(ieq,
                                 il,
                                 k,
                                 cols_acc,
                                 matrix_acc,
                                 x_acc) ;
            }
            std::cout<<"\nY_CPU["<<irow<<","<<ieq<<"]:"<<value<<std::endl;
          }
        }
      }*/
    // clang-format on
    }
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::mult(ValueBufferType& x, ValueBufferType& y) const
  {
    this->mult(x, y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::addExtMult(ValueBufferType& x,
                                                     ValueBufferType& y,
                                                     sycl::queue& queue) const
  {
    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * ellpack_size;

    // clang-format off
    auto nrows             = m_profile->getNRows();
    auto nnz               = m_profile->getNnz();

    auto interface_nrow    = m_ext_profile->getNRows();
    auto ext_profile       = m_ext_profile->internal();
    auto& kcol             = ext_profile->getKCol();
    auto& block_row_offset = ext_profile->getBlockRowOffset();
    auto& block_cols       = ext_profile->getBlockCols();
    // clang-format on

    {
      // clang-format off
      queue.submit([&](sycl::handler& cgh)
                   {
                     auto access_block_row_offset = block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                     auto access_cols             = block_cols.template get_access<sycl::access::mode::read>(cgh);
                     auto access_values           = m_ext_values->template get_access<sycl::access::mode::read>(cgh);

                     auto access_row_ids          = m_interface_row_ids->template get_access<sycl::access::mode::read>(cgh);

                     auto access_x                = x.template get_access<sycl::access::mode::read>(cgh);
                     auto access_y                = y.template get_access<sycl::access::mode::read_write>(cgh);

                     cgh.parallel_for<class compute_ext_mult>(sycl::range<1>{total_threads},
                                                              [=] (sycl::item<1> item_id)
                                                              {
                                                                auto id = item_id.get_id(0);

                                                                for (auto i = id; i < interface_nrow; i += item_id.get_range()[0])
                                                                {
                                                                   auto block_id = i/ellpack_size ;
                                                                   auto local_id = i%ellpack_size ;

                                                                   int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                                   auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                   ValueType value = 0. ;
                                                                   for(int j=0;j<block_row_size;++j)
                                                                   {
                                                                     auto k = block_row_offset+j*ellpack_size+local_id ;
                                                                     value += access_values[k]* access_x[access_cols[k]] ;
                                                                   }
                                                                   access_y[access_row_ids[i]] += value ;
                                                                }
                                                            });
                   });
      // clang-format on
    }
  }
  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::addExtMult(ValueBufferType& x,
                                                     ValueBufferType& y) const
  {
    this->addExtMult(x, y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::addLMult(ValueType alpha,
                                                   ValueBufferType& x,
                                                   ValueBufferType& y,
                                                   sycl::queue& queue) const
  {
    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * ellpack_size;

    // clang-format off
    auto nrows             = m_profile->getNRows();
    auto nnz               = m_profile->getNnz();

    auto internal_profile  = m_profile->internal();
    auto& kcol             = internal_profile->getKCol();
    auto& block_row_offset = internal_profile->getBlockRowOffset();
    auto& block_cols       = internal_profile->getBlockCols();

    auto& mask             = internal_profile->getLowerMask();
    // clang-format on
    {
      // clang-format off
        queue.submit([&](sycl::handler& cgh)
                 {
                   auto access_block_row_offset = block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                   auto access_cols             = block_cols.template get_access<sycl::access::mode::read>(cgh);
                   auto access_mask             = mask.template get_access<sycl::access::mode::read>(cgh);
                   auto access_values           = m_values.template get_access<sycl::access::mode::read>(cgh);


                   auto access_x                = x.template get_access<sycl::access::mode::read>(cgh);
                   auto access_y                = y.template get_access<sycl::access::mode::read_write>(cgh);


                   //sycl::nd_range<1> r{sycl::range<1>{total_threads},sycl::range<1>{ellpack_size}};
                   //cgh.parallel_for<class compute_mult>(r, [&](sycl::nd_item<1> item_id)
                   cgh.parallel_for<class compute_lmult>(sycl::range<1>{total_threads},
                                                        [=] (sycl::item<1> item_id)
                                                        {
                                                          auto id = item_id.get_id(0);
                                                          //auto local_id  = item_id.get_local_id(0);
                                                          //auto block_id  = item_id.get_group(0) ;
                                                          //auto global_id = item_id.get_global_id(0);

                                                          for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                          {
                                                             auto block_id = i/ellpack_size ;
                                                             auto local_id = i%ellpack_size ;

                                                             int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                             auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                             ValueType value = access_y[i] ;
                                                             for(int j=0;j<block_row_size;++j)
                                                             {
                                                               auto k = block_row_offset+j*ellpack_size+local_id ;
                                                                value += alpha * access_mask[k] * access_values[k]* access_x[access_cols[k]] ;
                                                             }
                                                             access_y[i] = value ;
                                                          }
                                                      });
                 });
      // clang-format on
    }
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::addLMult(ValueType alpha, ValueBufferType& x, ValueBufferType& y) const
  {
    this->addLMult(alpha, x, y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::addUMult(ValueType alpha,
                                                   ValueBufferType& x,
                                                   ValueBufferType& y,
                                                   sycl::queue& queue) const
  {

    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * ellpack_size;

    // clang-format off
      auto nrows = m_profile->getNRows() ;
      auto nnz   = m_profile->getNnz() ;

      auto internal_profile  = m_profile->internal() ;
      auto& kcol             = internal_profile->getKCol() ;
      auto& block_row_offset = internal_profile->getBlockRowOffset() ;
      auto& block_cols       = internal_profile->getBlockCols() ;
      auto& mask             = internal_profile->getUpperMask() ;
      {
        // COMPUTE VALUES
        queue.submit([&](sycl::handler& cgh)
                 {
                   auto access_block_row_offset = block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                   auto access_cols             = block_cols.template get_access<sycl::access::mode::read>(cgh);
                   auto access_mask             = mask.template get_access<sycl::access::mode::read>(cgh);
                   auto access_values           = m_values.template get_access<sycl::access::mode::read>(cgh);


                   auto access_x                = x.template get_access<sycl::access::mode::read>(cgh);
                   auto access_y                = y.template get_access<sycl::access::mode::read_write>(cgh);

                   cgh.parallel_for<class compute_umult>(sycl::range<1>{total_threads},
                                                        [=] (sycl::item<1> item_id)
                                                        {
                                                          auto id = item_id.get_id(0);
                                                          //auto local_id  = item_id.get_local_id(0);
                                                          //auto block_id  = item_id.get_group(0) ;
                                                          //auto global_id = item_id.get_global_id(0);

                                                          for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                          {
                                                             auto block_id = i/ellpack_size ;
                                                             auto local_id = i%ellpack_size ;

                                                             int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                             auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                             ValueType value = access_y[i] ;
                                                             for(int j=0;j<block_row_size;++j)
                                                             {
                                                               auto k = block_row_offset+j*ellpack_size+local_id ;
                                                                value += alpha * access_mask[k] * access_values[k]* access_x[access_cols[k]] ;
                                                             }
                                                             access_y[i] = value ;
                                                          }
                                                      });
                 });
      }
    // clang-format on
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::addUMult(ValueType alpha, ValueBufferType& x, ValueBufferType& y) const
  {
    this->addUMult(alpha, x, y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::multInvDiag(ValueBufferType& y, sycl::queue& queue) const
  {
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::multInvDiag(ValueBufferType& y) const
  {
    this->multInvDiag(y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::computeInvDiag(ValueBufferType& y, sycl::queue& queue) const
  {

    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * ellpack_size;

    // clang-format off
    int N      = this->m_N;
    int NxN    = this->m_NxN;
    int pack_size = ellpack_size ;
    auto nrows = m_profile->getNRows() ;
    auto nnz   = m_profile->getNnz() ;

    auto internal_profile  = m_profile->internal() ;
    auto& kcol             = internal_profile->getKCol() ;
    auto& block_row_offset = internal_profile->getBlockRowOffset() ;
    auto& block_cols       = internal_profile->getBlockCols() ;

    if(N==1)
    {
      // COMPUTE VALUES
      queue.submit([&](sycl::handler& cgh)
                   {
                     auto access_block_row_offset = block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                     auto access_cols             = block_cols.template get_access<sycl::access::mode::read>(cgh);
                     auto access_values           = m_values.template get_access<sycl::access::mode::read>(cgh);
                     auto access_y                = y.template get_access<sycl::access::mode::read_write>(cgh);


                     cgh.parallel_for<class compute_inv_diag>(sycl::range<1>{total_threads},
                                                          [=] (sycl::item<1> item_id)
                                                          {
                                                            auto id = item_id.get_id(0);
                                                            //auto local_id  = item_id.get_local_id(0);
                                                            //auto block_id  = item_id.get_group(0) ;
                                                            //auto global_id = item_id.get_global_id(0);

                                                            for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                            {
                                                               auto block_id = i/ellpack_size ;
                                                               auto local_id = i%ellpack_size ;

                                                               int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                               auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;
                                                               for(int j=0;j<block_row_size;++j)
                                                               {
                                                                 auto k = block_row_offset+j*ellpack_size+local_id ;
                                                                 if((access_cols[k])==int(i) && (access_values[k]!=0) )
                                                                   access_y[i] = 1./access_values[k] ;
                                                               }
                                                            }
                                                          });
                   });
    }
    else
    {
      // COMPUTE VALUES
      queue.submit([&](sycl::handler& cgh)
                   {
                     auto access_block_row_offset = block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                     auto access_cols             = block_cols.template get_access<sycl::access::mode::read>(cgh);
                     auto access_values           = m_values.template get_access<sycl::access::mode::read>(cgh);
                     auto access_y                = y.template get_access<sycl::access::mode::read_write>(cgh);


                     cgh.parallel_for<class compute_inv_diag>(sycl::range<1>{total_threads},
                                                          [=] (sycl::item<1> item_id)
                                                          {
                                                            auto id = item_id.get_id(0);
                                                            //auto local_id  = item_id.get_local_id(0);
                                                            //auto block_id  = item_id.get_group(0) ;
                                                            //auto global_id = item_id.get_global_id(0);

                                                            for (std::size_t i = id; i < nrows; i += item_id.get_range()[0])
                                                            {
                                                               auto block_id = i/pack_size ;
                                                               auto local_id = i%pack_size ;

                                                               int block_row_offset   = access_block_row_offset[block_id] ;
                                                               auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;
                                                               for(int k=0;k<block_row_size;++k)
                                                               {
                                                                 auto kcol = (block_row_offset+k)*ellpack_size+local_id ;
                                                                 std::size_t col = access_cols[kcol] ;
                                                                 if(col==i)
                                                                 {
                                                                   for(int ieq=0;ieq<N;++ieq)
                                                                   {
                                                                     auto kval = (block_row_offset+k)*NxN + ieq*N + ieq ;
                                                                     auto diag = access_values[kval*ellpack_size+local_id] ;
                                                                     if(diag!=0)
                                                                       access_y[i*N+ieq] = 1./diag;
                                                                     else
                                                                       access_y[i*N+ieq] = 1.;
                                                                   }
                                                                 }
                                                               }
                                                            }
                                                          });
                   });
      /*
      sycl::host_accessor<ValueT, 1, sycl::access::mode::read> h_acc(y);
      sycl::host_accessor<ValueT, 1, sycl::access::mode::read> mat_acc(m_values);
      sycl::host_accessor<int, 1, sycl::access::mode::read> kcol_acc(block_row_offset);
      sycl::host_accessor<int, 1, sycl::access::mode::read> cols_acc(block_cols);
      for(int irow=0;irow<nrows;++irow)
      {
        auto ib=irow/ellpack_size ;
        auto il=irow%ellpack_size ;
        std::cout<<"MAT["<<irow<<","<<ib<<"]:";
        for(int k=kcol_acc[ib];k<kcol_acc[ib+1];++k)
        {
            std::cout<<"("<<cols_acc[k*ellpack_size+il]<<",";
            for(int i=0;i<N;++i)
              for(int j=0;j<N;++j)
                std::cout<<mat_acc[(k*NxN+i*N+j)*ellpack_size+il]<<",";
            std::cout<<")"<<std::endl;
        }
      }
      for(int irow=0;irow<nrows;++irow)
      {
        auto ib=irow/ellpack_size ;
        auto il=irow%ellpack_size ;
        std::cout<<"DIAG["<<irow<<","<<il<<"]:";
        for(int i=0;i<N;++i)
            std::cout<<h_acc[il*N+i]<<",";
        std::cout<<std::endl;
      }*/
    }
    // clang-format on
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::computeInvDiag(ValueBufferType& y) const
  {
    this->computeInvDiag(y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::scal(ValueBufferType& y, sycl::queue& queue)
  {

    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * ellpack_size;

    // clang-format off
    auto nrows = m_profile->getNRows() ;
    auto nnz   = m_profile->getNnz() ;

    auto internal_profile  = m_profile->internal() ;
    auto& kcol             = internal_profile->getKCol() ;
    auto& block_row_offset = internal_profile->getBlockRowOffset() ;
    auto& block_cols       = internal_profile->getBlockCols() ;
    {
      // COMPUTE VALUES
      queue.submit([&](sycl::handler& cgh)
                   {
                     auto access_block_row_offset = block_row_offset.template get_access<sycl::access::mode::read>(cgh);
                     auto access_cols             = block_cols.template get_access<sycl::access::mode::read>(cgh);
                     auto access_values           = m_values.template get_access<sycl::access::mode::read_write>(cgh);
                     auto access_y                = y.template get_access<sycl::access::mode::read>(cgh);


                     cgh.parallel_for<class scal_matrix>(sycl::range<1>{total_threads},
                                                          [=] (sycl::item<1> item_id)
                                                          {
                                                            auto id = item_id.get_id(0);
                                                            //auto local_id  = item_id.get_local_id(0);
                                                            //auto block_id  = item_id.get_group(0) ;
                                                            //auto global_id = item_id.get_global_id(0);

                                                            for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                            {
                                                               auto block_id = i/ellpack_size ;
                                                               auto local_id = i%ellpack_size ;

                                                               int block_row_offset   = access_block_row_offset[block_id]*ellpack_size ;
                                                               auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;
                                                               for(int j=0;j<block_row_size;++j)
                                                               {
                                                                 auto k = block_row_offset+j*ellpack_size+local_id ;
                                                                 access_values[k] *= access_y[i] ;
                                                               }
                                                            }
                                                          });
                   });
    }
    // clang-format on
  }

  template <typename ValueT, int EllPackSize>
  void MatrixInternal<ValueT, EllPackSize>::scal(ValueBufferType& y)
  {
    this->scal(y, SYCLEnv::instance()->internal()->queue());
  }
} // namespace SYCLInternal




template <typename ValueT>
SYCLBEllPackMatrix<ValueT>::SYCLBEllPackMatrix()
: IMatrixImpl(nullptr, AlgebraTraits<BackEnd::tag::sycl>::name())
, m_send_policy(SimpleCSRInternal::CommProperty::ASynch)
, m_recv_policy(SimpleCSRInternal::CommProperty::ASynch)
{}

template <typename ValueT>
SYCLBEllPackMatrix<ValueT>::SYCLBEllPackMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::sycl>::name())
, m_send_policy(SimpleCSRInternal::CommProperty::ASynch)
, m_recv_policy(SimpleCSRInternal::CommProperty::ASynch)
{}

template <typename ValueT>
SYCLBEllPackMatrix<ValueT>::~SYCLBEllPackMatrix()
{
#ifdef ALIEN_USE_PERF_TIMER
  m_timer.printInfo("SYCLBELLPACK-MATRIX");
#endif
}

template <typename ValueT>
bool SYCLBEllPackMatrix<ValueT>::
initMatrix(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
           Integer local_offset,
           Integer global_size,
           std::size_t nrows,
           int const* kcol,
           int const* cols,
           SimpleCSRInternal::DistStructInfo const& matrix_dist_info,
           int block_size)
{
  m_nproc = 1;
  m_myrank = 0;
  m_parallel_mng = parallel_mng;
  if (parallel_mng) {
    m_nproc = m_parallel_mng->commSize();
    m_myrank = m_parallel_mng->commRank();
  }
  m_is_parallel = (m_nproc > 1);

  // clang-format off
  m_local_offset = 0;
  m_local_size   = nrows;
  m_global_size  = m_local_size;
  m_ghost_size   = 0;
  // clang-format on

  m_matrix_dist_info.copy(matrix_dist_info);

  m_ellpack_size = 1024;
  if (m_nproc > 1) {
    UniqueArray<Integer> offset(m_nproc + 1);
    Arccore::MessagePassing::mpAllGather(m_parallel_mng,
                                         ConstArrayView<Integer>(1, &m_local_offset), offset.subView(0, m_nproc));

    offset[m_nproc] = m_global_size;

    m_local_offset = offset[m_myrank];
    m_global_size = offset[m_nproc];
    m_ghost_size = m_matrix_dist_info.m_ghost_nrow;

    auto& local_row_size = m_matrix_dist_info.m_local_row_size;
    auto sorted_cols = m_matrix_dist_info.m_cols.data();

    std::size_t block_nrows = ProfileInternal1024::nbBlocks(nrows);

    ProfileInternal1024::computeBlockRowOffset(m_block_row_offset, nrows, kcol);

    // clang-format off
    alien_debug([&] {
                      cout() << "NROWS  = "<<nrows;
                      cout() << "NNZ    = "<<kcol[nrows];
                      cout() << "BNROWS = "<<block_nrows;
                      cout() << "BNNZ   = "<<m_block_row_offset[block_nrows];
                    });
    // clang-format on
    //Universe().traceMng()->flush();
    m_profile1024.reset( new ProfileInternal1024{ nrows,
                                                  kcol,
                                                  sorted_cols,
                                                  m_block_row_offset.data(),
                                                  local_row_size.data() });

    m_matrix1024.reset(new MatrixInternal1024{ m_profile1024.get(), block_size });

    // EXTRACT EXTERNAL PROFILE
    std::size_t interface_nrows = m_matrix_dist_info.m_interface_nrow;
    std::vector<int> ext_kcol(interface_nrows + 1);

    int ext_nnz = 0;
    for (std::size_t i = 0; i < interface_nrows; ++i) {
      auto id = m_matrix_dist_info.m_interface_rows[i];
      ext_kcol[i] = ext_nnz;
      ext_nnz += kcol[id + 1] - kcol[id] - local_row_size[id];
    }
    ext_kcol[interface_nrows] = ext_nnz;

    std::vector<int> ext_cols(ext_nnz);
    {
      int jcol = 0;
      for (std::size_t i = 0; i < interface_nrows; ++i) {
        auto id = m_matrix_dist_info.m_interface_rows[i];
        for (int k = kcol[id] + local_row_size[id]; k < kcol[id + 1]; ++k) {
          ext_cols[jcol++] = (int)(sorted_cols[k] - nrows);
        }
      }
    }

    std::size_t ext_block_nrows = ProfileInternal1024::nbBlocks(interface_nrows);
    ProfileInternal1024::computeBlockRowOffset(m_ext_block_row_offset, interface_nrows, ext_kcol.data());
    // clang-format off
    alien_debug([&] {
                      cout() << "EXT NROWS  = "<<interface_nrows;
                      cout() << "EXT NNZ    = "<<ext_kcol[interface_nrows];
                      cout() << "EXT BNROWS = "<<ext_block_nrows;
                      cout() << "EXT BNNZ   = "<<m_block_row_offset[ext_block_nrows];
                    });
    // clang-format on

    m_ext_profile1024.reset( new ProfileInternal1024{ interface_nrows,
                                                      ext_kcol.data(),
                                                      ext_cols.data(),
                                                      m_ext_block_row_offset.data(),
                                                      nullptr });

    m_matrix1024->m_ext_profile = m_ext_profile1024.get();
    typedef typename MatrixInternal1024::IndexBufferType IndexBufferType;
    m_matrix1024->m_h_interface_row_ids = m_matrix_dist_info.m_interface_rows.data();
    m_matrix1024->m_interface_row_ids.reset(new IndexBufferType{ m_matrix_dist_info.m_interface_rows.data(),
                                                                 m_matrix_dist_info.m_interface_rows.size() });

    m_matrix1024->m_send_ids.reset(new IndexBufferType{ m_matrix_dist_info.m_send_info.m_ids.data(),
                                                        m_matrix_dist_info.m_send_info.m_ids.size() });
    m_matrix1024->m_recv_ids.reset(new IndexBufferType{ m_matrix_dist_info.m_recv_info.m_ids.data(),
                                                        m_matrix_dist_info.m_recv_info.m_ids.size() });
  }
  else {

    std::size_t block_nrows = ProfileInternal1024::nbBlocks(nrows);

    ProfileInternal1024::computeBlockRowOffset(m_block_row_offset, nrows, kcol);

    // clang-format off
    alien_debug([&] {
                      cout() << "NROWS  = "<<nrows;
                      cout() << "NNZ    = "<<kcol[nrows];
                      cout() << "BNROWS = "<<block_nrows;
                      cout() << "BNNZ   = "<<m_block_row_offset[block_nrows];
                    });
    // clang-format on

    m_profile1024.reset( new ProfileInternal1024{ nrows,
                                                  kcol,
                                                  cols,
                                                  m_block_row_offset.data(),
                                                  nullptr });

    m_matrix1024.reset( new MatrixInternal1024{ m_profile1024.get(), block_size });
  }

  return true;
}

template <typename ValueT>
SYCLBEllPackMatrix<ValueT>*
SYCLBEllPackMatrix<ValueT>::cloneTo(const MultiMatrixImpl* multi) const
{
  SYCLBEllPackMatrix<ValueT>* matrix = new SYCLBEllPackMatrix<ValueT>(multi);
  matrix->initMatrix(m_parallel_mng,
                     m_local_offset,
                     m_global_size,
                     m_profile1024->getNRows(),
                     m_profile1024->kcol(),
                     m_profile1024->cols(),
                     m_matrix_dist_info);
  matrix->setMatrixValues(getAddressData(), true);
  return matrix;
}

template <typename ValueT>
bool SYCLBEllPackMatrix<ValueT>::setMatrixValues(Arccore::Real const* values, bool only_host)
{
  return m_matrix1024->setMatrixValues(values, only_host);
}

template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::notifyChanges()
{
  if (m_matrix1024.get())
    m_matrix1024->notifyChanges();
}

template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::endUpdate()
{
  if (m_matrix1024.get() && m_matrix1024->needUpdate()) {
    m_matrix1024->endUpdate();
    this->updateTimestamp();
  }
}

template <typename ValueT>
ValueT const* SYCLBEllPackMatrix<ValueT>::getAddressData() const
{
  return m_matrix1024->getHCsrData();
}

template <typename ValueT>
ValueT const* SYCLBEllPackMatrix<ValueT>::data() const
{
  return m_matrix1024->getHCsrData();
}

template <typename ValueT>
ValueT* SYCLBEllPackMatrix<ValueT>::getAddressData()
{
  return m_matrix1024->getHCsrData();
}

template <typename ValueT>
ValueT* SYCLBEllPackMatrix<ValueT>::data()
{
  return m_matrix1024->getHCsrData();
}

template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::mult(SYCLVector<ValueT> const& x, SYCLVector<ValueT>& y) const
{
  return m_matrix1024->mult(x.internal()->values(), y.internal()->values());
}

template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::endDistMult(SYCLVector<ValueT> const& x, SYCLVector<ValueT>& y) const
{
  m_matrix1024->addExtMult(x.internal()->ghostValues(getGhostSize()), y.internal()->values());
}

template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::addLMult(ValueT alpha, SYCLVector<ValueT> const& x, SYCLVector<ValueT>& y) const
{
  m_profile1024->dcol();
  return m_matrix1024->addLMult(alpha, x.internal()->values(), y.internal()->values());
}

template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::addUMult(ValueT alpha, SYCLVector<ValueT> const& x, SYCLVector<ValueT>& y) const
{
  m_profile1024->dcol();
  return m_matrix1024->addUMult(alpha, x.internal()->values(), y.internal()->values());
}

template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::multInvDiag(SYCLVector<ValueType>& y) const
{
  return m_matrix1024->multInvDiag(y.internal()->values());
}

template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::computeInvDiag(SYCLVector<ValueType>& y) const
{
  return m_matrix1024->computeInvDiag(y.internal()->values());
}

template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::scal(SYCLVector<ValueType> const& diag)
{
  return m_matrix1024->scal(diag.internal()->values());
}


template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::copy(SYCLBEllPackMatrix<ValueT> const& matrix)
{
  auto block_size = blockSize() ;
  initMatrix(matrix.m_parallel_mng,
             matrix.m_local_offset,
             matrix.m_global_size,
             matrix.m_profile1024->getNRows(),
             matrix.m_profile1024->kcol(),
             matrix.m_profile1024->cols(),
             matrix.m_matrix_dist_info,
             block_size);
  if(block_size==matrix.blockSize())
    m_matrix1024->setMatrixValues(matrix.m_matrix1024->m_values);
  else
  {
    assert(m_profile1024.get());
    assert(m_matrix1024.get());
    auto nb_blocks = m_profile1024->getBlockNnz();
    m_matrix1024->copy(nb_blocks, block_size, matrix.m_matrix1024->m_values, matrix.blockSize()) ;
  }
}
/*---------------------------------------------------------------------------*/

template class ALIEN_EXPORT SYCLBEllPackMatrix<double>;
template class ALIEN_EXPORT BEllPackStructInfo<1024, Integer>;
template class ALIEN_EXPORT SYCLInternal::MatrixInternal<double,1024>;


//template bool Alien::SYCLInternal::MatrixInternal<double,1014>::setMatrixValues(Alien::SYCLInternal::MatrixInternal<double,1014>::ValueBufferType& buffer) ;
// to force instanciation may be there is a better way to do it
void force_int_instance()
{
  SYCLBEllPackMatrix<double> matrix ;
  SYCLBEllPackMatrix<double>::MatrixInternal1024::ValueBufferType buffer (sycl::range<1>(10)) ;
  matrix.internal()->setMatrixValues(buffer) ;
}
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
