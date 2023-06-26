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

template <int BlockSize, typename IndexT>
void BEllPackStructInfo<BlockSize, IndexT>::computeBlockRowOffset(std::vector<int>& block_row_offset,
                                                                  std::size_t nrows,
                                                                  int const* kcol)
{
  std::size_t block_nrows = BEllPackStructInfo<BlockSize, IndexT>::nbBlocks(nrows);
  block_row_offset.resize(block_nrows + 1);
  int offset = 0;
  for (std::size_t ib = 0; ib < block_nrows; ++ib) {
    block_row_offset[ib] = offset;
    int max_row_size = 0;
    for (int i = 0; i < std::min(BlockSize, int(nrows - ib * BlockSize)); ++i) {
      auto irow = ib * BlockSize + i;
      int row_size = kcol[irow + 1] - kcol[irow];
      max_row_size = std::max(max_row_size, row_size);
    }
    offset += max_row_size;
  }
  block_row_offset[block_nrows] = offset;
}

template <int BlockSize, typename IndexT>
SYCLInternal::StructInfoInternal<BlockSize, IndexT>::
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
, m_h_block_cols(block_nnz * block_size)
, m_block_row_offset(h_block_row_offset, cl::sycl::range<1>(m_block_nrows + 1))
, m_block_cols(m_h_block_cols.data(), cl::sycl::range<1>(m_block_nnz * block_size))
, m_kcol(m_h_kcol.data(), cl::sycl::range<1>(m_nrows + 1))
{
  auto env = SYCLEnv::instance();

  auto& queue = env->internal()->queue();

  auto num_groups = env->internal()->maxNumGroups();

  auto max_work_group_size = env->internal()->maxWorkGroupSize();

  // building the best number of global thread
  auto total_threads = num_groups * block_size;

  // clang-format off
  if(h_local_row_size==nullptr)
  {
    IndexBufferType cols_buffer(m_h_cols.data(), cl::sycl::range<1>(m_nnz));

    queue.submit([&](cl::sycl::handler& cgh)
                 {
                   auto access_kcol_buffer      = m_kcol.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_block_row_offset = m_block_row_offset.template get_access<cl::sycl::access::mode::read>(cgh);

                   auto access_cols_buffer      = cols_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_block_cols       = m_block_cols.template get_access<cl::sycl::access::mode::read_write>(cgh);

                   cgh.parallel_for<class vector_rs>(cl::sycl::range<1>{total_threads},
                                                     [=] (cl::sycl::item<1> itemId)
                                                     {
                                                         auto id = itemId.get_id(0);

                                                         for (auto i = id; i < nrows; i += itemId.get_range()[0])
                                                         {
                                                           auto block_id = i/block_size ;
                                                           auto local_id = i%block_size ;

                                                           int begin              = access_kcol_buffer[i] ;
                                                           int end                = access_kcol_buffer[i+1] ;
                                                           int row_size           = end - begin ;

                                                           int block_row_offset   = access_block_row_offset[block_id]*block_size ;
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
                                                              access_block_cols[block_row_offset+(k-begin)*block_size+local_id] = access_cols_buffer[k] ;
                                                            }
                                                            for(int k=row_size;k<block_row_size;++k)
                                                            {
                                                              access_block_cols[block_row_offset+k*block_size+local_id] = 0 ;
                                                            }

                                                         }
                                                      });
                 });
  }
  else
  {
    IndexBufferType cols_buffer(m_h_cols.data(), cl::sycl::range<1>(m_nnz));
    IndexBufferType lrowsize_buffer(h_local_row_size, cl::sycl::range<1>(nrows));
    // clang-format off
    queue.submit([&](cl::sycl::handler& cgh)
                 {
                   auto access_kcol_buffer      = m_kcol.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_block_row_offset = m_block_row_offset.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_lrowsize_buffer  = lrowsize_buffer.template get_access<cl::sycl::access::mode::read>(cgh);

                   auto access_cols_buffer      = cols_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_block_cols       = m_block_cols.template get_access<cl::sycl::access::mode::read_write>(cgh);

                   cgh.parallel_for<class vector_rs>(cl::sycl::range<1>{total_threads},
                                                     [=] (cl::sycl::item<1> itemId)
                                                     {
                                                         auto id = itemId.get_id(0);

                                                         for (auto i = id; i < nrows; i += itemId.get_range()[0])
                                                         {
                                                           auto block_id = i/block_size ;
                                                           auto local_id = i%block_size ;

                                                           int begin              = access_kcol_buffer[i] ;
                                                           int lrow_size          = access_lrowsize_buffer[i] ;

                                                           int block_row_offset   = access_block_row_offset[block_id]*block_size ;
                                                           auto block_row_size = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                            for(int k=begin;k<begin+lrow_size;++k)
                                                            {
                                                              access_block_cols[block_row_offset+(k-begin)*block_size+local_id] = access_cols_buffer[k] ;
                                                            }
                                                            for(int k=lrow_size;k<block_row_size;++k)
                                                            {
                                                              access_block_cols[block_row_offset+k*block_size+local_id] = 0 ;
                                                            }
                                                         }
                                                      });
                 });
  }
  // clang-format on
}

template <int BlockSize, typename IndexT>
void SYCLInternal::StructInfoInternal<BlockSize, IndexT>::getUpperDiagOffset() const
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

template <int BlockSize, typename IndexT>
void SYCLInternal::StructInfoInternal<BlockSize, IndexT>::computeLowerUpperMask() const
{

  if (not m_lower_upper_mask_ready) {
    m_lower_mask.reset(new MaskBufferType(cl::sycl::range<1>(m_block_nnz * block_size)));
    m_upper_mask.reset(new MaskBufferType(cl::sycl::range<1>(m_block_nnz * block_size)));

    auto env = SYCLEnv::instance();

    auto& queue = env->internal()->queue();

    auto num_groups = env->internal()->maxNumGroups();

    auto max_work_group_size = env->internal()->maxWorkGroupSize();

    // building the best number of global thread
    auto total_threads = num_groups * block_size;

    {
      IndexBufferType dcol_buffer(m_h_dcol.data(), cl::sycl::range<1>(m_nrows));

      auto nrows = m_nrows;

      // clang-format off
      queue.submit([&](cl::sycl::handler& cgh)
                   {
                     auto access_dcol_buffer      = dcol_buffer.template get_access<cl::sycl::access::mode::read>(cgh);

                     auto access_kcol_buffer      = m_kcol.template get_access<cl::sycl::access::mode::read>(cgh);
                     auto access_block_row_offset = m_block_row_offset.template get_access<cl::sycl::access::mode::read>(cgh);

                     auto access_lower_mask       = cl::sycl::accessor { *m_lower_mask, cgh, cl::sycl::write_only, cl::sycl::property::no_init{}};
                     auto access_upper_mask       = cl::sycl::accessor { *m_upper_mask, cgh, cl::sycl::write_only, cl::sycl::property::no_init{}};


                     cgh.parallel_for<class vector_rs>(cl::sycl::range<1>{total_threads},
                                                       [=] (cl::sycl::item<1> itemId)
                                                       {
                                                           auto id = itemId.get_id(0);

                                                           for (auto i = id; i < nrows; i += itemId.get_range()[0])
                                                           {
                                                             auto block_id = i/block_size ;
                                                             auto local_id = i%block_size ;

                                                             int begin              = access_kcol_buffer[i] ;
                                                             int end                = access_kcol_buffer[i+1] ;
                                                             int diag               = access_dcol_buffer[i] ;
                                                             int row_size           = end - begin ;

                                                             int block_row_offset   = access_block_row_offset[block_id]*block_size ;
                                                             auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                             access_lower_mask[block_row_offset+(diag-begin)*block_size+local_id] = 0 ;
                                                             access_upper_mask[block_row_offset+(diag-begin)*block_size+local_id] = 0 ;

                                                             for(int k=begin;k<diag;++k)
                                                             {
                                                                access_lower_mask[block_row_offset+(k-begin)*block_size+local_id] = 1 ;
                                                                access_upper_mask[block_row_offset+(k-begin)*block_size+local_id] = 0 ;
                                                             }
                                                             for(int k=diag+1;k<end;++k)
                                                             {
                                                                access_lower_mask[block_row_offset+(k-begin)*block_size+local_id] = 0 ;
                                                                access_upper_mask[block_row_offset+(k-begin)*block_size+local_id] = 1 ;
                                                             }
                                                             for(int k=row_size;k<block_row_size;++k)
                                                             {
                                                                access_lower_mask[block_row_offset+k*block_size+local_id] = 0 ;
                                                                access_upper_mask[block_row_offset+k*block_size+local_id] = 0 ;
                                                             }

                                                           }
                                                        });
                   });
      // clang-format on
    }
    m_lower_upper_mask_ready = true;
  }
}

template <int BlockSize, typename IndexT>
typename SYCLInternal::StructInfoInternal<BlockSize, IndexT>::MaskBufferType&
SYCLInternal::StructInfoInternal<BlockSize, IndexT>::getLowerMask() const
{
  computeLowerUpperMask();
  return *m_lower_mask;
}

template <int BlockSize, typename IndexT>
typename SYCLInternal::StructInfoInternal<BlockSize, IndexT>::MaskBufferType&
SYCLInternal::StructInfoInternal<BlockSize, IndexT>::getUpperMask() const
{
  computeLowerUpperMask();
  return *m_upper_mask;
}

template <int BlockSize, typename IndexT>
BEllPackStructInfo<BlockSize, IndexT>::BEllPackStructInfo(std::size_t nrows,
                                                          int const* kcol,
                                                          int const* cols,
                                                          int const* h_block_row_offset,
                                                          int const* h_local_row_size)

: BaseBEllPackStructInfo(nrows, kcol[nrows])
, m_block_nrows(BEllPackStructInfo<BlockSize, IndexT>::nbBlocks(nrows))
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

  //m_h_block_cols.assign(m_block_nnz*block_size,-1) ;
  m_internal = new InternalType{ this->m_nrows,
                                 this->m_nnz,
                                 m_block_nrows,
                                 m_block_nnz,
                                 kcol,
                                 cols,
                                 h_block_row_offset,
                                 h_local_row_size };
}

template <int BlockSize, typename IndexT>
typename BEllPackStructInfo<BlockSize, IndexT>::IndexType const*
BEllPackStructInfo<BlockSize, IndexT>::kcol() const
{
  return m_internal->kcol();
}

template <int BlockSize, typename IndexT>
typename BEllPackStructInfo<BlockSize, IndexT>::IndexType const*
BEllPackStructInfo<BlockSize, IndexT>::cols() const
{
  return m_internal->cols();
}

template <int BlockSize, typename IndexT>
typename BEllPackStructInfo<BlockSize, IndexT>::IndexType const*
BEllPackStructInfo<BlockSize, IndexT>::dcol() const
{
  return m_internal->dcol();
}

namespace SYCLInternal
{
  template <typename ValueT, int BlockSize>
  MatrixInternal<ValueT, BlockSize>::MatrixInternal(ProfileType const* profile)
  : m_profile(profile)
  , m_h_values(profile->getBlockNnz() * block_size)
  , m_values(m_h_values.data(), cl::sycl::range<1>(profile->getBlockNnz() * block_size))
  {
    //m_values.set_final_data(nullptr);
    alien_debug([&] { cout() << "SYCL InternalMATRIX" << profile->getBlockNnz() * block_size; });
  }

  template <typename ValueT, int BlockSize>
  bool MatrixInternal<ValueT, BlockSize>::setMatrixValuesFromHost()
  {
    alien_debug([&] { cout() << "SYCLMatrix setMatrixValuesFromHost "; });
    auto env = SYCLEnv::instance();
    auto& queue = env->internal()->queue();
    auto num_groups = env->internal()->maxNumGroups();
    auto max_work_group_size = env->internal()->maxWorkGroupSize();
    auto total_threads = num_groups * block_size;

    auto nrows = m_profile->getNRows();
    auto nnz = m_profile->getNnz();
    auto block_nnz = m_profile->getBlockNnz();

    auto internal_profile = m_profile->internal();
    auto& kcol = internal_profile->getKCol();
    auto& block_row_offset = internal_profile->getBlockRowOffset();

    auto local_row_size = m_profile->localRowSize();
    if (local_row_size == nullptr) {
      ValueBufferType values_buffer(m_h_csr_values.data(), cl::sycl::range<1>(nnz));
      // COMPUTE COLS
      // clang-format off
        queue.submit([&](cl::sycl::handler& cgh)
                     {
                       auto access_kcol_buffer      = internal_profile->getKCol().template get_access<cl::sycl::access::mode::read>(cgh);
                       auto access_block_row_offset = internal_profile->getBlockRowOffset().template get_access<cl::sycl::access::mode::read>(cgh);
                       auto access_values_buffer    = values_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
                       auto access_block_values     = m_values.template get_access<cl::sycl::access::mode::read_write>(cgh);

                       //cgh.parallel_for<class vector_axpy>(cl::sycl::nd_range<1>{cl::sycl::range<1>{total_threads},cl::sycl::range<1>{block_size}},[=](cl::sycl::nd_item<1> item_id)
                       cgh.parallel_for<class vector_values>(cl::sycl::range<1>{total_threads},
                                                             [=] (cl::sycl::item<1> item_id)
                                                             {
                                                               auto id = item_id.get_id(0);
                                                               //auto local_id  = item_id.get_local_id(0);
                                                               //auto block_id  = item_id.get_group(0) ;
                                                               //auto global_id = item_id.get_global_id(0);

                                                               for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                               {
                                                                  auto block_id = i/block_size ;
                                                                  auto local_id = i%block_size ;

                                                                  auto begin              = access_kcol_buffer[i] ;
                                                                  auto end                = access_kcol_buffer[i+1] ;
                                                                  auto row_size           = end - begin ;

                                                                  int block_row_offset   = access_block_row_offset[block_id]*block_size ;
                                                                  auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                  for(int k=begin;k<end;++k)
                                                                  {
                                                                    access_block_values[block_row_offset+(k-begin)*block_size+local_id] = access_values_buffer[k] ;
                                                                  }
                                                                  for(int k=row_size;k<block_row_size;++k)
                                                                  {
                                                                    access_block_values[block_row_offset+k*block_size+local_id] = 0 ;
                                                                  }
                                                               }
                                                            });
                     }) ;
      // clang-format on
      m_values_is_update = true;
    }
    else {
      ValueBufferType values_buffer(m_h_csr_values.data(), cl::sycl::range<1>(nnz));
      IndexBufferType lrowsize_buffer(local_row_size, cl::sycl::range<1>(nrows));
      // COMPUTE COLS
      // clang-format off
      queue.submit([&](cl::sycl::handler& cgh)
                   {
                     auto access_kcol_buffer      = internal_profile->getKCol().template get_access<cl::sycl::access::mode::read>(cgh);
                     auto access_lrowsize_buffer  = lrowsize_buffer.template get_access<cl::sycl::access::mode::read>(cgh);

                     auto access_block_row_offset = internal_profile->getBlockRowOffset().template get_access<cl::sycl::access::mode::read>(cgh);
                     auto access_values_buffer    = values_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
                     auto access_block_values     = m_values.template get_access<cl::sycl::access::mode::read_write>(cgh);

                     //cgh.parallel_for<class vector_axpy>(cl::sycl::nd_range<1>{cl::sycl::range<1>{total_threads},cl::sycl::range<1>{block_size}},[=](cl::sycl::nd_item<1> item_id)
                     cgh.parallel_for<class vector_values>(cl::sycl::range<1>{total_threads},
                                                           [=] (cl::sycl::item<1> item_id)
                                                           {
                                                             auto id = item_id.get_id(0);
                                                             //auto local_id  = item_id.get_local_id(0);
                                                             //auto block_id  = item_id.get_group(0) ;
                                                             //auto global_id = item_id.get_global_id(0);

                                                             for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                             {
                                                                auto block_id = i/block_size ;
                                                                auto local_id = i%block_size ;

                                                                auto begin              = access_kcol_buffer[i] ;
                                                                auto lrow_size          = access_lrowsize_buffer[i] ;
                                                                auto end                = begin + lrow_size ;

                                                                int block_row_offset   = access_block_row_offset[block_id]*block_size ;
                                                                auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                for(int k=begin;k<end;++k)
                                                                {
                                                                  access_block_values[block_row_offset+(k-begin)*block_size+local_id] = access_values_buffer[k] ;
                                                                }
                                                                for(int k=lrow_size;k<block_row_size;++k)
                                                                {
                                                                  access_block_values[block_row_offset+k*block_size+local_id] = 0 ;
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
      m_ext_values.reset(new ValueBufferType(ext_block_nnz * block_size)) ;
      {
        ValueBufferType ext_csr_values_buffer(m_h_csr_ext_values.data(), cl::sycl::range<1>(ext_nnz));
        queue.submit([&](cl::sycl::handler& cgh)
                     {
                       auto access_kcol_buffer      = ext_kcol.template get_access<cl::sycl::access::mode::read>(cgh);
                       auto access_block_row_offset = ext_block_row_offset.template get_access<cl::sycl::access::mode::read>(cgh);
                       auto access_values_buffer    = ext_csr_values_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
                       //auto access_block_values     = m_ext_values->template get_access<cl::sycl::access::mode::read_write>(cgh);
                       auto access_block_values     = cl::sycl::accessor { *m_ext_values, cgh, cl::sycl::write_only, cl::sycl::property::no_init{}};

                       cgh.parallel_for<class vector_ghost_values>(cl::sycl::range<1>{total_threads},
                                                                   [=] (cl::sycl::item<1> item_id)
                                                                   {
                                                                     auto id = item_id.get_id(0);

                                                                     for (auto i = id; i < interface_nrows; i += item_id.get_range()[0])
                                                                     {
                                                                        auto block_id = i/block_size ;
                                                                        auto local_id = i%block_size ;
                                                                        auto begin              = access_kcol_buffer[i] ;
                                                                        auto end                = access_kcol_buffer[i+1] ;
                                                                        auto row_size           = end - begin ;

                                                                        int block_row_offset   = access_block_row_offset[block_id]*block_size ;
                                                                        auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                        for(int k=begin;k<end;++k)
                                                                        {
                                                                          access_block_values[block_row_offset+(k-begin)*block_size+local_id] = access_values_buffer[k] ;
                                                                        }
                                                                        for(int k=row_size;k<block_row_size;++k)
                                                                        {
                                                                          access_block_values[block_row_offset+k*block_size+local_id] = 0 ;
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

  template <typename ValueT, int BlockSize>
  bool MatrixInternal<ValueT, BlockSize>::setMatrixValues(ValueT const* values, bool only_host)
  {
    alien_debug([&] { cout() << "SYCLMatrix setMatrixValues " << only_host; });
    auto nnz = m_profile->getNnz();
    m_h_csr_values.resize(nnz);
    std::copy(values, values + nnz, m_h_csr_values.begin());
    if (m_ext_profile) {
      // clang-format off
      auto kcol            = m_profile->kcol() ;
      auto local_row_size  = m_profile->localRowSize() ;

      auto interface_nrows = m_ext_profile->getNRows() ;
      auto ext_nnz         = m_ext_profile->getNnz();
      // clang-format on

      m_h_csr_ext_values.resize(ext_nnz);

      // EXTRACT EXTERNAL PROFILE
      {
        int jcol = 0;
        for (std::size_t i = 0; i < interface_nrows; ++i) {
          auto id = m_h_interface_row_ids[i];
          for (int k = kcol[id] + local_row_size[id]; k < kcol[id + 1]; ++k) {
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

  template <typename ValueT, int BlockSize>
  bool MatrixInternal<ValueT, BlockSize>::needUpdate()
  {
    return m_values_is_update != true;
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::notifyChanges()
  {
    m_values_is_update = false;
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::endUpdate()
  {
    if (not m_values_is_update) {
      setMatrixValuesFromHost();
    }
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::mult(ValueBufferType& x, ValueBufferType& y, cl::sycl::queue& queue) const
  {

    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * block_size;

    // clang-format off
    auto nrows             = m_profile->getNRows();
    auto nnz               = m_profile->getNnz();

    auto internal_profile  = m_profile->internal();
    auto& kcol             = internal_profile->getKCol();
    auto& block_row_offset = internal_profile->getBlockRowOffset();
    auto& block_cols       = internal_profile->getBlockCols();
    // clang-format on
    {
      // COMPUTE VALUES
      // clang-format off
        queue.submit([&](cl::sycl::handler& cgh)
                 {
                   auto access_block_row_offset = block_row_offset.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_cols             = block_cols.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_values           = m_values.template get_access<cl::sycl::access::mode::read>(cgh);


                   auto access_x                = x.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_y                = y.template get_access<cl::sycl::access::mode::read_write>(cgh);


                   //cl::sycl::nd_range<1> r{cl::sycl::range<1>{total_threads},cl::sycl::range<1>{block_size}};
                   //cgh.parallel_for<class compute_mult>(r, [&](cl::sycl::nd_item<1> item_id)
                   cgh.parallel_for<class compute_mult>(cl::sycl::range<1>{total_threads},
                                                        [=] (cl::sycl::item<1> item_id)
                                                        {
                                                          auto id = item_id.get_id(0);
                                                          //auto local_id  = item_id.get_local_id(0);
                                                          //auto block_id  = item_id.get_group(0) ;
                                                          //auto global_id = item_id.get_global_id(0);

                                                          for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                          {
                                                             auto block_id = i/block_size ;
                                                             auto local_id = i%block_size ;

                                                             int block_row_offset   = access_block_row_offset[block_id]*block_size ;
                                                             auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                             ValueType value = 0. ;
                                                             for(int j=0;j<block_row_size;++j)
                                                             {
                                                               auto k = block_row_offset+j*block_size+local_id ;
                                                                value += access_values[k]* access_x[access_cols[k]] ;
                                                             }
                                                             access_y[i] = value ;
                                                          }
                                                      });
                 });
      // clang-format on
    }
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::mult(ValueBufferType& x, ValueBufferType& y) const
  {
    this->mult(x, y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::addExtMult(ValueBufferType& x,
                                                     ValueBufferType& y,
                                                     cl::sycl::queue& queue) const
  {
    //alien_debug([&] {cout() << "SYCL MatrixInternal::addExMult: ";});
    //Universe().traceMng()->flush() ;

    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * block_size;

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
      queue.submit([&](cl::sycl::handler& cgh)
                   {
                     auto access_block_row_offset = block_row_offset.template get_access<cl::sycl::access::mode::read>(cgh);
                     auto access_cols             = block_cols.template get_access<cl::sycl::access::mode::read>(cgh);
                     auto access_values           = m_ext_values->template get_access<cl::sycl::access::mode::read>(cgh);

                     auto access_row_ids          = m_interface_row_ids->template get_access<cl::sycl::access::mode::read>(cgh);

                     auto access_x                = x.template get_access<cl::sycl::access::mode::read>(cgh);
                     auto access_y                = y.template get_access<cl::sycl::access::mode::read_write>(cgh);

                     cgh.parallel_for<class compute_ext_mult>(cl::sycl::range<1>{total_threads},
                                                              [=] (cl::sycl::item<1> item_id)
                                                              {
                                                                auto id = item_id.get_id(0);

                                                                for (auto i = id; i < interface_nrow; i += item_id.get_range()[0])
                                                                {
                                                                   auto block_id = i/block_size ;
                                                                   auto local_id = i%block_size ;

                                                                   int block_row_offset   = access_block_row_offset[block_id]*block_size ;
                                                                   auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                                   ValueType value = 0. ;
                                                                   for(int j=0;j<block_row_size;++j)
                                                                   {
                                                                     auto k = block_row_offset+j*block_size+local_id ;
                                                                     value += access_values[k]* access_x[access_cols[k]] ;
                                                                   }
                                                                   access_y[access_row_ids[i]] += value ;
                                                                }
                                                            });
                   });
      // clang-format on
    }
  }
  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::addExtMult(ValueBufferType& x,
                                                     ValueBufferType& y) const
  {
    this->addExtMult(x, y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::addLMult(ValueType alpha,
                                                   ValueBufferType& x,
                                                   ValueBufferType& y,
                                                   cl::sycl::queue& queue) const
  {
    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * block_size;

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
        queue.submit([&](cl::sycl::handler& cgh)
                 {
                   auto access_block_row_offset = block_row_offset.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_cols             = block_cols.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_mask             = mask.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_values           = m_values.template get_access<cl::sycl::access::mode::read>(cgh);


                   auto access_x                = x.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_y                = y.template get_access<cl::sycl::access::mode::read_write>(cgh);


                   //cl::sycl::nd_range<1> r{cl::sycl::range<1>{total_threads},cl::sycl::range<1>{block_size}};
                   //cgh.parallel_for<class compute_mult>(r, [&](cl::sycl::nd_item<1> item_id)
                   cgh.parallel_for<class compute_mult>(cl::sycl::range<1>{total_threads},
                                                        [=] (cl::sycl::item<1> item_id)
                                                        {
                                                          auto id = item_id.get_id(0);
                                                          //auto local_id  = item_id.get_local_id(0);
                                                          //auto block_id  = item_id.get_group(0) ;
                                                          //auto global_id = item_id.get_global_id(0);

                                                          for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                          {
                                                             auto block_id = i/block_size ;
                                                             auto local_id = i%block_size ;

                                                             int block_row_offset   = access_block_row_offset[block_id]*block_size ;
                                                             auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                             ValueType value = access_y[i] ;
                                                             for(int j=0;j<block_row_size;++j)
                                                             {
                                                               auto k = block_row_offset+j*block_size+local_id ;
                                                                value += alpha * access_mask[k] * access_values[k]* access_x[access_cols[k]] ;
                                                             }
                                                             access_y[i] = value ;
                                                          }
                                                      });
                 });
      // clang-format on
    }
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::addLMult(ValueType alpha, ValueBufferType& x, ValueBufferType& y) const
  {
    this->addLMult(alpha, x, y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::addUMult(ValueType alpha,
                                                   ValueBufferType& x,
                                                   ValueBufferType& y,
                                                   cl::sycl::queue& queue) const
  {

    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * block_size;

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
        queue.submit([&](cl::sycl::handler& cgh)
                 {
                   auto access_block_row_offset = block_row_offset.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_cols             = block_cols.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_mask             = mask.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_values           = m_values.template get_access<cl::sycl::access::mode::read>(cgh);


                   auto access_x                = x.template get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_y                = y.template get_access<cl::sycl::access::mode::read_write>(cgh);

                   cgh.parallel_for<class compute_mult>(cl::sycl::range<1>{total_threads},
                                                        [=] (cl::sycl::item<1> item_id)
                                                        {
                                                          auto id = item_id.get_id(0);
                                                          //auto local_id  = item_id.get_local_id(0);
                                                          //auto block_id  = item_id.get_group(0) ;
                                                          //auto global_id = item_id.get_global_id(0);

                                                          for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                          {
                                                             auto block_id = i/block_size ;
                                                             auto local_id = i%block_size ;

                                                             int block_row_offset   = access_block_row_offset[block_id]*block_size ;
                                                             auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;

                                                             ValueType value = access_y[i] ;
                                                             for(int j=0;j<block_row_size;++j)
                                                             {
                                                               auto k = block_row_offset+j*block_size+local_id ;
                                                                value += alpha * access_mask[k] * access_values[k]* access_x[access_cols[k]] ;
                                                             }
                                                             access_y[i] = value ;
                                                          }
                                                      });
                 });
      }
    // clang-format on
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::addUMult(ValueType alpha, ValueBufferType& x, ValueBufferType& y) const
  {
    this->addUMult(alpha, x, y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::multInvDiag(ValueBufferType& y, cl::sycl::queue& queue) const
  {
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::multInvDiag(ValueBufferType& y) const
  {
    this->multInvDiag(y, SYCLEnv::instance()->internal()->queue());
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::computeInvDiag(ValueBufferType& y, cl::sycl::queue& queue) const
  {

    auto device = queue.get_device();

    auto num_groups = queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto max_work_group_size = queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * block_size;

    // clang-format off
    auto nrows = m_profile->getNRows() ;
    auto nnz   = m_profile->getNnz() ;

    auto internal_profile  = m_profile->internal() ;
    auto& kcol             = internal_profile->getKCol() ;
    auto& block_row_offset = internal_profile->getBlockRowOffset() ;
    auto& block_cols       = internal_profile->getBlockCols() ;
    {
      // COMPUTE VALUES
      queue.submit([&](cl::sycl::handler& cgh)
                   {
                     auto access_block_row_offset = block_row_offset.template get_access<cl::sycl::access::mode::read>(cgh);
                     auto access_cols             = block_cols.template get_access<cl::sycl::access::mode::read>(cgh);
                     auto access_values           = m_values.template get_access<cl::sycl::access::mode::read>(cgh);
                     auto access_y                = y.template get_access<cl::sycl::access::mode::read_write>(cgh);


                     cgh.parallel_for<class compute_mult>(cl::sycl::range<1>{total_threads},
                                                          [=] (cl::sycl::item<1> item_id)
                                                          {
                                                            auto id = item_id.get_id(0);
                                                            //auto local_id  = item_id.get_local_id(0);
                                                            //auto block_id  = item_id.get_group(0) ;
                                                            //auto global_id = item_id.get_global_id(0);

                                                            for (auto i = id; i < nrows; i += item_id.get_range()[0])
                                                            {
                                                               auto block_id = i/block_size ;
                                                               auto local_id = i%block_size ;

                                                               int block_row_offset   = access_block_row_offset[block_id]*block_size ;
                                                               auto block_row_size    = access_block_row_offset[block_id+1]-access_block_row_offset[block_id] ;
                                                               for(int j=0;j<block_row_size;++j)
                                                               {
                                                                 auto k = block_row_offset+j*block_size+local_id ;
                                                                 if((access_cols[k])==int(i) && (access_values[k]!=0) )
                                                                   access_y[i] = 1./access_values[k] ;
                                                               }
                                                            }
                                                          });
                   });
    }
    // clang-format on
  }

  template <typename ValueT, int BlockSize>
  void MatrixInternal<ValueT, BlockSize>::computeInvDiag(ValueBufferType& y) const
  {
    this->computeInvDiag(y, SYCLEnv::instance()->internal()->queue());
  }
} // namespace SYCLInternal

template <typename ValueT>
SYCLBEllPackMatrix<ValueT>::~SYCLBEllPackMatrix()
{
  delete m_profile1024;
  delete m_matrix1024;

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
           SimpleCSRInternal::DistStructInfo const& matrix_dist_info)
{
  alien_debug([&] { cout() << "INIT SYCL MATRIX " << parallel_mng; });
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

  m_block_size = 1024;
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
    delete m_profile1024;
    m_profile1024 = new ProfileInternal1024{ nrows,
                                             kcol,
                                             sorted_cols,
                                             m_block_row_offset.data(),
                                             local_row_size.data() };

    delete m_matrix1024;
    m_matrix1024 = new MatrixInternal1024{ m_profile1024 };

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

    delete m_ext_profile1024;
    m_ext_profile1024 = new ProfileInternal1024{ interface_nrows,
                                                 ext_kcol.data(),
                                                 ext_cols.data(),
                                                 m_ext_block_row_offset.data(),
                                                 nullptr };

    m_matrix1024->m_ext_profile = m_ext_profile1024;
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

    delete m_profile1024;
    m_profile1024 = new ProfileInternal1024{ nrows,
                                             kcol,
                                             cols,
                                             m_block_row_offset.data(),
                                             nullptr };

    delete m_matrix1024;
    m_matrix1024 = new MatrixInternal1024{ m_profile1024 };
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
  if (m_matrix1024)
    m_matrix1024->notifyChanges();
}

template <typename ValueT>
void SYCLBEllPackMatrix<ValueT>::endUpdate()
{
  if (m_matrix1024 && m_matrix1024->needUpdate()) {
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

/*---------------------------------------------------------------------------*/

template class ALIEN_EXPORT SYCLBEllPackMatrix<double>;
template class ALIEN_EXPORT BEllPackStructInfo<1024, Integer>;
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
