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

#include <alien/kernels/sycl/data/SYCLSendRecvOp.h>
#include <alien/kernels/sycl/data/SYCLLUSendRecvOp.h>

#include <alien/kernels/sycl/data/BEllPackStructInfo.h>

/*---------------------------------------------------------------------------*/

namespace Alien::SYCLInternal
{

#ifndef USE_SYCL2020
  using namespace cl ;
#endif

template <int EllPackSize, typename IndexT>
struct ALIEN_EXPORT StructInfoInternal
{
  // clang-format off
  static const int ellpack_size = EllPackSize ;
  using index_type              = IndexT;
  using IndexType               = IndexT;
  using index_buffer_type       = sycl::buffer<index_type, 1>;
  using IndexBufferType         = sycl::buffer<index_type, 1>;
  using MaskBufferType          = sycl::buffer<uint8_t, 1>;
  // clang-format on

  StructInfoInternal(std::size_t nrows,
                     std::size_t nnz,
                     std::size_t block_nrows,
                     std::size_t block_nnz,
                     int const* h_kcol,
                     int const* h_cols,
                     int const* h_block_row_offset,
                     int const* h_local_row_size);

  IndexBufferType& getBlockRowOffset() const { return m_block_row_offset; }

  IndexBufferType& getBlockCols() const { return m_block_cols; }

  IndexBufferType& getKCol() const { return m_kcol; }

  int const* kcol() const
  {
    return m_h_kcol.data();
  }

  int const* cols() const
  {
    return m_h_cols.data();
  }

  int const* dcol() const
  {
    getUpperDiagOffset();
    return m_h_dcol.data();
  }

  void getUpperDiagOffset() const;
  void computeLowerUpperMask() const;

  MaskBufferType& getLowerMask() const;
  MaskBufferType& getUpperMask() const;

  // clang-format off
  std::size_t m_nrows       = 0 ;
  std::size_t m_nnz         = 0 ;
  std::size_t m_block_nrows = 0 ;
  std::size_t m_block_nnz   = 0 ;

  std::vector<index_type> m_h_kcol ;
  std::vector<index_type> m_h_cols ;
  std::vector<index_type> m_h_block_cols ;

  mutable IndexBufferType                   m_block_row_offset ;
  mutable IndexBufferType                   m_block_cols ;
  mutable IndexBufferType                   m_kcol ;

  mutable bool                              m_lower_upper_mask_ready = false ;
  mutable std::vector<index_type>           m_h_dcol ;
  mutable std::unique_ptr<MaskBufferType>   m_lower_mask ;
  mutable std::unique_ptr<MaskBufferType>   m_upper_mask ;
  // clang-format on
};

/*---------------------------------------------------------------------------*/

template <typename ValueT, int EllPackSize>
class MatrixInternal
{
 public:
  // clang-format off
  using ThisType = MatrixInternal<ValueT,EllPackSize>;

  static const int ellpack_size = EllPackSize ;

  using ValueType           = ValueT;
  using value_type          = ValueT;

  using ProfileType         = BEllPackStructInfo<EllPackSize,int>;
  using InternalProfileType = typename ProfileType::InternalType;
  using IndexType           = typename InternalProfileType::IndexType;
  using IndexBufferType     = typename InternalProfileType::IndexBufferType;
  using IndexBufferPtrType  = std::unique_ptr<IndexBufferType>;

  using value_buffer_type   = sycl::buffer<value_type, 1>;
  using ValueBufferType     = sycl::buffer<value_type, 1>;
  using ValueBufferPtrType  = std::unique_ptr<ValueBufferType>;

  using QueueType           = sycl::queue;
  // clang-format on

  struct Tile
  {
    static const int ellpack_size = EllPackSize ;
    int m_N = 0 ;
    int m_NxN = 0 ;

    Tile(int N)
    : m_N(N)
    , m_NxN(N*N)
    {}

    inline std::size_t _ijk(std::size_t k, int i, int j) const
    {
      return (k*m_NxN + i*m_N + j)*ellpack_size;
    }

    inline std::size_t _ij(std::size_t local_id,int i, int j) const
    {
      return local_id*m_NxN+ i*m_N + j;
    }

    template<typename MatrixValueAccessorT,
             typename MatrixColAccessorT,
             typename VectorAccessorT>
    ValueType mult(int ieq,
                std::size_t local_id,
                std::size_t k,
                MatrixColAccessorT& cols,
                MatrixValueAccessorT& matrix,
                VectorAccessorT& x) const
    {
      ValueType value = 0. ;
      auto x_offset = cols[k*ellpack_size+local_id]*m_N ;
      if(x_offset>=0)
      {
        for(int j=0;j<m_N;++j)
        {
          auto mat_offset = _ijk(k,ieq,j)+local_id ;
          value += matrix[mat_offset]*x[x_offset+j] ;
          //printf("\n %d %d %d %d : %f += %f*%f ",ieq,j,int(k),int(mat_offset),value,matrix[mat_offset],x[x_offset+j]) ;
        }
      }
      return value ;
    }

    template<typename MatrixValueAccessorT,
             typename MatrixColAccessorT,
             typename MaskAccessorT,
             typename VectorAccessorT>
    ValueType mult(int ieq,
                   std::size_t local_id,
                   std::size_t k,
                   MatrixColAccessorT& cols,
                   MaskAccessorT& mask,
                   MatrixValueAccessorT& matrix,
                   VectorAccessorT& x) const
    {
      ValueType value = 0. ;
      auto x_offset = cols[k*ellpack_size+local_id]*m_N ;
      auto ma = mask[k*ellpack_size+local_id] ;
      if(x_offset>=0 && ma==1)
      {
        for(int j=0;j<m_N;++j)
        {
          auto mat_offset = _ijk(k,ieq,j)+local_id ;
          value += matrix[mat_offset]*x[x_offset+j] ;
          //printf("\n %d %d %d %d : %f += %f*%f ",ieq,j,int(k),int(mat_offset),value,matrix[mat_offset],x[x_offset+j]) ;
        }
      }
      return value ;
    }
  };

  template<typename MatrixAccT,
           typename VectorAccT,
           typename LUAccT>
  struct LU
  {
    static const int ellpack_size = EllPackSize ;
    int        m_N = 0 ;
    int        m_NxN = 0 ;
    MatrixAccT m_matrix;
    VectorAccT m_y;
    LUAccT     m_LU;

    LU(int N, MatrixAccT matrix, VectorAccT y, LUAccT lu)
    : m_N(N)
    , m_NxN(N*N)
    , m_matrix(matrix)
    , m_y(y)
    , m_LU(lu)
    {}

     inline std::size_t _ijk(std::size_t k, int i, int j) const
     {
       return (k*m_NxN + i*m_N + j)*ellpack_size;
     }

     inline std::size_t _ij(std::size_t local_id,int i, int j) const
     {
       return local_id*m_NxN+ i*m_N + j;
     }

     void factorize(std::size_t global_id,
                    std::size_t local_id,
                    std::size_t block_id,
                    std::size_t k) const
    {
      // Copy Diag Matrix in A
      for(int i=0;i<m_N;++i)
        for(int j=0;j<m_N;++j)
          m_LU[_ijk(block_id,i,j)+local_id] = m_matrix[_ijk(k,i,j)+local_id] ;

      //Factorize A = LU
      for (int k = 0; k < m_N; ++k)
      {
        //assert(m_LU[_ijk(block_id,k,k)+local_id] != 0);
        m_LU[_ijk(block_id,k,k)+local_id] = 1 / m_LU[_ijk(block_id,k,k)+local_id];
        for (int i = k + 1; i < m_N; ++i) {
          m_LU[_ijk(block_id,i,k)+local_id] *= m_LU[_ijk(block_id,k,k)+local_id];
        }
        for (int i = k + 1; i < m_N; ++i) {
          for (int j = k + 1; j < m_N; ++j) {
            m_LU[_ijk(block_id,i,j)+local_id] -= m_LU[_ijk(block_id,i,k)+local_id] * m_LU[_ijk(block_id,k,j)+local_id];
          }
        }
      }
    }

    void inverse(std::size_t global_id,
                 std::size_t local_id,
                 std::size_t block_id) const
    {
      // SET Y to Id
      for(int i=0;i<m_N;++i)
        for(int j=0;j<m_N;++j)
          m_y[_ij(global_id,i,j)] = 0. ;
      for(int i=0;i<m_N;++i)
        m_y[_ij(global_id,i,i)] = 1. ;

      // L solve
      for (int i = 1; i < m_N; ++i)
      {
        for (int j = 0; j < i; ++j)
        {
          for(int k=0;k<m_N;++k)
            m_y[_ij(global_id,i,k)] -= m_LU[_ijk(block_id,i,j)+local_id] * m_y[_ij(global_id,j,k)];
        }
      }

      // U solve
      for (int i = m_N - 1; i >= 0; --i)
      {
        for (int j = m_N - 1; j > i; --j)
        {
          for(int k=0;k<m_N;++k)
            m_y[_ij(global_id,i,k)] -= m_LU[_ijk(block_id,i,j)+local_id] * m_y[_ij(global_id,j,k)];
        }
        for(int k=0;k<m_N;++k)
          m_y[_ij(global_id,i,k)] *= m_LU[_ijk(block_id,i,i)+local_id];
      }
    }
  };

 public:
  MatrixInternal(ProfileType const* profile, int blk_size=1);

  ~MatrixInternal() {}

  bool setMatrixValues(ValueType const* values, bool only_host);
  bool setMatrixValuesFromHost();

  bool setMatrixValues(ValueBufferType& values);

  bool copy(std::size_t nb_blocks,
            Integer block_size,
            ValueBufferType& rhs_values,
            Integer rhs_block_size);

  bool needUpdate();
  void notifyChanges();
  void endUpdate();

  void mult(ValueBufferType& x, ValueBufferType& y) const;
  void mult(ValueBufferType& x, ValueBufferType& y, QueueType& queue) const;

  void addExtMult(ValueBufferType& x, ValueBufferType& y) const;
  void addExtMult(ValueBufferType& x, ValueBufferType& y, QueueType& queue) const;

  void addLMult(ValueType alpha, ValueBufferType& x, ValueBufferType& y) const;
  void addUMult(ValueType alpha, ValueBufferType& x, ValueBufferType& y) const;

  void addLMult(ValueType alpha, ValueBufferType& x, ValueBufferType& y, QueueType& queue) const;
  void addUMult(ValueType alpha, ValueBufferType& x, ValueBufferType& y, QueueType& queue) const;

  void multDiag(ValueBufferType& x, ValueBufferType& y) const;
  void multDiag(ValueBufferType& x, ValueBufferType& y, QueueType& queue) const;

  void multDiag(ValueBufferType& y) const;
  void multDiag(ValueBufferType& y, QueueType& queue) const;

  void computeDiag(ValueBufferType& y) const;
  void computeDiag(ValueBufferType& y, QueueType& queue) const;

  void computeBlockDiag(ValueBufferType& y) const;
  void computeBlockDiag(ValueBufferType& y, QueueType& queue) const;

  void multInvDiag(ValueBufferType& y) const;
  void multInvDiag(ValueBufferType& y, QueueType& queue) const;

  void computeInvDiag(ValueBufferType& y) const;
  void computeInvDiag(ValueBufferType& y, QueueType& queue) const;

  void computeInvBlockDiag(ValueBufferType& y) const;
  void computeInvBlockDiag(ValueBufferType& y, QueueType& queue) const;

  void scal(ValueBufferType& y);

  void scal(ValueBufferType& y, QueueType& queue);

  ValueBufferType& getValues() { return m_values; }

  ValueBufferType const getValues() const { return m_values; }

  //ProfileType* getProfile() { return m_profile; }

  ProfileType const* getProfile() const { return m_profile; }

  ValueType const* getHCsrData() const
  {
    return m_h_csr_values.data();
  }

  ValueType* getHCsrData()
  {
    return m_h_csr_values.data();
  }

  IndexBufferType& getSendIds() const
  {
    return *m_send_ids;
  }
  IndexBufferType& getRecvIds() const
  {
    return *m_recv_ids;
  }

  // clang-format off
  int                        m_N           = 1;
  int                        m_NxN         = 1;
  ProfileType const*         m_profile     = nullptr;
  ProfileType const*         m_ext_profile = nullptr;

  std::vector<ValueType>     m_h_csr_values ;
  std::vector<ValueType>     m_h_values ;
  mutable ValueBufferType    m_values ;

  std::vector<ValueType>     m_h_csr_ext_values ;
  std::vector<ValueType>     m_h_ext_values ;
  mutable ValueBufferPtrType m_ext_values ;
  bool                       m_values_is_update = false ;

  int const*                 m_h_interface_row_ids = nullptr;
  mutable IndexBufferPtrType m_interface_row_ids ;
  mutable IndexBufferPtrType m_send_ids ;
  mutable IndexBufferPtrType m_recv_ids ;
  // clang-format on
};

/*---------------------------------------------------------------------------*/

} // namespace Alien::SYCLInternal

/*---------------------------------------------------------------------------*/
