// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <numeric>

#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>

/*---------------------------------------------------------------------------*/

namespace Alien::SimpleCSRInternal
{

/*---------------------------------------------------------------------------*/

template <typename ValueT = Real>
class MatrixInternal
{
 public:
  typedef ValueT ValueType;
  typedef MatrixInternal<ValueType> ThisType;
  typedef CSRStructInfo ProfileType;

 public:
  MatrixInternal(bool is_variable_block = false)
  : m_profile(new ProfileType(is_variable_block))
  {}

  ~MatrixInternal() {}

  void setValues(ValueT value)
  {
    m_values.fill(value) ;
  }

  ConstArrayView<ValueType> getValues() const { return m_values; }

  UniqueArray<ValueType>& getValues() { return m_values; }

  ValueType* getDataPtr() { return m_values.data(); }

  ValueType const* getDataPtr() const { return m_values.data(); }

  // Remark: once a profile is associated to a matrix he should not allow profile change
  CSRStructInfo& getCSRProfile() { return *m_profile; }

  const CSRStructInfo& getCSRProfile() const { return *m_profile; }

  Integer getRowSize(Integer row) const { return m_profile->getRowSize(row); }

  void scal(ValueType const* diag)
  {
    auto nrows = m_profile->getNRows() ;
    auto kcol = m_profile->kcol() ;
    for(int irow=0;irow<nrows;++irow)
    {
      ValueType scal = diag[irow] ;
      for(int k=kcol[irow];k<kcol[irow+1];++k)
        m_values[k] *= scal ;
    }
  }

  void clear() { m_values.resize(0); }

  MatrixInternal<ValueT>* clone() const { return new MatrixInternal<ValueT>(*this); }

  template <typename T>
  void copy(const MatrixInternal<T>& internal)
  {
    m_values.copy(internal.getValues());
    m_profile->copy(internal.getCSRProfile());
  }

  template <typename T>
  void copy(const MatrixInternal<T>& internal, Integer block_size1, Integer block_size2, Integer nb_blocks)
  {
    auto const& values2 = internal.getValues() ;
    if(block_size1==block_size2)
      m_values.copy(values2);
    else if(block_size1==1)
    {
      Integer stride2 = block_size2*block_size2 ;
      Integer offset2 = 0 ;
      m_values.resize(nb_blocks) ;
      for(Integer ib=0;ib<nb_blocks;++ib)
      {
        m_values[ib] = values2[offset2];
        offset2 += stride2 ;
      }
    }
    else
    {
      Integer stride1 = block_size1*block_size1 ;
      Integer stride2 = block_size2*block_size2 ;
      Integer offset1 = 0 ;
      Integer offset2 = 0 ;
      m_values.resize(nb_blocks*stride1) ;
      for(Integer ib=0;ib<nb_blocks;++ib)
      {
        for(Integer i=0;i<block_size1;++i)
          for(Integer j=0;j<block_size1;++j)
            m_values[offset1 + i*block_size1+j] = values2[offset2+ i*block_size2+j];
        offset1 += stride1 ;
        offset2 += stride2 ;
      }
    }
    m_profile->copy(internal.getCSRProfile());
  }


  void computeSubMatrix(ConstArrayView<Integer> parent_lids,
                        ConstArrayView<Integer> old2new_uids,
                        Integer parent_nrow,
                        ConstArrayView<ValueType> parent_values,
                        ConstArrayView<Integer> parent_kcol,
                        ConstArrayView<Integer> parent_cols,
                        Integer parent_local_offset,
                        ConstArrayView<Integer> filter,
                        ConstArrayView<Integer> new_offsets,
                        Integer my_rank)
  {
    m_profile->computeSubProfile(parent_lids,
                                 old2new_uids,
                                 parent_nrow,
                                 parent_kcol,
                                 parent_cols,
                                 filter,
                                 new_offsets,
                                 my_rank);

    m_values.resize(m_profile->getNnz()) ;
    auto kcol = m_profile->getRowOffset() ;
    for (Integer irow = 0; irow < m_profile->getNRows(); ++irow)
    {
      Integer parent_row    = parent_lids[irow];
      Integer kstart        = kcol[irow];
      Integer kend          = kcol[irow + 1];
      Integer parent_kstart = parent_kcol[parent_row];
      Integer parent_kend   = parent_kcol[parent_row + 1];
      Integer k = kstart;
      for (Integer pk = parent_kstart; pk < parent_kend; ++pk)
      {
        Integer col = parent_cols[pk] - parent_local_offset;
        if (filter[col] != 1)
          continue;
        m_values[k] = parent_values[pk];
        ++k;
      }
    }
  }

  void computeSubMatrix(ConstArrayView<Integer> parent_lids,
                        ConstArrayView<Integer> old2new_uids,
                        Integer parent_local_offset,
                        Integer parent_nrow,
                        ConstArrayView<ValueType> parent_values,
                        ConstArrayView<Integer> parent_kcol,
                        ConstArrayView<Integer> parent_cols,
                        ConstArrayView<Integer> parent_local_row_sizes,
                        ConstArrayView<Integer> all_filter,
                        ConstArrayView<Integer> new_offsets,
                        Integer my_rank)
  {
    Integer local_offset = new_offsets[my_rank] ;
    m_profile->computeSubProfile(parent_lids,
                                 old2new_uids,
                                 parent_local_offset,
                                 parent_nrow,
                                 parent_kcol,
                                 parent_cols,
                                 parent_local_row_sizes,
                                 all_filter,
                                 new_offsets,
                                 my_rank);

    m_values.resize(m_profile->getNnz()) ;
    auto kcol = m_profile->getRowOffset() ;
    for (Integer irow = 0; irow < m_profile->getNRows(); ++irow)
    {
      Integer parent_row     = parent_lids[irow];
      Integer kstart         = kcol[irow];
      Integer kend           = kcol[irow + 1];
      Integer parent_kstart  = parent_kcol[parent_row];
      Integer parent_kend    = parent_kcol[parent_row + 1];
      Integer local_row_size = parent_local_row_sizes[parent_row];
      Integer k = kstart;
      for (Integer pk = parent_kstart; pk < parent_kstart + local_row_size; ++pk)
      {
        Integer col = parent_cols[pk];
        if (all_filter[col] != 1)
          continue;
        m_values[k] = parent_values[pk];
        ++k;
      }
      for (Integer pk = parent_kstart + local_row_size; pk < parent_kend; ++pk)
      {
        Integer col = parent_cols[pk];
        if (all_filter[col] != 1)
          continue;
        m_values[k] = parent_values[pk];
        ++k;
      }
    }
  }

  bool needUpdate()
  {
    return m_is_update != true;
  }

  void notifyChanges()
  {
    m_is_update = false;
  }

  void endUpdate()
  {
    m_is_update = true;
  }

  bool m_is_update = false;
  UniqueArray<ValueType> m_values;
  std::shared_ptr<CSRStructInfo> m_profile;
};

/*---------------------------------------------------------------------------*/

} // namespace Alien::SimpleCSRInternal

/*---------------------------------------------------------------------------*/
