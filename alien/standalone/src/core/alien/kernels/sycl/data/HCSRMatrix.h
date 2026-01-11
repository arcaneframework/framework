// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#pragma once

#include <alien/core/impl/IMatrixImpl.h>
#include <alien/core/impl/MultiMatrixImpl.h>

#include <alien/data/ISpace.h>

#include <alien/kernels/sycl/SYCLPrecomp.h>

#include <alien/kernels/sycl/data/BEllPackStructInfo.h>
#include <alien/kernels/sycl/data/SYCLDistStructInfo.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>

#include <alien/utils/StdTimer.h>



/*---------------------------------------------------------------------------*/

namespace Alien
{

  namespace HCSRInternal
  {
    template <typename ValueT>
    class MatrixInternal;
  }
/*---------------------------------------------------------------------------*/

template <typename ValueT>
class ALIEN_EXPORT HCSRMatrix : public IMatrixImpl
{
 public:
  // clang-format off
  static const bool                                         on_host_only = false ;
  typedef BackEnd::tag::hcsr                                TagType ;
  typedef ValueT                                            ValueType;
  typedef ValueT                                            value_type ;

  typedef SimpleCSRInternal::CSRStructInfo                  CSRStructInfo;
  typedef SimpleCSRInternal::CSRStructInfo                  ProfileType;
  typedef SYCLInternal::SYCLDistStructInfo                  DistStructInfo;
  typedef HCSRInternal::MatrixInternal<ValueType>           MatrixInternal;
  typedef HCSRInternal::MatrixInternal<ValueType>           InternalType;

  typedef typename ProfileType::IndexType                   IndexType ;
  // clang-format on

  class CSRView
  {
  public:
    CSRView(HCSRMatrix const* parent,
            BackEnd::Memory::eType,
            int nrows,
            int nnz) ;

    virtual ~CSRView() ;

    HCSRMatrix const* m_parent = nullptr ;
    BackEnd::Memory::eType m_memory = BackEnd::Memory::Host ;
    int m_nrows         = 0 ;
    int m_nnz           = 0 ;
    IndexType* m_rows   = nullptr ;
    IndexType* m_ncols  = nullptr ;
    IndexType* m_cols   = nullptr ;
    ValueType* m_values = nullptr ;
  };



 public:
  /** Constructeur de la classe */
  HCSRMatrix() ;

  /** Constructeur de la classe */
  HCSRMatrix(const MultiMatrixImpl* multi_impl);

  /** Destructeur de la classe */
  virtual ~HCSRMatrix();

  void setTraceMng(ITraceMng* trace_mng) { m_trace = trace_mng; }

  void allocate() ;

  CSRStructInfo& getCSRProfile() { return *m_profile; }

  const CSRStructInfo& getCSRProfile() const { return *m_profile; }

  const CSRStructInfo& getProfile() const { return *m_profile; }

  const DistStructInfo& getDistStructInfo() const { return m_matrix_dist_info; }

  IMessagePassingMng* getParallelMng()
  {
    return m_parallel_mng;
  }


  void sequentialStart()
  {
    m_local_offset = 0;
    m_local_size = getCSRProfile().getNRows();
    m_global_size = m_local_size;
    m_myrank = 0;
    m_nproc = 1;
    m_is_parallel = false;
    m_matrix_dist_info.m_local_row_size.resize(m_local_size);
    auto& profile = internal()->getCSRProfile();
    ConstArrayView<Integer> offset = profile.getRowOffset();
    for (Integer i = 0; i < m_local_size; ++i)
      m_matrix_dist_info.m_local_row_size[i] = offset[i + 1] - offset[i];
  }

  void parallelStart(ConstArrayView<Integer> offset, IMessagePassingMng* parallel_mng,
                     bool need_sort_ghost_col = false)
  {
    m_local_size = getCSRProfile().getNRows();
    m_parallel_mng = parallel_mng;
    // m_trace = parallel_mng->traceMng();
    if (m_parallel_mng == NULL) {
      m_local_offset = 0;
      m_global_size = m_local_size;
      m_myrank = 0;
      m_nproc = 1;
      m_is_parallel = false;
    }
    else {
      m_myrank = m_parallel_mng->commRank();
      m_nproc = m_parallel_mng->commSize();
      m_local_offset = offset[m_myrank];
      m_global_size = offset[m_nproc];
      m_is_parallel = (m_nproc > 1);
    }
    if (m_is_parallel) {
      if (need_sort_ghost_col)
        sortGhostCols(offset);
      m_matrix_dist_info.compute(
      m_nproc, offset, m_myrank, m_parallel_mng, getCSRProfile(), m_trace);

      m_ghost_size = m_matrix_dist_info.m_ghost_nrow;
    }
  }

 public:
  bool initMatrix(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
                  Integer local_offset,
                  Integer global_size,
                  std::size_t nrows,
                  int const* kcol,
                  int const* cols,
                  SimpleCSRInternal::DistStructInfo const& matrix_dist_info);

  HCSRMatrix* cloneTo(const MultiMatrixImpl* multi) const;

  bool isParallel() const { return m_is_parallel; }

  Integer getLocalSize() const { return m_local_size; }

  Integer getLocalOffset() const { return m_local_offset; }

  Integer getGlobalSize() const { return m_global_size; }

  Integer getGhostSize() const { return m_ghost_size; }

  Integer getAllocSize() const { return m_local_size + m_ghost_size; }

  bool setMatrixValues(Arccore::Real const* values, bool only_host);

  void notifyChanges();
  void endUpdate();

  MatrixInternal* internal() { return m_internal.get(); }

  MatrixInternal const* internal() const { return m_internal.get(); }

  void allocateDevicePointers(int** ncols, int** rows, int** cols, ValueType** values) const ;
  void initDevicePointers(int** ncols, int** rows, int** cols, ValueType** values) const ;
  void freeDevicePointers(int* ncols, int* rows, int* cols, ValueType* values) const ;
  void copyDevicePointers(int* rows, int* ncols, int* cols, ValueType* values) const ;

  CSRView csrView(BackEnd::Memory::eType memory, int nrows, int nnz) const;

  void initCOODevicePointers(int** dof_uids, int** rows, int** cols, ValueType** values) const ;
  void freeCOODevicePointers(int* dof_uids, int* rows, int* cols, ValueType* values) const ;

 private:
  class IsLocal
  {
   public:
    IsLocal(const ConstArrayView<Integer> offset, const Integer myrank)
    : m_offset(offset)
    , m_myrank(myrank)
    {}
    bool operator()(Arccore::Integer col) const
    {
      return (col >= m_offset[m_myrank]) && (col < m_offset[m_myrank + 1]);
    }

   private:
    const ConstArrayView<Integer> m_offset;
    const Integer m_myrank;
  };


  void sortGhostCols([[maybe_unused]] ConstArrayView<Integer> offset)
  {
    //TODO
    /*
    IsLocal isLocal(offset, m_myrank);
    //UniqueArray<ValueType>& values = m_internal->getValues();
    auto& values = m_internal->getHostValues() ;;
    ProfileType& profile = getCSRProfile();
    UniqueArray<Integer>& cols = profile.getCols();
    ConstArrayView<Integer> kcol = profile.getRowOffset();
    Integer next = 0;
    UniqueArray<Integer> gcols;
    UniqueArray<ValueType> gvalues;
    for (Integer irow = 0; irow < m_local_size; ++irow) {
      bool need_sort = false;
      Integer first = next;
      next = kcol[irow + 1];
      Integer row_size = next - first;
      for (Integer k = first; k < next; ++k) {
        if (!isLocal(cols[k])) {
          need_sort = true;
          break;
        }
      }
      if (need_sort) {
        gvalues.resize(row_size);
        gcols.resize(row_size);
        Integer local_count = 0;
        Integer ghost_count = 0;
        for (Integer k = first; k < next; ++k) {
          Integer col = cols[k];
          if (isLocal(col)) {
            cols[first + local_count] = col;
            values[first + local_count] = values[k];
            ++local_count;
          }
          else {
            gcols[ghost_count] = col;
            gvalues[ghost_count] = values[k];
            ++ghost_count;
          }
        }
        for (Integer k = 0; k < ghost_count; ++k) {
          cols[first + local_count] = gcols[k];
          values[first + local_count] = gvalues[k];
          ++local_count;
        }
      }
    }
    */
  }


  // clang-format off
  Alien::BackEnd::Memory::eType m_mem_kind = Alien::BackEnd::Memory::Device;
  std::unique_ptr<ProfileType>            m_profile;
  std::unique_ptr<InternalType>           m_internal;
  //InternalType* m_internal = nullptr ;


  bool                                    m_is_parallel  = false;
  IMessagePassingMng*                     m_parallel_mng = nullptr;
  Integer                                 m_nproc        = 1;
  Integer                                 m_myrank       = 0;

  Integer                                 m_local_size   = 0;
  Integer                                 m_local_offset = 0;
  Integer                                 m_global_size  = 0;
  Integer                                 m_ghost_size   = 0;

  //SimpleCSRInternal::DistStructInfo            m_matrix_dist_info;
  DistStructInfo                               m_matrix_dist_info;
  ITraceMng*                                   m_trace = nullptr;
  // clang-format on


};

//extern template class SYCLBEllPackMatrix<double>;
} // namespace Alien
