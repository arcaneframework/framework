// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/data/IMatrix.h>
#include <alien/data/utils/MatrixElement.h>
#include <span>

#include <alien/handlers/scalar/BaseProfiledMatrixBuilder.h>

namespace Alien
{

  template <typename Scalar>
  class HCSRMatrix;

  namespace SYCL
  {

    template <typename ValueT, typename IndexT>
    class ALIEN_EXPORT ProfiledMatrixBuilderT
    {
     public:
      using ResetFlag = ProfiledMatrixOptions::ResetFlag;

     public:
      using MatrixElement = MatrixElementT<ProfiledMatrixBuilderT<ValueT,IndexT>>;

      typedef ValueT ValueType ;

      class Impl ;

      class View ;

      class ConstView ;

      class HostView ;


     public:
      ProfiledMatrixBuilderT(IMatrix& matrix, ResetFlag reset_values);

      virtual ~ProfiledMatrixBuilderT();

      ProfiledMatrixBuilderT(const ProfiledMatrixBuilderT&) = delete;
      ProfiledMatrixBuilderT(ProfiledMatrixBuilderT&&) = delete;
      ProfiledMatrixBuilderT& operator=(const ProfiledMatrixBuilderT&) = delete;
      ProfiledMatrixBuilderT& operator=(ProfiledMatrixBuilderT&&) = delete;

     public:
      inline MatrixElement operator()(const Integer iIndex, const Integer jIndex)
      {
        return MatrixElement(iIndex, jIndex, *this);
      }

      View view(SYCLControlGroupHandler& cgh) ;

      ConstView constView(SYCLControlGroupHandler& cgh) const ;

      HostView hostView() const ;

      void finalize();

     private:
      bool isLocal(Integer jIndex) const
      {
        return (jIndex >= m_local_offset) && (jIndex < m_next_offset);
      }

      void _startTimer() {}
      void _stopTimer() {}

     private:
      IMatrix& m_matrix;
      HCSRMatrix<ValueType>* m_matrix_impl;
      std::unique_ptr<Impl> m_impl;

      Integer m_local_offset = 0;
      Integer m_local_size = 0;
      Integer m_next_offset= 0;
      ConstArrayView<Integer> m_row_starts;
      ConstArrayView<Integer> m_cols;
      ConstArrayView<Integer> m_local_row_size;
      ArrayView<ValueType> m_values;
      bool m_finalized = false;
    };


    /*---------------------------------------------------------------------------*/
    /*---------------------------------------------------------------------------*/

    typedef ProfiledMatrixBuilderT<Real,Integer> ProfiledMatrixBuilder ;

  } // namespace SYCL

}

