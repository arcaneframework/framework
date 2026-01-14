// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

namespace Alien {

template<typename MatrixT,typename VectorT>
class CxrOperator
{
public:
  using MatrixType = MatrixT ;
  using VectorType = VectorT ;


  CxrOperator(MatrixT const& matrix)
  : m_matrix(matrix)
  {}

  virtual ~CxrOperator()
  {}

  template<typename AlgebraT>
  void apply(AlgebraT& algebra, const VectorType &x, VectorType &y)
  {
    if(m_v.getAllocSize()==0)
      algebra.allocate(AlgebraT::resource(m_matrix),m_v);
    algebra.mult(m_matrix,x,m_v) ;

    get(algebra,m_v,y) ;
  }

  MatrixType const& getCxrMatrix() const {
    return m_cxr_matrix ;
  }

  MatrixType& getCxrMatrix() {
    return m_cxr_matrix ;
  }

  template<typename AlgebraT>
  void computeCxrMatrix(AlgebraT& algebra)
  {
    if constexpr (requires{algebra.computeCxr(m_matrix,m_cxr_matrix) ;})
    {
      m_block_size = algebra.computeCxr(m_matrix,m_cxr_matrix) ;
    }
    else
      throw Arccore::FatalErrorException(A_FUNCINFO, "Using Algebra that does not implemented computeCxr Op");
  }

  template<typename AlgebraT>
  void computeCxrMatrix(AlgebraT& algebra, VectorType const& diag)
  {
    m_block_size = algebra.computeCxr(m_matrix,m_cxr_matrix) ;
    {
      m_use_diag_scal = true ;
      algebra.copy(diag,m_cxr_diag_scal) ;
      algebra.multDiagScal(m_cxr_matrix,m_cxr_diag_scal) ;
    }
  }


  template<typename AlgebraT>
  void get(AlgebraT& algebra, VectorType const& x,VectorType& cxr_x)
  {
    if(m_use_diag_scal)
    {
      if(m_cxr_v.getAllocSize()==0)
        algebra.allocate(AlgebraT::resource(m_matrix),m_cxr_v);
      algebra.copy(x,m_block_size,m_cxr_v,1);
      algebra.pointwiseMult(m_cxr_diag_scal,m_cxr_v,cxr_x) ;
    }
    else
      algebra.copy(x,m_block_size,cxr_x,1);
  }

  template<typename AlgebraT>
  void combine(AlgebraT& algebra, VectorType const& cxr_x, VectorType& x)
  {
    algebra.axpy(1.,cxr_x,1,x,m_block_size);
  }


  template<typename AlgebraT>
  void copy(AlgebraT& algebra, VectorType const& cxr_x, VectorType& x)
  {
    algebra.copy(cxr_x,1,x,m_block_size);
  }

private:
  MatrixType const& m_matrix ;
  MatrixType        m_cxr_matrix ;
  VectorType        m_v;
  int               m_block_size = 1 ;

  bool              m_use_diag_scal = false;
  VectorType        m_cxr_diag_scal ;
  VectorType        m_cxr_v;

};

}
