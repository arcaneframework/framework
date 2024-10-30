// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RealArray2Variant.h                                         (C) 2000-2024 */
/*                                                                           */
/* Variant pouvant contenir les types ConstArray2View, Real2x2 et Real3x3.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_REALARRAY2VARIANT_H
#define ARCANE_DATATYPE_REALARRAY2VARIANT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array2.h"
#include "arcane/utils/Array2View.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#if defined(ARCANE_HAS_ACCELERATOR_API)
#include "arcane/utils/NumArray.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Variant pouvant contenir les types ConstArray2View, Real2x2 et Real3x3.
 */
class RealArray2Variant
{
 public:

  static const Integer MAX_DIM1_SIZE = 3;
  static const Integer MAX_DIM2_SIZE = 3;

  RealArray2Variant() = default;
  RealArray2Variant(UniqueArray2<Real> v)
  : RealArray2Variant(v.constView())
  {}
  RealArray2Variant(ConstArray2View<Real> v)
  {
    _setValue(v.data(), v.dim1Size(), v.dim2Size());
  }
  RealArray2Variant(Real2x2 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 2, 2);
  }
  RealArray2Variant(Real3x3 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 3, 3);
  }

#if defined(ARCANE_HAS_ACCELERATOR_API)
  template<typename LayoutType>
  RealArray2Variant(const NumArray<Real,MDDim2,LayoutType>& v)
  : RealArray2Variant(v.mdspan())
  {}
  template<typename LayoutType>
  RealArray2Variant(MDSpan<Real,MDDim2,LayoutType> v)
  {
    _setValue(v.to1DSpan().data(), v.extent0(), v.extent1());
  }
  template<typename LayoutType>
  RealArray2Variant(MDSpan<const Real,MDDim2,LayoutType> v)
  {
    _setValue(v.to1DSpan().data(), v.extent0(), v.extent1());
  }
#endif

  RealArray2Variant& operator=(const RealArray2Variant& rhs) = default;
  RealArray2Variant& operator=(ConstArray2View<Real> v)
  {
    _setValue(v.data(), v.dim1Size(), v.dim2Size());
    return (*this);
  }
  RealArray2Variant& operator=(Real2x2 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 2, 2);
    return (*this);
  }
  RealArray2Variant& operator=(Real3x3 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 3, 3);
    return (*this);
  }
  
  Real* operator[](Integer index) 
  { 
     ARCANE_ASSERT(index < m_nb_dim1, ("Index out of range"));
     return m_value[index];
  }
  const Real* operator [](Integer index) const
  { 
     ARCANE_ASSERT(index < m_nb_dim1, ("Index out of range"));
     return m_value[index];
  }

  Real& operator()(Int32 i,Int32 j)
  { 
     ARCANE_ASSERT(i < m_nb_dim1, ("Index i out of range"));
     ARCANE_ASSERT(j < m_nb_dim2, ("Index j out of range"));
     return m_value[i][j];
  }
  Real operator()(Int32 i,Int32 j) const
  { 
    ARCANE_ASSERT(i < m_nb_dim1, ("Index i out of range"));
    ARCANE_ASSERT(j < m_nb_dim2, ("Index j out of range"));
    return m_value[i][j];
  }

  Int32 dim1Size() const { return m_nb_dim1; }
  Int32 dim2Size() const { return m_nb_dim2; }
  Real* data() { return reinterpret_cast<Real*>(&m_value[0]); }
  const Real* data() const { return reinterpret_cast<const Real*>(m_value); }
  operator ConstArray2View<Real>() const
  {
      return ConstArray2View<Real>(data(), m_nb_dim1, m_nb_dim2);
  }
  operator Real2x2() const
  {
    return Real2x2::fromLines(m_value[0][0], m_value[0][1], m_value[1][0], m_value[1][1]);
  }
  operator Real3x3() const
  {
    return Real3x3::fromLines(m_value[0][0], m_value[0][1], m_value[0][2], 
                              m_value[1][0], m_value[1][1], m_value[1][2],
                              m_value[2][0], m_value[2][1], m_value[2][2]);
  }
#if defined(ARCANE_HAS_ACCELERATOR_API)
  template<typename LayoutType>
  operator NumArray<Real,MDDim2,LayoutType>() const
  {
    NumArray<Real,MDDim2> v(m_nb_dim1,m_nb_dim2);
    for( Integer i=0, m=m_nb_dim1; i<m; ++i )
      for( Integer j=0, n=m_nb_dim2; j<n; ++j )
        v(i,j) = m_value[i][j];
    return v;
  }
#endif

 private:

  Real m_value[MAX_DIM1_SIZE][MAX_DIM2_SIZE];
  Int32 m_nb_dim1 = 0;
  Int32 m_nb_dim2 = 0;

 private:

  void _setValue(const Real* v, Integer nb_dim1, Integer nb_dim2)
  {
    m_nb_dim1 = nb_dim1;
    ARCANE_ASSERT(nb_dim1 <= MAX_DIM1_SIZE, ("Dim1 size too large"));
    m_nb_dim2 = nb_dim2;
    ARCANE_ASSERT(nb_dim2 <= MAX_DIM2_SIZE, ("Dim2 size too large"));
    for (Integer i = 0 ; i < nb_dim1; ++i)
      for (Integer j = 0 ; j < nb_dim2; ++j)
        m_value[i][j] = v[i * nb_dim2 + j];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
