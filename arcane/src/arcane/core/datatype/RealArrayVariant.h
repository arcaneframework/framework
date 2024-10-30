// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RealArrayVariant.h                                          (C) 2000-2023 */
/*                                                                           */
/* Variant pouvant contenir les types ConstArrayView, Real2 et Real3.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_REALARRAYVARIANT_H
#define ARCANE_DATATYPE_REALARRAYVARIANT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
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
 * \brief Variant pouvant contenir les types ConstArrayView, Real2 et Real3.
 */
class RealArrayVariant
{
 public:

  static const Integer MAX_SIZE = 9;

  RealArrayVariant() = default;
  RealArrayVariant(UniqueArray<Real> v)
  : RealArrayVariant(v.constView())
  {}
  RealArrayVariant(ConstArrayView<Real> v)
  {
    _setValue(v.data(), v.size());
  }
#if defined(ARCANE_HAS_ACCELERATOR_API)
  template<typename LayoutType>
  RealArrayVariant(const NumArray<Real,MDDim1,LayoutType>& v)
  : RealArrayVariant(v.mdspan())
  {}
  template<typename LayoutType>
  RealArrayVariant(MDSpan<Real,MDDim1,LayoutType> v)
  {
    _setValue(v.to1DSpan().data(), v.extent0());
  }
  template<typename LayoutType>
  RealArrayVariant(MDSpan<const Real,MDDim1,LayoutType> v)
  {
    _setValue(v.to1DSpan().data(), v.extent0());
  }
#endif
  RealArrayVariant(Real2 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 2);
  }
  RealArrayVariant(Real3 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 3);
  }

  RealArrayVariant& operator=(const RealArrayVariant& rhs) = default;
  RealArrayVariant& operator=(ConstArrayView<Real> v)
  {
    _setValue(v.data(), v.size());
    return (*this);
  }
  RealArrayVariant& operator=(Real2 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 2);
    return (*this);
  }
  RealArrayVariant& operator=(Real3 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 3);
    return (*this);
  }
  
  Real& operator[](Integer index) 
  { 
    ARCANE_ASSERT(index < m_nb_value, ("Index out of range"));
    return m_value[index];
  }
  Real operator[](Integer index) const
  { 
    ARCANE_ASSERT(index < m_nb_value, ("Index out of range"));
    return m_value[index];
  }
  Real& operator()(Integer index) 
  { 
    ARCANE_ASSERT(index < m_nb_value, ("Index out of range"));
    return m_value[index];
  }
  Real operator()(Integer index) const
  { 
    ARCANE_ASSERT(index < m_nb_value, ("Index out of range"));
    return m_value[index];
  }

  Int32 size() const { return m_nb_value; }
  Real* data() { return m_value; }
  const Real* data() const { return m_value; }
  operator ConstArrayView<Real>() const { return ConstArrayView<Real>(m_nb_value, m_value); }
  operator Real2() const { return Real2(m_value[0], m_value[1]); }
  operator Real3() const { return Real3(m_value[0], m_value[1], m_value[2]); }

#if defined(ARCANE_HAS_ACCELERATOR_API)
  operator NumArray<Real,MDDim1>() const
  {
    NumArray<Real,MDDim1> v(m_nb_value);
    for( Integer i=0, n=m_nb_value; i<n; ++i )
      v[i] = m_value[i];
    return v;
  }
#endif

 private:

  Real m_value[MAX_SIZE];
  Int32 m_nb_value = 0;

 private:

  void _setValue(const Real* v, Int32 nb_value)
  {
    m_nb_value = nb_value;
    ARCANE_ASSERT(nb_value <= MAX_SIZE, ("Size is too large"));
    for (Integer i = 0 ; i < nb_value; ++i)
      m_value[i] = v[i];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
