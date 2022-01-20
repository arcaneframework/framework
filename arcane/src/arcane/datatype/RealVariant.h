// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RealVariant.h                                               (C) 2000-2022 */
/*                                                                           */
/* Variant pouvant contenir les types 'Real*'.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_REALVARIANT_H
#define ARCANE_DATATYPE_REALVARIANT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Variant pouvant contenir les types 'Real*'.
 *
 * Les types possibles sont Real, Real2, Real3, Real2x2 et Real3x3.
 */
class RealVariant
{
 public:

  RealVariant() = default;
  RealVariant(const Real* v, Int32 nb_value)
  {
    _setValue(v, nb_value);
  }
  RealVariant(ConstArrayView<Real> v)
  : RealVariant(v.data(), v.size())
  {}
  RealVariant(UniqueArray<Real> v)
  : RealVariant(v.data(), v.size())
  {}
  RealVariant(Real r)
  : RealVariant(reinterpret_cast<Real*>(&r), 1)
  {}
  RealVariant(Real2 r)
  : RealVariant(reinterpret_cast<Real*>(&r), 2)
  {}
  RealVariant(Real3 r)
  : RealVariant(reinterpret_cast<Real*>(&r), 3)
  {}
  RealVariant(Real2x2 r)
  : RealVariant(reinterpret_cast<Real*>(&r), 4)
  {}
  RealVariant(Real3x3 r)
  : RealVariant(reinterpret_cast<Real*>(&r), 9)
  {}

  RealVariant& operator=(const RealVariant& rhs) = default;

  RealVariant& operator=(Real r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 1);
    return (*this);
  }
  RealVariant& operator=(Real2 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 2);
    return (*this);
  }
  RealVariant& operator=(Real3 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 3);
    return (*this);
  }
  RealVariant& operator=(Real2x2 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 4);
    return (*this);
  }
  RealVariant& operator=(Real3x3 r)
  {
    _setValue(reinterpret_cast<Real*>(&r), 9);
    return (*this);
  }
  
  Real& operator [](Integer index) 
  { 
     ARCANE_ASSERT(index < nb_value, ("Index out of range"));
     return m_value[index];
  }
  const Real& operator [](Integer index) const
  { 
     ARCANE_ASSERT(index < nb_value, ("Index out of range"));
     return m_value[index];
  }

  Int32 size() const { return m_nb_value; }
  Real* data() { return m_value; }
  const Real* data() const { return m_value; }
  operator Real() const { return Real(m_value[0]); }
  operator Real2() const { return Real2(m_value[0], m_value[1]); }
  operator Real3() const { return Real3(m_value[0], m_value[1], m_value[2]); }
  operator Real2x2() const { return Real2x2::fromLines(m_value[0], m_value[1], m_value[2], m_value[3]); }
  operator Real3x3() const
  {
    return Real3x3::fromLines(m_value[0], m_value[1], m_value[2], m_value[3], m_value[4],
                              m_value[5], m_value[6], m_value[7], m_value[8]);
  }

 private:

  Real m_value[9];
  Int32 m_nb_value = 0;

 private:

  void _setValue(const Real v, Integer index)
  {
     ARCANE_ASSERT(index < nb_value, ("Index out of range"));
     m_value[index] = v;
  }

  void _setValue(const Real* v, Int32 nb_value)
  {
    m_nb_value = nb_value;
    ARCANE_ASSERT(nb_value <= 9, ("Size is too large"));
    for (Integer i = 0, n = nb_value; i < n; ++i)
      m_value[i] = v[i];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
