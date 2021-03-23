// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Vector.cc                                                   (C) 2000-2014 */
/*                                                                           */
/* Vecteur d'algèbre linéraire.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"

#include "arcane/matvec/Vector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
namespace MatVec
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VectorImpl
{
 public:
  VectorImpl(Integer size)
  : m_values(size), m_nb_reference(0)
  {
  }
  VectorImpl(Integer size,Real init_value)
  : m_values(size,init_value), m_nb_reference(0)
  {
    m_values.fill(init_value);
  }
  VectorImpl() : m_nb_reference(0)
  {
  }
  VectorImpl(RealConstArrayView v)
  : m_values(v), m_nb_reference(0)
  {
  }
  VectorImpl(const VectorImpl& rhs)
  : m_values(rhs.m_values), m_nb_reference()
  {
  }
 private:
  void operator=(const VectorImpl& rhs);
 public:
  Integer size() const { return m_values.size(); }
  RealArrayView values() { return m_values; }
  RealConstArrayView values() const { return m_values; }
  void dump(std::ostream& o) const
  {
    Integer size = m_values.size();
    o << "(Vector ptr=" << this << " size=" << size << ")\n";
    for( Integer i=0; i<size; ++i )
      o << "[" << i << "]=" << m_values[i] << '\n';
    //o << ")";
  }
  VectorImpl* clone()
  {
    return new VectorImpl(m_values);
  }
  void copy(const VectorImpl& rhs)
  {
    m_values.copy(rhs.m_values);
  }
  Real normInf();
  static Vector readHypre(const String& file_name);
 public:
  void addReference()
  {
    ++m_nb_reference;
  }
  void removeReference()
  {
    --m_nb_reference;
    if (m_nb_reference==0)
      delete this;
  }
 private:
  UniqueArray<Real> m_values;
  Integer m_nb_reference;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Vector::
Vector(Integer size)
: m_impl(new VectorImpl(size))
{
  m_impl->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Vector::
Vector(const Vector& v)
: m_impl(v.m_impl)
{
  m_impl->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Vector& Vector::
operator=(const Vector& rhs)
{
  VectorImpl* vi = rhs.m_impl;
  vi->addReference();
  m_impl->removeReference();
  m_impl = vi;
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Vector::
~Vector()
{
  m_impl->removeReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Vector::
size() const
{
  return m_impl->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealArrayView Vector::
values()
{
  return m_impl->values();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealConstArrayView Vector::
values() const
{
  return m_impl->values();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Vector::
copy(const Vector& rhs)
{
  return m_impl->copy(*rhs.m_impl);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Vector::
dump(std::ostream& o) const
{
  m_impl->dump(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real VectorImpl::
normInf()
{
  Real v = 0.0;
  Integer size = m_values.size();
  for( Integer i=0; i<size; ++i ){
    Real v2 = math::abs(m_values[i]);
    if (v2>v)
      v = v2;
  }
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real Vector::
normInf()
{
  return m_impl->normInf();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Vector Vector::
readHypre(const String& file_name)
{
  std::ifstream ifile(file_name.localstr());
  Integer xmin = 0;
  Integer xmax = 0;
  ifile >> ws >> xmin >> ws >> xmax;
  Integer nb = (xmax-xmin)+1;
  Vector vec(nb);
  RealArrayView values = vec.values();
  for( Integer i=0; i<nb; ++i ){
    Integer column_id = 0;
    Real v = 0.0;
    ifile >> ws >> column_id >> ws >> v;
    values[column_id] = v;
  }
  return vec;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
