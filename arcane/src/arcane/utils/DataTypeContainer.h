// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTypeContainer.h                                         (C) 2000-2022 */
/*                                                                           */
/* Conteneur contenant une instance d'une classe par type de donnée.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_DATATYPECONTAINER_H
#define ARCANE_UTILS_DATATYPECONTAINER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur contenant une instance d'une classe par type de donnée.
 *
 * La classe template \a Traits doit contenir un type \a InstanceType qui
 * indique pour chaque type du language le type de l'instance du conteneur.
 * Par exemple
 *
 * \code
 * template<typename DataType>
 * class Instancer
 * {
 *  public:
 *   typedef UniqueArray<DataType> InstanceType;
 * };
 * \endcode
 *
 * La méthode instance() permet de récupérer une référence sur l'instance
 * à partir de son type.
 */
template <template <typename DataType> class Traits>
class BuiltInDataTypeContainer
{
 public:

  typename Traits<char>::InstanceType& instance(char*) { return m_char; }
  typename Traits<signed char>::InstanceType& instance(signed char*) { return m_signed_char; }
  typename Traits<unsigned char>::InstanceType& instance(unsigned char*) { return m_unsigned_char; }
  typename Traits<short>::InstanceType& instance(short*) { return m_short; }
  typename Traits<unsigned short>::InstanceType& instance(unsigned short*) { return m_unsigned_short; }
  typename Traits<int>::InstanceType& instance(int*) { return m_int; }
  typename Traits<unsigned int>::InstanceType& instance(unsigned int*) { return m_unsigned_int; }
  typename Traits<long>::InstanceType& instance(long*) { return m_long; }
  typename Traits<unsigned long>::InstanceType& instance(unsigned long*) { return m_unsigned_long; }
  typename Traits<long long>::InstanceType& instance(long long*) { return m_long_long; }
  typename Traits<unsigned long long>::InstanceType& instance(unsigned long long*) { return m_unsigned_long_long; }
  typename Traits<float>::InstanceType& instance(float*) { return m_float; }
  typename Traits<double>::InstanceType& instance(double*) { return m_double; }
  typename Traits<long double>::InstanceType& instance(long double*) { return m_long_double; }

 public:

  typename Traits<char>::InstanceType m_char;
  typename Traits<signed char>::InstanceType m_signed_char;
  typename Traits<unsigned char>::InstanceType m_unsigned_char;
  typename Traits<short>::InstanceType m_short;
  typename Traits<unsigned short>::InstanceType m_unsigned_short;
  typename Traits<int>::InstanceType m_int;
  typename Traits<unsigned int>::InstanceType m_unsigned_int;
  typename Traits<long>::InstanceType m_long;
  typename Traits<unsigned long>::InstanceType m_unsigned_long;
  typename Traits<long long>::InstanceType m_long_long;
  typename Traits<unsigned long long>::InstanceType m_unsigned_long_long;
  typename Traits<float>::InstanceType m_float;
  typename Traits<double>::InstanceType m_double;
  typename Traits<long double>::InstanceType m_long_double;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur contenant une instance d'une classe par type de donnée %Arcane.
 */
template <template <typename DataType> class Traits>
class ArcaneDataTypeContainer
: public BuiltInDataTypeContainer<Traits>
{
 public:

  typedef BuiltInDataTypeContainer<Traits> Base;
  using Base::instance;

 public:

  typename Traits<APReal>::InstanceType& instance(APReal*) { return m_apreal; }
  typename Traits<Real2>::InstanceType& instance(Real2*) { return m_real2; }
  typename Traits<Real3>::InstanceType& instance(Real3*) { return m_real3; }
  typename Traits<Real2x2>::InstanceType& instance(Real2x2*) { return m_real2x2; }
  typename Traits<Real3x3>::InstanceType& instance(Real3x3*) { return m_real3x3; }
  typename Traits<HPReal>::InstanceType& instance(HPReal*) { return m_hpreal; }

 public:

  typename Traits<APReal>::InstanceType m_apreal;
  typename Traits<Real2>::InstanceType m_real2;
  typename Traits<Real3>::InstanceType m_real3;
  typename Traits<Real2x2>::InstanceType m_real2x2;
  typename Traits<Real3x3>::InstanceType m_real3x3;
  typename Traits<HPReal>::InstanceType m_hpreal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
