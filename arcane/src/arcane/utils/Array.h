// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array.h                                                     (C) 2000-2018 */
/*                                                                           */
/* Tableau 1D.                                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAY_H
#define ARCANE_UTILS_ARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/Array.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class ArrayFullAccessorT
{
 public:
  ArrayFullAccessorT(Array<T>& v) : m_array(&v) {}
  ~ArrayFullAccessorT() {}
 public:
  T operator[](Integer i) const { return m_array->item(i); }
  T& operator[](Integer i) { return (*m_array)[i]; }
  Integer size() const { return m_array->size(); }
  void resize(Integer s){ m_array->resize(s); }
  void add(T v){ m_array->add(v); }
 private:
  Array<T>* m_array;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique à \a ids un remplissage en fin de tableau.
 *
 * Cette méthode remplit les éléments de \a ids après la dernière valeur
 * pour que \a ids ait un nombre d'éléments valide multiple de la taille
 * d'un vecteur Simd.
 *
 * \a ids doit utiliser l'allocateur AlignedMemoryAllocator::Simd().
 * Le remplissage se fait avec comme valeur celle du dernier élément
 * valide de \a ids.
 *
 * Par exemple, si ids.size()==5 et que la taille de vecteur Simd est de 8,
 * alors ids[5], ids[6] et ids[7] sont remplis avec la valeur de ids[4].
 */
//@{
extern ARCANE_UTILS_EXPORT void
applySimdPadding(Array<Int32>& ids);

extern ARCANE_UTILS_EXPORT void
applySimdPadding(Array<Int16>& ids);

extern ARCANE_UTILS_EXPORT void
applySimdPadding(Array<Int64>& ids);

extern ARCANE_UTILS_EXPORT void
applySimdPadding(Array<Real>& ids);
//@}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
