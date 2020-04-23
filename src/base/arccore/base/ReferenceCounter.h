// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ReferenceCounter.h                                          (C) 2000-2019 */
/*                                                                           */
/* Encapsulation d'un pointeur avec compteur de référence.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REFERENCECOUNTER_H
#define ARCCORE_BASE_REFERENCECOUNTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/CheckedPointer.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulation d'un pointeur avec compteur de référence.
 *
 Cette classe renferme un pointeur d'un type qui doit implémenter
 les méthodes suivantes:
 - addReference() pour ajouter une référence
 - removeReference() pour supprimer une référence.
 
 A la différence de std::shared_ptr, le compteur de référence est donc géré
 en interne par le type *T*.
 Cette classe n'effectue aucune action basée sur la valeur de compteur de référence.
 la destruction éventuelle de l'objet lorsque le compteur de référence arrive
 à zéro est gérée par l'objet lui même.
 */
template<class T>
class ReferenceCounter
: public CheckedPointer<T>
{
 public:

  //! Type de la classe de base
  typedef CheckedPointer<T> BaseClass;

  using BaseClass::m_value;

 public:

  //! Construit une instance sans référence
  ReferenceCounter() : BaseClass(nullptr) {}
  //! Construit une instance référant \a t
  explicit ReferenceCounter(T* t) : BaseClass(nullptr) { _changeValue(t); }
  //! Construit une référence référant \a from
  ReferenceCounter(const ReferenceCounter<T>& from)
  : BaseClass(nullptr) { _changeValue(from.m_value); }

  //! Opérateur de copie
  ReferenceCounter<T>& operator=(const ReferenceCounter<T>& from)
  {
    _changeValue(from.m_value);
    return (*this);
  }

  //! Affecte à l'instance la value \a new_value
  ReferenceCounter<T>& operator=(T* new_value)
  {
    _changeValue(new_value);
    return (*this);
  }

  //! Destructeur. Décrément le compteur de référence de l'objet pointé
  ~ReferenceCounter() { _removeRef(); }

 private:
	
  //! Supprimer une référence à l'objet encapsulé si non nul
  void _removeRef()
  {
    if (m_value)
      ReferenceCounterAccessor<T>::removeReference(m_value);
  }
  //! Change l'objet référencé en \a new_value
  void _changeValue(T* new_value)
  {
    if (m_value==new_value)
      return;
    // Toujours ajouter avant pour le cas où la nouvelle valeur
    // et l'ancienne seraient issues de la même instance.
    if (new_value)
      ReferenceCounterAccessor<T>::addReference(new_value);
    _removeRef();
    m_value = new_value;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

