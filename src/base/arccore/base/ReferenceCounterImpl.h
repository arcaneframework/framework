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
/* ReferenceCounterImpl.h                                      (C) 2000-2020 */
/*                                                                           */
/* Implémentations liées au gestionnaire de compteur de référence.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REFERENCECOUNTERIMPL_H
#define ARCCORE_BASE_REFERENCECOUNTERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

// Ce fichier ne doit être inclu que pour une instantation spécifique.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T> ARCCORE_IMPORT void
ExternalReferenceCounterAccessor<T>::
addReference(T* t)
{
  t->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T> ARCCORE_IMPORT void
ExternalReferenceCounterAccessor<T>::
removeReference(T* t)
{
  t->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ReferenceCounterImpl
{
 public:
  ReferenceCounterImpl() {}
  virtual ~ReferenceCounterImpl() = default;
 public:
  void addReference()
  {
    ++m_nb_ref;
  }
  void removeReference()
  {
    // Décrémente et retourne la valeur d'avant.
    // Si elle vaut 1, cela signifie qu'on n'a plus de références
    // sur l'objet et qu'il faut le détruire.
    Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
    if (v==1)
      delete this;
  }
 public:
  std::atomic<Int32> m_nb_ref = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! A mettre dans la définition de la classe implémentation
#define ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS()\
 public:\
  Arccore::ReferenceCounterImpl* _internalReferenceCounter() override { return this; } \
  void addReference() override { Arccore::ReferenceCounterImpl::addReference(); } \
  void removeReference() override { Arccore::ReferenceCounterImpl::removeReference(); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! A mettre dans le '.cc' d'une classe gérant un compteur de référence
#define ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(class_name)      \
template class ExternalReferenceCounterAccessor<class_name>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
