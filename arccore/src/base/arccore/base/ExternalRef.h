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
/* ExternalRef.h                                               (C) 2000-2019 */
/*                                                                           */
/* Gestion d'une référence sur un objet externe au C++.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_EXTERNALREF_H
#define ARCCORE_BASE_EXTERNALREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion des références sur un objet externe.
 *
 * Cette classe permet de sauver des références à des objets qui ne sont
 * pas gérés directement par %Arccore. C'est le cas par exemple lorsqu'on
 * utiliser le wrapping C# et qu'on souhaite manipuler  des
 * objets 'managé' par '.Net' et dont il n'existe plus
 * obligatoirement de référence explicite en mode managé. Cette classe permet
 * de maintenir une référence sur ces objets pour les empêcher d'être
 * récupéré par le ramasse-miette (Garbage collector).
 *
 * Cette classe est interne à %Arccore et ne doit en principe par être utilisée
 * directement.
 */
class ARCCORE_BASE_EXPORT ExternalRef
{
 private:
  struct Handle
  {
    Handle() : handle(nullptr){}
    Handle(void* h) : handle(h){}
    ~Handle();
    void addReference() { ++m_nb_ref; }
    void removeReference()
    {
      Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
      if (v==1)
        delete this;
    }
    void* handle;
    std::atomic<int> m_nb_ref = 0;
  };

 public:

  typedef void (*DestroyFuncType)(void* handle);

 public:

  ExternalRef() = default;
  ExternalRef(void* handle) : m_handle(new Handle(handle)){}

 public:
  bool isValid() const { return _internalHandle()!=nullptr; }
  void* _internalHandle() const { return m_handle->handle; }
 private:
  Arccore::ReferenceCounter<Handle> m_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
