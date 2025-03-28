// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExternalRef.h                                               (C) 2000-2025 */
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

namespace Arcane::Internal
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
    Handle()
    : handle(nullptr)
    {}
    Handle(void* h)
    : handle(h)
    {}
    ~Handle();
    void addReference() { ++m_nb_ref; }
    void removeReference()
    {
      Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
      if (v == 1)
        delete this;
    }
    void* handle;
    std::atomic<int> m_nb_ref = 0;
  };

 public:

  typedef void (*DestroyFuncType)(void* handle);

 public:

  ExternalRef() = default;
  ExternalRef(void* handle)
  : m_handle(new Handle(handle))
  {}

 public:

  bool isValid() const
  {
    Handle* p = m_handle.get();
    if (!p)
      return false;
    return _internalHandle() != nullptr;
  }
  void* _internalHandle() const { return m_handle->handle; }

 private:

  Arccore::ReferenceCounter<Handle> m_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::Internal
{
using Arcane::Internal::ExternalRef;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
