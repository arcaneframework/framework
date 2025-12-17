// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedReference.h                                           (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'un compteur de référence.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SHAREDREFERENCE_H
#define ARCANE_CORE_SHAREDREFERENCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISharedReference.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Implémentation d'un compteur de référence utilisant std::atomic.
 */
class ARCANE_CORE_EXPORT SharedReference
: public ISharedReference
{
 public:

  SharedReference() : m_ref_count(0) {}
	
 public:
	
  void addRef() override;
  void removeRef() override;
  Int32 refCount() const override { return m_ref_count; }

  //! Détruit l'objet référencé
  virtual void deleteMe() =0;

 private:

  std::atomic<Int32> m_ref_count; //!< Nombre de références sur l'objet.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

