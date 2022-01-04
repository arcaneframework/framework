// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedReference.h                                           (C) 2000-2008 */
/*                                                                           */
/* Classe de base d'un compteur de référence.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SHAREDREFERENCE_H
#define ARCANE_SHAREDREFERENCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Atomic.h"
#include "arcane/ISharedReference.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Implémentation d'un compteur de référence.
 */
class ARCANE_CORE_EXPORT SharedReference
: public ISharedReference
{
 public:

  SharedReference() : m_ref_count(0) {}
	
 public:
	
  virtual void addRef();
  virtual void removeRef();
  virtual Int32 refCount() const;
	
  //! Détruit l'objet référencé
  virtual void deleteMe() =0;

 private:

  AtomicInt32 m_ref_count; //!< Nombre de références sur l'objet.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

