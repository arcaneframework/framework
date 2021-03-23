// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ObjectImpl.h                                                (C) 2000-2017 */
/*                                                                           */
/* Classe de base de l'implémentation d'un objet d'Arcane.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_OBJECTIMPL_H
#define ARCANE_UTILS_OBJECTIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Atomic.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'un objet avec compteur de référence.
 *
 * Ces objets sont gérés par compteur de référence.
 */
class ARCANE_UTILS_EXPORT ObjectImpl
{
 public:
  ObjectImpl() : m_ref_count(0) {}
  ObjectImpl(const ObjectImpl& rhs) : m_ref_count(rhs.m_ref_count){}
  virtual ~ObjectImpl(){}
  ObjectImpl& operator=(const ObjectImpl& rhs)
  {
    m_ref_count = rhs.m_ref_count;
    return *this;
  }
 public:
  //! Incrémente le compteur de référence
  inline void addRef()
  {
    ++m_ref_count;
  }
  //! Décrémente le compteur de référence
  inline void removeRef()
  {
    Int32 r = --m_ref_count;
    if (r<0)
      arcaneNoReferenceError(this);
    if (r==0)
      deleteMe();
  }
  //! Retourne la valeur du compteur de référence
  Int32 refCount() const
  {
    return m_ref_count.value();
  }

 public:
  
  //! Détruit cet objet
  virtual void deleteMe()
  {
    delete this;
  }

 private:

  AtomicInt32 m_ref_count; //!< Nombre de références sur l'objet.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
