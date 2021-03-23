// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ScopedPtr.h                                                 (C) 2000-2006 */
/*                                                                           */
/* Encapsulation d'un pointeur qui se détruit automatiquement.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SCOPEDPTR_H
#define ARCANE_UTILS_SCOPEDPTR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ptr.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Encapsulation d'un pointeur qui se détruit automatiquement.
 *
 Cette classe encapsule un pointeur sur un objet qui sera détruit (par
 l'intermédiaire de l'opérateur delete) lorsque l'instance de cette classe
 devient hors de portée.

 Cette classe est utile pour être sur qu'un objet sera désalloué même dans
 le cas où une exception survient.

 \since 0.4.40
 \author Gilles Grospellier
 \date 16/07/2001
 */
template<class T>
class ScopedPtrT
: public PtrT<T>
{
 public:

  //! Type de la classe de base
  typedef PtrT<T> BaseClass;

 public:

  //! Construit une instance sans référence
  ScopedPtrT() : BaseClass(0) {}

  //! Construit une instance référant \a t
  explicit ScopedPtrT(T* t) : BaseClass(t) {}

  //! Détruit l'objet référencé.
  ~ScopedPtrT() { delete this->m_value; }

 public:
  
  //! Opérateur de copie
  const ScopedPtrT<T>& operator=(const ScopedPtrT<T>& from)
    {
      if (this!=&from){
        delete this->m_value;
        BaseClass::operator=(from);
      }
      return (*this);
    }

  //! Affecte à l'instance la value \a new_value
  const ScopedPtrT<T>& operator=(T* new_value)
    {
      if (this->m_value!=new_value){
        delete this->m_value;
        this->m_value = new_value;
      }
      return (*this);
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

