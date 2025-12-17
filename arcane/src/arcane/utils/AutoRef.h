// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AutoRef.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Encapsulation d'un pointeur avec compteur de référence.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_AUTOREF_H
#define ARCANE_UTILS_AUTOREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ptr.h"
#include "arccore/base/AutoRef2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulation d'un pointeur avec compteur de référence.
 *
 * Cette classe renferme un pointeur d'un type qui implémente les méthodes
 * de la classe abstraite ISharedReference (le paramètre template n'a pas
 * besoin de dériver de cette classe) et incrémente (_addRef()) ou décrémente
 * (_removeRef()) le compteur de référence de l'élément pointé lors des
 * affectations succéssives. Cette classe n'effectue aucune action basée
 * sur la valeur de compteur de référence; la destruction éventuelle de l'objet
 * lorsque le compteur de référence arrive à zéro est gérée par l'objet lui même.
 */
template <class T>
class AutoRefT
: public PtrT<T>
{
 public:

  //! Type de la classe de base
  using BaseClass = PtrT<T>;

  using BaseClass::m_value;

 public:

  //! Construit une instance sans référence
  AutoRefT() = default;
  //! Construit une instance référant \a t
  explicit AutoRefT(T* t)
  : BaseClass()
  {
    _changeValue(t);
  }
  //! Construit une référence référant \a from
  AutoRefT(const AutoRefT<T>& from)
  : BaseClass()
  {
    _changeValue(from.m_value);
  }

  //! Opérateur de copie
  AutoRefT<T>& operator=(const AutoRefT<T>& from)
  {
    _changeValue(from.m_value);
    return (*this);
  }

  //! Affecte à l'instance la value \a new_value
  AutoRefT<T>& operator=(T* new_value)
  {
    _changeValue(new_value);
    return (*this);
  }

  //! Destructeur. Décrément le compteur de référence de l'objet pointé
  ~AutoRefT() { _removeRef(); }

 private:

  //! Ajoute une référence à l'objet encapsulé si non nul
  void _addRef()
  {
    if (m_value)
      m_value->addRef();
  }
  //! Supprimer une référence à l'objet encapsulé si non nul
  void _removeRef()
  {
    if (m_value)
      m_value->removeRef();
  }
  //! Change l'objet référencé en \a new_value
  void _changeValue(T* new_value)
  {
    if (m_value == new_value)
      return;
    _removeRef();
    m_value = new_value;
    _addRef();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
