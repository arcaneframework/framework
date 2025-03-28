// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckedPointer.h                                            (C) 2000-2025 */
/*                                                                           */
/* Classes encapsulant un pointeur permettant de vérifier l'utilisation.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_CHECKEDPOINTER_H
#define ARCCORE_BASE_CHECKEDPOINTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulation d'un pointeur.
 *
 * Cette classe ne fait rien de particulier si ce n'est encapsulé un
 * pointeur d'un type quelconque. Elle sert de classe de base à d'autres
 * classes qui fournissent des fonctionnalités plus évoluées comme AutoRefT.
 *
 * Afin d'éviter des copies malencontreuses, le constructeur de copie et
 * les opérateurs de copie sont protégés.
 *
 * En mode débug, vérifie qu'on accède pas à un pointeur nul.
 *
 * Le paramètre template n'a pas besoin d'être défini. Cette classe peut donc
 * être instanciée pour un type opaque.
 */
template<class T>
class CheckedPointer
{
 protected:

  //! Opérateur de copie
  const CheckedPointer<T>& operator=(const CheckedPointer<T>& from)
  {
    m_value = from.m_value;
    return (*this);
  }
  
  template <typename T2>
  const CheckedPointer<T>& operator=(const CheckedPointer<T2>& from)
  {
    m_value = from.get();
    return (*this);
  }


  //! Affecte à l'instance la value \a new_value
  const CheckedPointer<T>& operator=(T* new_value)
  {
    m_value = new_value; return (*this);
  }

  //! Construit une référence référant \a from
  CheckedPointer(const CheckedPointer<T>& from) : m_value(from.m_value) {}

  //! Construit une référence référant \a from
  template <typename T2>
  CheckedPointer(const CheckedPointer<T2>& from) : m_value(from.m_value) {}

 public:

  //! Construit une instance sans référence
  CheckedPointer() : m_value(nullptr) {}

  //! Construit une instance référant \a t
  explicit CheckedPointer(T* t) : m_value(t) {}

 public:
  explicit operator bool() const { return get()!=nullptr; }
 public:
  
  //! Retourne l'objet référé par l'instance
  inline T* operator->() const
    {
#ifdef ARCCORE_CHECK
      if (!m_value)
        arccoreNullPointerError();
#endif
      return m_value;
    }

  //! Retourne l'objet référé par l'instance
  inline T& operator*() const
    {
#ifdef ARCCORE_CHECK
      if (!m_value)
        arccoreNullPointerError();
#endif
      return *m_value;
    }

  /*!
   * \brief Retourne l'objet référé par l'instance
   *
   * \warning En général, il faut être prudent lorsqu'on utilise cette
   * fonction et ne pas conservé le pointeur retourné.
   */
  inline T* get() const
  {
    return m_value;
  }

  inline bool isNull() const
  {
    return (!m_value);
  }


  /*!
   * \brief Compare les objets référencés par \a v1 et \a v2
   * La comparaison se fait pointeur par pointeur.
   * \retval true s'ils sont égaux
   * \retval false sinon
   */
  template<typename T2> friend bool
  operator==(const CheckedPointer<T>& v1,const CheckedPointer<T2>& v2)
  {
    return v1.get() == v2.get();
  }

  /*!
   * \brief Compare les objets référencés par \a v1 et \a v2
   * La comparaison se fait pointeur par pointeur.
   * \retval false s'ils sont égaux
   * \retval true sinon
   */
  template<typename T2> friend bool
  operator!=(const CheckedPointer<T>& v1,const CheckedPointer<T2>& v2)
  {
    return v1.get() != v2.get();
  }

 protected:
  
  T* m_value; //!< Pointeur sur l'objet référencé
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

