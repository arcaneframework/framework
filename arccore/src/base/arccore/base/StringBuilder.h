// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringBuilder.h                                             (C) 2000-2025 */
/*                                                                           */
/* Constructeur de chaîne de caractère unicode.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STRINGBUILDER_H
#define ARCCORE_BASE_STRINGBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <string>
#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//class String;
//class StringImpl;
//class StringFormatterArg;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Constructeur de chaîne de caractère unicode.
 *
 * Permet de construire de manière efficace une chaîne de caractère
 * par concaténation.
 *
 * \not_thread_safe
 */
class ARCCORE_BASE_EXPORT StringBuilder
{
 public:

  //! Crée une chaîne nulle
  StringBuilder() : m_p(nullptr), m_const_ptr(nullptr) {}
  //! Créé une chaîne à partir de \a str dans l'encodage local
  StringBuilder(const char* str);
  //! Créé une chaîne à partir de \a str dans l'encodage local
  StringBuilder(const char* str,Integer len);
  //! Créé une chaîne à partir de \a str dans l'encodage local
  StringBuilder(const std::string& str);
  //! Créé une chaîne à partir de \a str dans l'encodage Utf16
  StringBuilder(const UCharConstArrayView& ustr);
  //! Créé une chaîne à partir de \a str dans l'encodage Utf8
  StringBuilder(const ByteConstArrayView& ustr);
  //! Créé une chaîne à partir de \a str_builder
  StringBuilder(const StringBuilder& str_builder);
  //! Créé une chaîne à partir de \a str dans l'encodage local
  explicit StringBuilder(StringImpl* impl);
  //! Créé une chaîne à partir de \a str
  StringBuilder(const String& str);

  //! Copie \a str dans cette instance.
  const StringBuilder& operator=(const String& str);
  //! Copie \a str dans cette instance.
  const StringBuilder& operator=(const char* str);
  //! Copie \a str dans cette instance.
  const StringBuilder& operator=(const StringBuilder& str);

  ~StringBuilder(); //!< Libère les ressources.

 public:

  /*!
   * \brief Retourne la chaîne de caractères construite.
   */
  operator String() const;

  /*!
   * \brief Retourne la chaîne de caractères construite.
   */
  String toString() const;

 public:

  //! Ajoute \a str.
  StringBuilder& append(const String& str);

  //! Clone cette chaîne.
  StringBuilder clone() const;

  /*! \brief Effectue une normalisation des caractères espaces.
   * Tous les caractères espaces sont remplacés par des blancs espaces #x20,
   * à savoir #xD (Carriage Return), #xA (Line Feed) et #x9 (Tabulation).
   * Cela correspond à l'attribut xs:replace de XMLSchema 1.0
   */
  StringBuilder& replaceWhiteSpace();

  /*! \brief Effectue une normalisation des caractères espaces.
   * Le comportement est identique à replaceWhiteSpace() avec en plus:
   * - remplacement de tous les blancs consécutifs par un seul.
   * - suppression des blancs en début et fin de chaîne.
   * Cela correspond à l'attribut xs:collapse de XMLSchema 1.0
   */
  StringBuilder& collapseWhiteSpace();

  //! Transforme tous les caracteres de la chaine en majuscules.
  StringBuilder& toUpper();

  //! Transforme tous les caracteres de la chaine en minuscules.
  StringBuilder& toLower();

  void operator+=(const char* str);
  void operator+=(const String& str);
  void operator+=(unsigned long v);
  void operator+=(unsigned int v);
  void operator+=(double v);
  void operator+=(long double v);
  void operator+=(int v);
  void operator+=(char v);
  void operator+=(long v);
  void operator+=(unsigned long long v);
  void operator+=(long long v);
  void operator+=(const APReal& v);

 public:

  friend ARCCORE_BASE_EXPORT bool operator==(const StringBuilder& a,const StringBuilder& b);
  friend bool operator!=(const StringBuilder& a,const StringBuilder& b)
  {
    return !operator==(a,b);
  }

 public:

  /*!
   * \brief Affiche les infos internes de la classe.
   *
   * Cette méthode n'est utile que pour débugger %Arccore
   */
  void internalDump(std::ostream& ostr) const;

 private:

  mutable StringImpl* m_p = nullptr; //!< Implémentation de la classe
  mutable const char* m_const_ptr = nullptr;

  void _checkClone() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Opérateur d'écriture d'une StringBuilder
ARCCORE_BASE_EXPORT std::ostream& operator<<(std::ostream& o,const StringBuilder&);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
