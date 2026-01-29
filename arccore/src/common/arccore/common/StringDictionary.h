// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringDictionary.h                                          (C) 2000-2025 */
/*                                                                           */
/* Dictionnaire de chaînes de caractères.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_STRINGDICTIONARY_H
#define ARCCORE_COMMON_STRINGDICTIONARY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Dictionnaire de chaînes unicode.
 *
 * Maintient une liste des couples (clé,valeur) permettant d'associer une
 * chaîne de caractère à une autre. Ce type de dictionnaire est
 * utilisé par exemple pour les traductions, auquel cas la clé est
 * le langage et la valeur la traduction correspondante.
 */
class ARCCORE_COMMON_EXPORT StringDictionary
{
 private:

  class Impl; //!< Implémentation

 public:

  //! Construit un dictionnaire
  StringDictionary();
  //! Construit un dictionnaire
  StringDictionary(const StringDictionary& rhs);
  ~StringDictionary(); //!< Libère les ressources

 public:

  /*! \brief Ajoute le couple (key,value) au dictionnaire.
   *
   * Si une valeur existe déjà pour \a key, elle est remplacée par
   * la nouvelle.
   */
  void add(const String& key, const String& value);

  /*! \brief Supprime la valeur associée à \a key.
   *
   * Si aucune valeur n'était associée à \a key, rien ne se passe.
   * \return la valeur supprimée s'il y en a une.
   */
  String remove(const String& key);

  /*! \brief Retourne la valeur associée à \a key.
   *
   * Si aucune valeur n'est associée à \a key, la chaîne nulle est retournée.
   * Il n'est pas possible de faire la différence entre une valeur
   * correspondant à la chaîne nulle et une valeur non trouvée sauf si
   * \a throw_exception est vrai, auquel cas une exception est renvoyée
   * s'il n'existe pas de valeur correspondant à \a key.
   */
  String find(const String& key, bool throw_exception = false) const;

  //! Remplit \a keys et \a values avec les valeurs correspondantes du dictionnaire
  void fill(StringList& param_names, StringList& values) const;

 private:

  Impl* m_p; //!< Implémentation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
