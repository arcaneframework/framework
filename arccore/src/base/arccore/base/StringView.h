// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringView.h                                                (C) 2000-2025 */
/*                                                                           */
/* Vue sur une chaîne de caractères UTF-8.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STRINGVIEW_H
#define ARCCORE_BASE_STRINGVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"
#include "arccore/base/Span.h"

#include <string_view>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une chaîne de caractères UTF-8.
 *
 * Cette classe est similaire à std::string_view du C++17 dans le sens où elle ne
 * conserve qu'un pointeur sur une donnée mémoire gérée par une autre classe.
 * Les instances de cette classe ne doivent donc pas être conservées. La
 * différence principale se situe au niveau de l'encodage qui est obligatoirement
 * UTF-8 avec cette classe.
 *
 * \note Comme la classe std::string_view, le tableau \a bytes() ne contient
 * pas nécessairement de zéro terminal. Cela signifie entre autre qu'il ne faut
 * donc pas utiliser cette classe pour passer des paramètres à des fonctions C.
 */
class ARCCORE_BASE_EXPORT StringView
{
 public:

  //! Crée une vue sur une chaîne vide
  StringView() = default;
  //! Créé une vue à partir de \a str codé en UTF-8. \a str peut être nul.
  StringView(const char* str) ARCCORE_NOEXCEPT
  : StringView(str ? std::string_view(str) : std::string_view()){}
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  StringView(std::string_view str) ARCCORE_NOEXCEPT
  : m_v(reinterpret_cast<const Byte*>(str.data()),str.size()){}
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  constexpr StringView(Span<const Byte> str) ARCCORE_NOEXCEPT
  : m_v(str){}
  //! Opérateur de recopie
  constexpr StringView(const StringView& str) = default;
  //! Copie la vue \a str dans cette instance.
  constexpr StringView& operator=(const StringView& str) = default;
  //! Créé une vue à partir de \a str codé en UTF-8
  StringView& operator=(const char* str) ARCCORE_NOEXCEPT
  {
    operator=(str ? std::string_view(str) : std::string_view());
    return (*this);
  }
  //! Créé une vue à partir de \a str codé en UTF-8
  StringView& operator=(std::string_view str) ARCCORE_NOEXCEPT
  {
    m_v = Span<const Byte>(reinterpret_cast<const Byte*>(str.data()),str.size());
    return (*this);
  }
  //! Créé une vue à partir de \a str codé en UTF-8
  constexpr StringView& operator=(Span<const Byte> str) ARCCORE_NOEXCEPT
  {
    m_v = str;
    return (*this);
  }

  ~StringView() = default; //!< Libère les ressources.

 public:

  /*!
   * \brief Retourne la conversion de l'instance dans l'encodage UTF-8.
   *
   * \warning L'instance retournée ne contient pas de zéro terminal.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance.
   */
  constexpr Span<const Byte> bytes() const ARCCORE_NOEXCEPT { return m_v; }

  //! Longueur en octet de la chaîne de caractères.
  constexpr Int64 length() const ARCCORE_NOEXCEPT { return m_v.size(); }

  //! Longueur en octet de la chaîne de caractères.
  constexpr Int64 size() const ARCCORE_NOEXCEPT { return m_v.size(); }

  //! Vrai si la chaîne est nulle ou vide.
  constexpr bool empty() const ARCCORE_NOEXCEPT { return size()==0; }

 public:

  /*!
   * \brief Retourne une vue de la STL de la vue actuelle.
   */
  std::string_view toStdStringView() const ARCCORE_NOEXCEPT
  {
    return std::string_view(reinterpret_cast<const char*>(m_v.data()),m_v.size());
  }

  //! Opérateur d'écriture d'une StringView
  friend ARCCORE_BASE_EXPORT std::ostream& operator<<(std::ostream& o,const StringView&);

  /*!
   * \brief Compare deux vues.
   * \retval true si elles sont égales,
   * \retval false sinon.
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const StringView& a,const StringView& b);

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont différentes,
   * \retval false si elles sont égales.
   * \relate String
   */
  friend inline bool operator!=(const StringView& a,const StringView& b)
  {
    return !operator==(a,b);
  }

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont égales,
   * \retval false sinon.
   * \relate String
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const char* a,const StringView& b);

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont différentes,
   * \retval false si elles sont égales.
   * \relate String
   */
  friend bool operator!=(const char* a,const StringView& b){ return !operator==(a,b); }

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont égales,
   * \retval false sinon.
   * \relate String
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const StringView& a,const char* b);

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont différentes,
   * \retval false si elles sont égales.
   * \relate String
   */
  friend inline bool operator!=(const StringView& a,const char* b)
  {
    return !operator==(a,b);
  }

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si a<b
   * \retval false sinon.
   * \relate String
   */
  friend ARCCORE_BASE_EXPORT bool operator<(const StringView& a,const StringView& b);

 public:

  //! Écrit la chaîne au format UTF-8 sur le flot \a o
  void writeBytes(std::ostream& o) const;

  //! Sous-chaîne commençant à la position \a pos
  StringView substring(Int64 pos) const;

  //! Sous-chaîne commençant à la position \a pos et de longueur \a len
  StringView substring(Int64 pos,Int64 len) const;

 private:

  Span<const Byte> m_v;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
