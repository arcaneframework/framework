// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* StringView.h                                                (C) 2000-2019 */
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
 * Cette classe est similaire à std::string_view dans le sens où elle ne
 * conserve qu'un pointeur sur une donnée mémoire gérée par une autre classe.
 * Les instances de cette classe ne doivent donc pas être conservées.
 *
 * L'encodage utilisé est toujours UTF-8.
 */
class ARCCORE_BASE_EXPORT StringView
{
 public:

  friend ARCCORE_BASE_EXPORT bool operator==(StringView a,StringView b);
  friend ARCCORE_BASE_EXPORT bool operator<(StringView a,StringView b);

 public:

  //! Crée une vue sur une chaîne vide
  StringView() {}
  //! Créé une vue à partir de \a str codé en UTF-8
  StringView(const char* str) : StringView(std::string_view(str)){}
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  StringView(std::string_view str)
  : m_v(reinterpret_cast<const Byte*>(str.data()),str.size()){}
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  StringView(Span<const Byte> str) : m_v(str){}

  //! Copie la vue \a str dans cette instance.
  constexpr StringView& operator=(const StringView& str) = default;
  //! Créé une vue à partir de \a str codé en UTF-8
  const StringView& operator=(const char* str)
  {
    operator=(std::string_view(str));
    return (*this);
  }
  //! Créé une vue à partir de \a str codé en UTF-8
  const StringView& operator=(std::string_view str)
  {
    m_v = Span<const Byte>(reinterpret_cast<const Byte*>(str.data()),str.size());
    return (*this);
  }
  //! Créé une vue à partir de \a str codé en UTF-8
  const StringView& operator=(Span<const Byte> str)
  {
    m_v = str;
    return (*this);
  }

  ~StringView() = default; //!< Libère les ressources.

 public:

 public:

  /*!
   * \brief Retourne la conversion de l'instance dans l'encodage UTF-8.
   *
   * L'instance retournée ne contient pas de zéro terminal.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance.
   */
  Span<const Byte> bytes() const { return m_v; }

  //! Longueur en octet de la chaîne de caractères.
  Int64 length() const { return m_v.size(); }

  //! Longueur en octet de la chaîne de caractères.
  Int64 size() const { return m_v.size(); }

 public:

  /*!
   * \brief Retourne une vue de la STL de la vue actuelle.
   */
  std::string_view toStdStringView() const
  {
    return std::string_view(reinterpret_cast<const char*>(m_v.data()),m_v.size());
  }

 public:

  //! Écrit la chaîne au format UTF-8 sur le flot \a o
  void writeBytes(std::ostream& o) const;

 public:

 private:

  Span<const Byte> m_v;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Opérateur d'écriture d'une StringView
ARCCORE_BASE_EXPORT std::ostream& operator<<(std::ostream& o,const StringView&);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compare deux vues.
 * \retval true si elles sont égales,
 * \retval false sinon.
 * \relate String
 */
extern "C++" ARCCORE_BASE_EXPORT bool operator==(StringView a,StringView b);
/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont différentes,
 * \retval false si elles sont égales.
 * \relate String
 */
inline bool operator!=(StringView a,StringView b)
{
  return !operator==(a,b);
}

/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont égales,
 * \retval false sinon.
 * \relate String
 */
extern "C++" ARCCORE_BASE_EXPORT bool operator==(const char* a,StringView b);

/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont différentes,
 * \retval false si elles sont égales.
 * \relate String
 */
inline bool operator!=(const char* a,StringView b)
{
  return !operator==(a,b);
}

/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont égales,
 * \retval false sinon.
 * \relate String
 */
extern "C++" ARCCORE_BASE_EXPORT bool operator==(StringView a,const char* b);
/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont différentes,
 * \retval false si elles sont égales.
 * \relate String
 */
inline bool operator!=(StringView a,const char* b)
{
  return !operator==(a,b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si a<b
 * \retval false sinon.
 * \relate String
 */
extern "C++" ARCCORE_BASE_EXPORT bool operator<(StringView a,StringView b);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
