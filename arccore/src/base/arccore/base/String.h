// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* String.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Chaîne de caractère unicode.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STRING_H
#define ARCCORE_BASE_STRING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/StringView.h"

#include <string>
#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//class StringFormatterArg;
//class StringBuilder;
//class StringImpl;
//class StringView;
//class StringUtilsImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Chaîne de caractères unicode.
 *
 * Cette classe permet de gérer une chaîne de caractères soit avec l'encodage
 * UTF-8 soit avec l'encodage UTF-16. A noter que l'encodage UTF-16 est
 * obsolète et sera supprimé dans une version ultérieure lorsque le C++20
 * sera disponible.
 *
 * Toutes les méthodes utilisant des `const char*` en arguments supposent
 * que l'encodage utilisé est en UTF-8.
 *
 * Les instances de cette classe sont immutables.
 *
 * Cette classe est similaire à std::string mais avec les différences
 * suivantes:
 * - la classe \a String utilise l'encodage UTF-8 alors que pour std::string
 *   l'encodage est indéfini.
 * - contrairement à std::string, il n'est pas possible actuellement de
 *   conserver des caractères nuls à l'intérieur d'une \a String.
 * - pour String, il y a une distinction entre la chaîne nulle et la chaîne vide.
 *   Le constructeur String::String() créé une chaîne nulle alors que
 *   String::String("") créé une chaîne vide. Si la chaîne est nulle,
 *   les appels à view() ou toStdStringView() retourne une chaîne vide.
 *
 * Lorsque le C++20 sera disponible, la classe \a String correspondra
 * au type std::optional<std::u8string>.
 *
 * Pour des raisons de performance, pour construire par morceaux une chaîne
 * de caractères, il est préférable d'utiliser la classe 'StringBuilder'.
 */
class ARCCORE_BASE_EXPORT String
{
 public:

  friend ARCCORE_BASE_EXPORT bool operator<(const String& a,const String& b);
  friend class StringBuilder;
  friend class StringUtilsImpl;

 public:

  //! Crée une chaîne nulle
  String() {}
  /*!
   * \brief Créé une chaîne à partir de \a str dans l'encodage UTF-8
   *
   * \warning Attention, la chaine est supposée constante sa validité
   * infinie (i.e il s'agit d'une chaîne constante à la compilation.
   * Si la chaîne passée en argument peut être désallouée,
   * il faut utiliser String(std::string_view) à la place.
   */   
  String(const char* str)
  : m_const_ptr(str)
  {
    if (str)
      m_const_ptr_size = std::string_view(str).size();
  }
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  String(char* str);
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  ARCCORE_DEPRECATED_2019("Use String::String(StringView) instead")
  String(const char* str,bool do_alloc);
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  ARCCORE_DEPRECATED_2019("Use String::String(StringView) instead")
  String(const char* str,Integer len);
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  String(std::string_view str);
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  String(StringView str);
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  String(const std::string& str);
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-16
  String(const UCharConstArrayView& ustr);
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  String(const Span<const Byte>& ustr);
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  //String(const Span<Byte>& ustr);
  //! Créé une chaîne à partir de \a str dans l'encodage UTF-8
  explicit String(StringImpl* impl);
  //! Créé une chaîne à partir de \a str
  String(const String& str);
  //! Créé une chaîne à partir de \a str
  String(String&& str)
  : m_p(str.m_p), m_const_ptr(str.m_const_ptr),  m_const_ptr_size(str.m_const_ptr_size)
  {
    str._resetFields();
  }

  //! Copie \a str dans cette instance.
  String& operator=(const String& str);
  //! Copie \a str dans cette instance.
  String& operator=(String&& str);
  //! Copie \a str dans cette instance.
  String& operator=(StringView str);
  /*!
   * \brief Référence \a str codé en UTF-8 dans cette instance.
   *
   * \warning Attention, la chaine est supposée constante sa validité
   * infinie (i.e il s'agit d'une chaîne constante à la compilation.
   * Si la chaîne passée en argument peut être désallouée,
   * il faut utiliser String::operator=(std::string_view) à la place.
   */
  String& operator=(const char* str)
  {
    m_const_ptr = str;
    m_const_ptr_size = 0;
    if (m_const_ptr)
      m_const_ptr_size = std::string_view(str).size();
    _removeReferenceIfNeeded();
    m_p = nullptr;
    return (*this);
  }
  //! Copie \a str codé en UTF-8 dans cette instance.
  String& operator=(std::string_view str);
  //! Copie \a str codé en UTF-8 dans cette instance.
  String& operator=(const std::string& str);

  //! Libère les ressources.
  ~String()
  {
    _removeReferenceIfNeeded();
  }

 public:

  /*!
   * \brief Retourne une vue sur la chaîne actuelle.
   *
   * L'encodage utilisé est UTF-8.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance. La vue
   * retournée ne doit pas être conservée.
   */
  operator StringView() const;

 public:

  static String fromUtf8(Span<const Byte> bytes);

 public:

  /*!
   * \brief Retourne la conversion de l'instance dans l'encodage UTF-16.
   *
   * Le tableau retourné contient toujours un zéro terminal si la chaîne n'est
   * par nulle. Par conséquent, la taille de toute chaîne non nulle est
   * celle du tableau moins 1.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance.
   *
   * \deprecated Il faut utiliser StringUtils::asUtf16BE() à la place. A noter que
   * la fonction StringUtils::asUtf16BE() ne contient pas de 0x00 terminal.
   */
  [[deprecated("Y2022: Use StringUtils::asUtf16BE() instead")]]
  ConstArrayView<UChar> utf16() const;

  /*!
   * \brief Retourne la conversion de l'instance dans l'encodage UTF-8.
   *
   * Le tableau retourné contient toujours un zéro terminal si la chaîne n'est
   * par nulle. Par conséquent, la taille de toute chaîne non nulle est
   * celle du tableau moins 1.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance.
   */
  ByteConstArrayView utf8() const;

  /*!
   * \brief Retourne la conversion de l'instance dans l'encodage UTF-8.
   *
   * \a bytes().size() correspond à la longueur de la chaîne de caractères mais
   * la vue retournée contient toujours un '\0' terminal.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance.
   */
  Span<const Byte> bytes() const;

  /*!
   * \brief Retourne la conversion de l'instance dans l'encodage UTF-8.
   *
   * Si null() est vrai, retourne la chaîne vide. Sinon, cette méthode est équivalent
   * à appeler bytes().data(). Il y a toujours un '\0' terminal à la fin de la
   * chaîne retournée.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance.
   */
  const char* localstr() const;

 public:

  /*!
   * \brief Retourne une vue de la STL sur la chaîne actuelle.
   *
   * L'encodage utilisé est UTF-8.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance. La vue
   * retournée ne doit pas être conservée.
   */
  std::string_view toStdStringView() const;

  /*!
   * \brief Retourne une vue sur la chaîne actuelle.
   *
   * L'encodage utilisé est UTF-8.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance. La vue
   * retournée ne doit pas être conservée.
   */
  StringView view() const;

 public:

  //! Clone cette chaîne.
  String clone() const;

  /*!
   * \brief Effectue une normalisation des caractères espaces.
   *
   * Tous les caractères espaces sont remplacés par des blancs espaces #x20,
   * à savoir #xD (Carriage Return), #xA (Line Feed) et #x9 (Tabulation).
   * Cela correspond à l'attribut xs:replace de XMLSchema 1.0
   */
  static String replaceWhiteSpace(const String& rhs);

  /*!
   * \brief Effectue une normalisation des caractères espaces.
   *
   * Le comportement est identique à replaceWhiteSpace() avec en plus:
   * - remplacement de tous les blancs consécutifs par un seul.
   * - suppression des blancs en début et fin de chaîne.
   * Cela correspond à l'attribut xs:collapse de XMLSchema 1.0
   */
  static String collapseWhiteSpace(const String& rhs);

  //! Transforme tous les caractères de la chaîne en majuscules.
  String upper() const;

  //! Transforme tous les caractères de la chaîne en minuscules.
  String lower() const;

  //! Retourne \a true si la chaîne est nulle.
  bool null() const;

  //! Retourne la longueur de la chaîne en 32 bits.
  ARCCORE_DEPRECATED_2019("Use method String::length() instead")
  Integer len() const;
  
  //! Retourne la longueur de la chaîne.
  Int64 length() const;

  //! Vrai si la chaîne est vide (nulle ou "")
  bool empty() const;

  //! Calcule une valeur de hashage pour cette chaîne de caractères
  Int32 hashCode() const;

  //! Écrit la chaîne au format UTF-8 sur le flot \a o
  void writeBytes(std::ostream& o) const;

 public:

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont égales,
   * \retval false sinon.
   * \relate String
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const String& a,const String& b);

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont différentes,
   * \retval false si elles sont égales.
   * \relate String
   */
  friend bool operator!=(const String& a,const String& b)
  {
    return !operator==(a,b);
  }

  //! Opérateur d'écriture d'une String
  friend ARCCORE_BASE_EXPORT std::ostream& operator<<(std::ostream& o,const String&);
  //! Opérateur de lecture d'une String
  friend ARCCORE_BASE_EXPORT std::istream& operator>>(std::istream& o,String&);

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont égales,
   * \retval false sinon.
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const char* a,const String& b);

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont différentes,
   * \retval false si elles sont égales.
   */
  inline friend bool operator!=(const char* a,const String& b)
  {
    return !operator==(a,b);
  }

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont égales,
   * \retval false sinon.
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const String& a,const char* b);

  /*!
   * \brief Compare deux chaînes unicode.
   * \retval true si elles sont différentes,
   * \retval false si elles sont égales.
   */
  inline friend bool operator!=(const String& a,const char* b)
  {
    return !operator==(a,b);
  }

  //! Ajoute deux chaînes.
  friend ARCCORE_BASE_EXPORT String operator+(const char* a,const String& b);

  friend ARCCORE_BASE_EXPORT bool operator<(const String& a,const String& b);

 public:

  //! Retourne la concaténation de cette chaîne avec la chaîne \a str encodée en UTF-8
  String operator+(const char* str) const
  {
    if (!str)
      return (*this);
    return operator+(std::string_view(str));
  }
  //! Retourne la concaténation de cette chaîne avec la chaîne \a str encodée en UTF-8
  String operator+(std::string_view str) const;
  //! Retourne la concaténation de cette chaîne avec la chaîne \a str encodée en UTF-8
  String operator+(const std::string& str) const;
  //! Retourne la concaténation de cette chaîne avec la chaîne \a str.
  String operator+(const String& str) const;
  String operator+(unsigned long v) const;
  String operator+(unsigned int v) const;
  String operator+(double v) const;
  String operator+(long double v) const;
  String operator+(int v) const;
  String operator+(long v) const;
  String operator+(unsigned long long v) const;
  String operator+(long long v) const;
  String operator+(const APReal& v) const;

  static String fromNumber(unsigned long v);
  static String fromNumber(unsigned int v);
  static String fromNumber(double v);
  static String fromNumber(double v,Integer nb_digit_after_point);
  static String fromNumber(long double v);
  static String fromNumber(int v);
  static String fromNumber(long v);
  static String fromNumber(unsigned long long v);
  static String fromNumber(long long v);
  static String fromNumber(const APReal& v);

 public:

  static String format(const String& str);
  static String format(const String& str,const StringFormatterArg& arg1);
  static String format(const String& str,const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2);
  static String format(const String& str,const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3);
  static String format(const String& str,const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4);
  static String format(const String& str,const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4,
                       const StringFormatterArg& arg5);
  static String format(const String& str,const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4,
                       const StringFormatterArg& arg5,
                       const StringFormatterArg& arg6);
  static String format(const String& str,const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4,
                       const StringFormatterArg& arg5,
                       const StringFormatterArg& arg6,
                       const StringFormatterArg& arg7);
  static String format(const String& str,const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4,
                       const StringFormatterArg& arg5,
                       const StringFormatterArg& arg6,
                       const StringFormatterArg& arg7,
                       const StringFormatterArg& arg8);
  static String format(const String& str,const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4,
                       const StringFormatterArg& arg5,
                       const StringFormatterArg& arg6,
                       const StringFormatterArg& arg7,
                       const StringFormatterArg& arg8,
                       const StringFormatterArg& arg9);
  static String concat(const StringFormatterArg& arg1);
  static String concat(const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2);
  static String concat(const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3);
  static String concat(const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4);

  //! Forme standard du pluriel par ajout d'un 's'
  static String plural(const Integer n, const String & str, const bool with_number = true);
  //! Forme particulière du pluriel par variante
  static String plural(const Integer n, const String & str, const String & str2, const bool with_number = true);

  //! Indique si la chaîne contient \a s
  bool contains(const String& s) const;

  //! Indique si la chaîne commence par les caractères de \a s
  bool startsWith(const String& s) const;

  //! Indique si la chaîne se termine par les caractères de \a s
  bool endsWith(const String& s) const;

  //! Sous-chaîne commençant à la position \a pos
  String substring(Int64 pos) const;

  //! Sous-chaîne commençant à la position \a pos et de longueur \a len
  String substring(Int64 pos,Int64 len) const;

  static String join(String delim,ConstArrayView<String> strs);

  //! Découpe la chaîne suivant le caractère \a c
  template<typename StringContainer> void
  split(StringContainer& str_array,char c) const
  {
    const String& str = *this;
    //TODO: passer par String::bytes().
    const char* str_str = str.localstr();
    Int64 offset = 0;
    Int64 len = str.length();
    for( Int64 i=0; i<len; ++i ){
      // GG: remet temporairement l'ancienne sémantique (équivalente à strtok())
      // et supprime la modif IFPEN car cela cause trop d'incompatibilités avec
      // le code existant. A noter que l'implémentation de l'ancienne sémantique
      // a plusieurs bugs:
      //   1. dans le cas où on répète 3 fois ou plus consécutivement le
      // délimiteur. Par exemple 'X:::Y' retourne {'X',':','Y'} au lieu de
      // {'X','Y'}
      //   2. Si on commence par le délimiteur, ce dernier est retourné:
      // Avec ':X:Y', on retourne {':X','Y'} au lieu de {'X','Y'}
      //if (str_str[i]==c){
      if (str_str[i]==c && i!=offset){
        str_array.push_back(std::string_view(str_str+offset,i-offset));
        offset = i+1;
      }
    }
    if (len!=offset)
      str_array.push_back(std::string_view(str_str+offset,len-offset));
  }

 public:
  /*!
   * \brief Affiche les infos internes de la classe.
   *
   * Cette méthode n'est utile que pour débugger Arccore
   */
  void internalDump(std::ostream& ostr) const;

 private:

  mutable StringImpl* m_p = nullptr; //!< Implémentation de la classe
  mutable const char* m_const_ptr = nullptr;
  mutable Int64 m_const_ptr_size = 0; //!< Longueur de la chaîne si constante (-1 sinon)

  void _checkClone() const;
  bool isLess(const String& s) const;
  String& _append(const String& str);
  // A n'appeler que si 'm_const_ptr' est non nul sinon m_const_ptr_size vaut (-1)
  std::string_view _viewFromConstChar() const
  {
    return std::string_view(m_const_ptr,m_const_ptr_size);
  }
  void _removeReference();
  ConstArrayView<UChar> _internalUtf16BE() const;
  void _resetFields()
  {
    m_p = nullptr;
    m_const_ptr = nullptr;
    m_const_ptr_size = 0;
  }
  void _copyFields(const String& str)
  {
    m_p = str.m_p;
    m_const_ptr = str.m_const_ptr;
    m_const_ptr_size = str.m_const_ptr_size;
  }

  /*!
   * \brief Supprime la référence à l'implémentation si elle n'est pas nulle.
   */
  void _removeReferenceIfNeeded()
  {
    if (m_p)
      _removeImplReference();
  }

  /*!
   * \brief Supprime la référence à l'implémentation.
   * \pre m_p != nullptr
   */
  void _removeImplReference();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename U>
class StringFormatterArgToString
{
 public:
  static void toString(const U& v,String& s)
  {
    std::ostringstream ostr;
    ostr << v;
    s = ostr.str();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe utilisée pour formatter une chaîne de caractères.
 */
class ARCCORE_BASE_EXPORT StringFormatterArg
{
 public:
  template<typename U>
  StringFormatterArg(const U& avalue)
  {
    StringFormatterArgToString<U>::toString(avalue,m_str_value);
  }
  StringFormatterArg(Real avalue)
  {
    _formatReal(avalue);
  }
  StringFormatterArg(const String& s)
  : m_str_value(s)
  {
  }
 public:
  const String& value() const { return m_str_value; }
 private:
  String m_str_value;
 private:
  void _formatReal(Real avalue);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Span<const std::byte>
asBytes(const String& v)
{
  return asBytes(v.bytes());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
