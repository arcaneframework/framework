/*---------------------------------------------------------------------------*/
/* String.h                                                    (C) 2000-2018 */
/*                                                                           */
/* Chaîne de caractère unicode.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STRING_H
#define ARCCORE_BASE_STRING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <string>
#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StringFormatterArg;
class StringBuilder;
class StringImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Chaine de caractères unicode.
 *
 * Utilise un compteur de référence avec sémantique indentique à QString.
 * A terme la class 'String' doit être immutable pour éviter des
 * problèmes en multi-thread.
 * Pour construire par morceaux une chaine de caractère, il faut
 * utiliser la classe 'StringBuilder'.
 * \warning Il ne doit pas y avoir de variables globales de cette classe
 */
class ARCCORE_BASE_EXPORT String
{
 public:

  friend ARCCORE_BASE_EXPORT bool operator==(const String& a,const String& b);
  friend ARCCORE_BASE_EXPORT bool operator<(const String& a,const String& b);
  friend class StringBuilder;

 public:

  //! Crée une chaîne nulle
  String() : m_p(nullptr), m_const_ptr(nullptr) {}
  /*!
   * \brief Créé une chaîne à partir de \a str dans l'encodage local.
   *
   * \warning Attention, la chaine est supposée constante sa validité
   * infinie (i.e il s'agit d'une chaîne constante à la compilation.
   * Si la chaîne passée en argument peut être désallouée,
   * il faut utiliser le constructeur avec allocation.
   */   
  String(const char* str) : m_p(nullptr), m_const_ptr(str) {}
  //! Créé une chaîne à partir de \a str dans l'encodage local
  String(char* str);
  //! Créé une chaîne à partir de \a str dans l'encodage local
  String(const char* str,bool do_alloc);
  //! Créé une chaîne à partir de \a str dans l'encodage local
  String(const char* str,Integer len);
  //! Créé une chaîne à partir de \a str dans l'encodage local
  String(const std::string& str);
  //! Créé une chaîne à partir de \a str dans l'encodage Utf16
  String(const UCharConstArrayView& ustr);
  //! Créé une chaîne à partir de \a str dans l'encodage Utf8
  String(const ConstLargeArrayView<Byte>& ustr);
  //! Créé une chaîne à partir de \a str dans l'encodage local
  explicit String(StringImpl* impl);
  //! Créé une chaîne à partir de \a str
  String(const String& str);

  //! Copie \a str dans cette instance.
  const String& operator=(const String& str);
  //! Copie \a str dans cette instance.
  const String& operator=(const char* str);

  ~String(); //!< Libère les ressources.

 public:

  static String fromUtf8(ConstLargeArrayView<Byte> bytes);

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
   */
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
   * L'instance retournée ne contient pas de zéro terminal.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance.
   */
  ConstLargeArrayView<Byte> bytes() const;

  /*!
   * \brief Retourne la conversion de l'instance dans l'encodage local.
   *
   * La conversion n'est pas garanti si certaines valeurs unicode n'existent
   * pas dans l'encodage local.
   *
   * \warning L'instance reste propriétaire de la valeur retournée et cette valeur
   * est invalidée par toute modification de cette instance.
   */
  const char* localstr() const;

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


  //! Transforme tous les caracteres de la chaine en majuscules.
  String upper() const;

  //! Transforme tous les caracteres de la chaine en minuscules.
  String lower() const;

  //! Retourne \a true si la chaîne est nulle.
  bool null() const;

  //! Retourne la longueur de la chaîne.
  Integer len() const;
  
  //! Vrai si la chaîne est vide (nulle ou "")
  bool empty() const;

  //! Calcule une valeur de hashage cette la chaîne de caractère
  Int32 hashCode() const;

  //! Écrit la chaîne au format UTF-8 sur le flot \a o
  void writeBytes(std::ostream& o) const;

 public:

  String operator+(const char* str) const;
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
  String substring(Integer pos) const;

  //! Sous-chaîne commençant à la position \a pos et de longueur \a len
  String substring(Integer pos,Integer len) const;

  static String join(String delim,ConstArrayView<String> strs);

  //! Découpe la chaîne suivant le caractère \a c
  template<typename StringContainer> void
  split(StringContainer& str_array,char c) const
  {
    const String& str = *this;
    const char* str_str = str.localstr();
    Integer offset = 0;
    Integer len = str.len();
    for( Integer i=0; i<len; ++i ){
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
        str_array.push_back(String(str_str+offset,i-offset));
        offset = i+1;
      }
    }
    if (len!=offset)
      str_array.push_back(String(str_str+offset,len-offset));
  }

 public:
  /*!
   * \brief Affiche les infos internes de la classe.
   *
   * Cette méthode n'est utile que pour débugger Arccore
   */
  void internalDump(std::ostream& ostr) const;

 private:

  mutable StringImpl* m_p; //!< Implémentation de la classe
  mutable const char* m_const_ptr;

  void _checkClone() const;
  bool isLess(const String& s) const;
  String& _append(const String& str);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Opérateur d'écriture d'une String
ARCCORE_BASE_EXPORT std::ostream& operator<<(std::ostream& o,const String&);
//! Opérateur de lecture d'une String
ARCCORE_BASE_EXPORT std::istream& operator>>(std::istream& o,String&);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont égales,
 * \retval false sinon.
 * \relate String
 */
extern "C++" ARCCORE_BASE_EXPORT bool operator==(const String& a,const String& b);
/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont différentes,
 * \retval false si elles sont égales.
 * \relate String
 */
inline bool operator!=(const String& a,const String& b)
{
  return !operator==(a,b);
}

/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont égales,
 * \retval false sinon.
 * \relate String
 */
extern "C++" ARCCORE_BASE_EXPORT bool operator==(const char* a,const String& b);

/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont différentes,
 * \retval false si elles sont égales.
 * \relate String
 */
inline bool operator!=(const char* a,const String& b)
{
  return !operator==(a,b);
}

/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont égales,
 * \retval false sinon.
 * \relate String
 */
extern "C++" ARCCORE_BASE_EXPORT bool operator==(const String& a,const char* b);
/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si elles sont différentes,
 * \retval false si elles sont égales.
 * \relate String
 */
inline bool operator!=(const String& a,const char* b)
{
  return !operator==(a,b);
}

//! Ajoute deux chaînes.
extern "C++" ARCCORE_BASE_EXPORT String operator+(const char* a,const String& b);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compare deux chaînes unicode.
 * \retval true si a<b
 * \retval false sinon.
 * \relate String
 */
extern "C++" ARCCORE_BASE_EXPORT bool operator<(const String& a,const String& b);

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

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
