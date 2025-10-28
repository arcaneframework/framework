// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.cc                                                  (C) 2000-2025 */
/*                                                                           */
/* Fonctions pour convertir une chaîne de caractère en un type donné.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/internal/ConvertInternal.h"
#include "arccore/base/Convert.h"

#include <charconv>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Convert::Impl
{
namespace
{
  /*!
   * \brief Retourne \a s converti en \a 'const char*'.
   *
   * \warning Si la valeur retournée est utilisée pour une fonction C,
   * il faut être sur que \a s a un '\0' terminal.
   */
  const char* _stringViewData(StringView s)
  {
    return reinterpret_cast<const char*>(s.bytes().data());
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne une vue en supprimant les caratères blancs du début.
 *
 * Un caractère blanc est un caractère pour lequel std::isspace() est vrai.
 * \a pos indique la position dans \a s à partir de laquelle
 * on cherche les blancs.
 */
StringView _removeLeadingSpaces(StringView s, Int64 pos)
{
  Span<const Byte> bytes = s.bytes();
  Int64 nb_byte = bytes.size();
  // Supprime les espaces potentiels
  for (; pos < nb_byte; ++pos) {
    int charv = static_cast<unsigned char>(bytes[pos]);
    // Visual Studio 2017 or less
#if defined(_MSC_VER) && _MSC_VER <= 1916
    if (std::isspace(charv, std::locale()) != 0)
      break;
#else
    if (!std::isspace(charv) != 0)
      break;
#endif
  }
  return s.subView(pos, nb_byte);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringViewInputStream::
StringViewInputStream(StringView v)
: m_view(v)
, m_stream(this)
{
  auto b = v.bytes();
  char* begin_ptr = const_cast<char*>(reinterpret_cast<const char*>(b.data()));
  char* end_ptr = begin_ptr + b.size();
  setg(begin_ptr, begin_ptr, end_ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ConvertPolicy::m_verbosity = 0;
bool ConvertPolicy::m_use_from_chars = true;
bool ConvertPolicy::m_use_same_convert_for_all_real = false;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti \a s en un double.
 *
 * Utilise std::from_chars() si \a global_use_from_chars est vrai.
 * Sinon, utilise strtod().
 */
Int64 StringViewToDoubleConverter::
_getDoubleValue(double& v, StringView s)
{
  if (ConvertPolicy::isUseFromChars()) {
    s = _removeLeadingSpaces(s);
    Int64 p = _getDoubleValueWithFromChars(v, s);
    return p;
  }

  const char* ptr = _stringViewData(s);
#ifdef WIN32
  if (s == "infinity" || s == "inf") {
    v = std::numeric_limits<double>::infinity();
    return s.size();
  }
#endif
  char* ptr2 = nullptr;
  if (ptr)
    v = ::strtod(ptr, &ptr2);
  return (ptr2 - ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti une chaîne de caractères en un double.
 *
 * Converti \a s en un double et range la valeur dans \a v.
 * Il ne doit pas y avoir de caractères blancs au début de \a s.
 *
 * Le comportement de cette méthode est identique à std::strtod()
 * avec le locale 'C' si on est en C++20. Sinon il est identique
 * à std::strtod() avec le locale actuel (ce qui peut changer par exemple
 * le séparateur décimal). La documentation de référence est
 * ici: https://en.cppreference.com/w/cpp/utility/from_chars.
 *
 * \retval (-1) si la conversion a échouée.
 * \retval la position dans \s du dernier caractère lu plus 1.
 */
Int64 StringViewToDoubleConverter::
_getDoubleValueWithFromChars(double& v, StringView s)
{
  // NOTE: si on veut le même comportement que 'strtod',
  // on suppose que l'appelant a enlevé les blancs en début de s.
  auto bytes = s.bytes();
  Int64 size = bytes.size();
  if (size == 0)
    // NOTE: Avec la version historique d'Arcane (avant la 3.15) il
    // n'y avait pas d'erreur retournée lorsqu'on converti une chaîne vide.
    // A priori cela n'était jamais utilisé donc cela ne pose pas de
    // problème de corriger ce bug.
    return (-1);
  const char* orig_data = reinterpret_cast<const char*>(bytes.data());
  const char* last_ptr = nullptr;
  std::chars_format fmt = std::chars_format::general;
  const char* data = orig_data;
  bool do_negatif = false;
  const bool is_verbose = ConvertPolicy::verbosity() > 0;

  // std::from_chars() ne supporte pas les '+' en début alors
  // que 'strto*' le supporte.
  if (bytes[0] == '+') {
    ++data;
    --size;
    bytes = bytes.subspan(1, size);
  }
  // std::from_chars() peut lire les valeurs au format hexadécimal
  // mais il ne doit pas contenir le '0x' ou '0X' du début, contrairement
  // à std::strtod(). On détecte ce cas et on commence la conversion
  // après le '0x' ou '0X'.
  // Détecte '-0x' ou '-0X'
  if (size >= 3 && (bytes[0] == '-') && (bytes[1] == '0') && (bytes[2] == 'x' || bytes[2] == 'X')) {
    fmt = std::chars_format::hex;
    data += 3;
    do_negatif = true;
  }
  // Détecte '0x' ou '0X'
  else if (size >= 2 && (bytes[0] == '0') && (bytes[1] == 'x' || bytes[1] == 'X')) {
    fmt = std::chars_format::hex;
    data += 2;
  }
  // Cas général
  {
    auto [ptr, ec] = std::from_chars(data, data + size, v, fmt);
    last_ptr = ptr;
    if (is_verbose)
      std::cout << "FromChars:TRY GET_DOUBLE data=" << data << " v=" << v << " is_ok=" << (ec == std::errc()) << "\n";
    if (ec != std::errc())
      return (-1);
  }
  // Prend en compte le signe '-' si demandé
  if (do_negatif)
    v = -v;
  if (is_verbose) {
    char* ptr2 = nullptr;
    double v2 = ::strtod(orig_data, &ptr2);
    std::cout << "FromChars: COMPARE GET_DOUBLE via strtod v2=" << v2 << " pos=" << (ptr2 - orig_data) << "\n";
  }
  return (last_ptr - orig_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
