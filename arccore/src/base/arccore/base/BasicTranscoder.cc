// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicTranscoder.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Conversions entre utf8 et utf16.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/CoreArray.h"
#include "arccore/base/BasicTranscoder.h"

#ifdef ARCCORE_HAS_GLIB
#include <glib.h>
#else
#include <cwctype>
#endif

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
using namespace Arcane;

bool _isSpace(Int32 wc)
{
#ifdef ARCCORE_HAS_GLIB
  return g_unichar_isspace(wc);
#else
  return std::iswspace(wc);
#endif
}
Int32 _toUpper(Int32 wc)
{
#ifdef ARCCORE_HAS_GLIB
  return g_unichar_toupper(wc);
#else
  return std::towupper(wc);
#endif
}
Int32 _toLower(Int32 wc)
{
#ifdef ARCCORE_HAS_GLIB
  return g_unichar_tolower(wc);
#else
  return std::towlower(wc);
#endif
}

int _invalidChar(Int32 pos, Int32& wc)
{
  std::cout << "WARNING: Invalid sequence '" << wc << "' in conversion input (position=" << pos << ")\n";
  wc = '?';
  return 1;
}

int _notEnoughChar(Int32& wc)
{
  std::cout << "WARNING: Invalid sequence '" << wc << "' in conversion input (unexpected eof)\n";
  wc = '?';
  return 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un caractère unicode (UCS4) en utf8.
 *
 * Routine récupérée dans libiconv.
 *
 * Un caractère ucs4 génère entre 1 et 6 caractères utf8.
 * Les caractères convertis sont ajoutés au tableau \a utf8.
 *
 * \param wc valeur ucs4 du caractère à convertir
 * \param utf8[out] Tableau contenant les caractères utf8 convertis
 */
void ucs4_to_utf8(Int32 wc, Impl::CoreArray<Byte>& utf8)
{
  Int32 r[6];
  int count;
  if (wc < 0x80)
    count = 1;
  else if (wc < 0x800)
    count = 2;
  else if (wc < 0x10000)
    count = 3;
  else if (wc < 0x200000)
    count = 4;
  else if (wc < 0x4000000)
    count = 5;
  else
    count = 6;
  switch (count) { /* note: code falls through cases! */
  case 6:
    r[5] = 0x80 | (wc & 0x3f);
    wc = wc >> 6;
    wc |= 0x4000000;
    [[fallthrough]];
  case 5:
    r[4] = 0x80 | (wc & 0x3f);
    wc = wc >> 6;
    wc |= 0x200000;
    [[fallthrough]];
  case 4:
    r[3] = 0x80 | (wc & 0x3f);
    wc = wc >> 6;
    wc |= 0x10000;
    [[fallthrough]];
  case 3:
    r[2] = 0x80 | (wc & 0x3f);
    wc = wc >> 6;
    wc |= 0x800;
    [[fallthrough]];
  case 2:
    r[1] = 0x80 | (wc & 0x3f);
    wc = wc >> 6;
    wc |= 0xc0;
    [[fallthrough]];
  case 1:
    r[0] = wc;
  }
  for (int i = 0; i < count; ++i)
    utf8.add((Byte)r[i]);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un caractère utf8 en unicode (UCS4).
 *
 * Routine récupérée dans libiconv.
 *
 * Un caractère ucs4 est créé à partir de 1 à 6 caractères utf8.
 *
 * \param uchar Tableau contenant les caractères utf8 à convertir
 * \param index indice du premier élément du tableau à convertir
 * \param wc [out] valeur ucs4 du caractère.
 * \return le nombre de caractères utf8 lus.
 */
Int64 utf8_to_ucs4(Span<const Byte> uchar, Int64 index, Int32& wc)
{
  const Byte* s = uchar.data() + index;
  unsigned char c = s[0];
  Int64 n = uchar.size() - index;
  if (c < 0x80) {
    wc = c;
    return 1;
  }

  if (c < 0xc2)
    return _invalidChar(1, wc);

  if (c < 0xe0) {
    if (n < 2)
      return _notEnoughChar(wc);
    if (!((s[1] ^ 0x80) < 0x40))
      return _invalidChar(2, wc);
    wc = ((Int32)(c & 0x1f) << 6) | (Int32)(s[1] ^ 0x80);
    return 2;
  }

  if (c < 0xf0) {
    if (n < 3)
      return _notEnoughChar(wc);
    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40 && (c >= 0xe1 || s[1] >= 0xa0)))
      return _invalidChar(4, wc);
    wc = ((Int32)(c & 0x0f) << 12) | ((Int32)(s[1] ^ 0x80) << 6) | (Int32)(s[2] ^ 0x80);
    return 3;
  }

  if (c < 0xf8) {
    if (n < 4)
      return _notEnoughChar(wc);
    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40 && (s[3] ^ 0x80) < 0x40 && (c >= 0xf1 || s[1] >= 0x90)))
      return _invalidChar(5, wc);
    wc = ((Int32)(c & 0x07) << 18) | ((Int32)(s[1] ^ 0x80) << 12) | ((Int32)(s[2] ^ 0x80) << 6) | (Int32)(s[3] ^ 0x80);
    return 4;
  }

  // On ne devrait jamais arriver ici
  // car il n'y a plus (depuis la norme 2003) de caractères UTF-8
  // encodés sur 5 ou 6 octets.

  if (c < 0xfc) {
    if (n < 5)
      return _notEnoughChar(wc);
    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40 && (s[3] ^ 0x80) < 0x40 && (s[4] ^ 0x80) < 0x40 && (c >= 0xf9 || s[1] >= 0x88)))
      return _invalidChar(7, wc);
    wc = ((Int32)(c & 0x03) << 24) | ((Int32)(s[1] ^ 0x80) << 18) | ((Int32)(s[2] ^ 0x80) << 12) | ((Int32)(s[3] ^ 0x80) << 6) | (Int32)(s[4] ^ 0x80);
    return 5;
  }
  if (c < 0xfe) {
    if (n < 6)
      return _notEnoughChar(wc);
    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40 && (s[3] ^ 0x80) < 0x40 && (s[4] ^ 0x80) < 0x40 && (s[5] ^ 0x80) < 0x40 && (c >= 0xfd || s[1] >= 0x84)))
      return _invalidChar(8, wc);
    wc = ((Int32)(c & 0x01) << 30) | ((Int32)(s[1] ^ 0x80) << 24) | ((Int32)(s[2] ^ 0x80) << 18) | ((Int32)(s[3] ^ 0x80) << 12) | ((Int32)(s[4] ^ 0x80) << 6) | (Int32)(s[5] ^ 0x80);
    return 6;
  }
  return _invalidChar(9, wc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un caractère utf16 en unicode (UCS4).
 *
 * Routine récupérée dans libiconv.
 *
 * Un caractère ucs4 est créé à partir de 1 ou 2 caractères utf16.
 *
 * \param uchar Tableau contenant les caractères utf16 à convertir
 * \param index indice du premier élément du tableau à convertir
 * \param wc [out] valeur ucs4 du caractère.
 * \return le nombre de caractères utf16 lus.
 */
Int64 utf16_to_ucs4(Span<const UChar> uchar, Int64 index, Int32& wc)
{
  wc = uchar[index];
  if (wc >= 0xd800 && wc < 0xdc00) {
    if ((index + 1) == uchar.size()) {
      std::cout << "WARNING: utf16_to_ucs4(): Invalid sequence in conversion input (unexpected eof)\n";
      wc = 0x1A;
      return 1;
    }
    Int32 wc2 = uchar[index + 1];
    if (!(wc2 >= 0xdc00 && wc2 < 0xe000)) {
      std::cout << "WARNING: utf16_to_ucs4(): Invalid sequence (1) '" << wc2 << "' in conversion input\n";
      wc = 0x1A;
      return 1;
    }
    wc = (0x10000 + ((wc - 0xd800) << 10) + (wc2 - 0xdc00));
    return 2;
  }
  else if (wc >= 0xdc00 && wc < 0xe0000) {
    std::cout << "WARNING: utf16_to_ucs4(): Invalid sequence (2) '" << wc << "' in conversion input\n";
    wc = 0x1A;
    return 1;
  }
  return 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un caractère (UCS4) en utf16 big-endian.
 *
 * Routine récupérée dans libiconv.
 *
 * Un caractère ucs4 est génère 1 ou 2 caractères utf16. Les
 * caractères convertis sont ajoutés au tableau \a uchar
 *
 * \param wc valeur ucs4 du caractère à convertir
 * \param uchar[out] Tableau contenant les caractères utf16 convertis
 */
void
ucs4_to_utf16(Int32 wc,Impl::CoreArray<UChar>& uchar)
{
  if (wc < 0xd800){
    uchar.add((UChar)wc);
    return;
  }
  if (wc < 0xe000){
    std::cout << "WARNING: ucs4_to_utf16(): Invalid sequence in conversion input\n";
    uchar.add(0x1A);
    return;
  }
  if (wc < 0x10000){
    uchar.add((UChar)wc);
    return;
  }
  if (wc < 0x110000){
	  uchar.add( (UChar) ((wc - 0x10000) / 0x400 + 0xd800) );
	  uchar.add( (UChar) ((wc - 0x10000) % 0x400 + 0xdc00) );
    return;
  }
  std::cerr << "WARNING: ucs4_to_utf16(): Invalid sequence in conversion input\n";
  uchar.add(0x1A);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 BasicTranscoder::
stringLen(const UChar* ustr)
{
  if (!ustr || ustr[0] == 0)
    return 0;
  const UChar* u = ustr + 1;
  while ((*u) != 0)
    ++u;
  return arccoreCheckLargeArraySize((std::size_t)(u - ustr));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Traduit depuis UTF16 vers UTF8
void BasicTranscoder::
transcodeFromUtf16ToUtf8(Span<const UChar> utf16, CoreArray<Byte>& utf8)
{
  for (Int64 i = 0, n = utf16.size(); i < n;) {
    Int32 wc;
    i += utf16_to_ucs4(utf16, i, wc);
    ucs4_to_utf8(wc, utf8);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
transcodeFromUtf8ToUtf16(Span<const Byte> utf8, CoreArray<UChar>& utf16)
{
  for (Int64 i = 0, n = utf8.size(); i < n;) {
    Int32 wc;
    i += utf8_to_ucs4(utf8, i, wc);
    ucs4_to_utf16(wc, utf16);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
replaceWS(CoreArray<Byte>& out_utf8)
{
  Impl::CoreArray<Byte> copy_utf8(out_utf8);
  Span<const Byte> utf8(copy_utf8);
  out_utf8.clear();
  for (Int64 i = 0, n = utf8.size(); i < n;) {
    Int32 wc;
    i += utf8_to_ucs4(utf8, i, wc);
    if (_isSpace(wc))
      out_utf8.add(' ');
    else
      ucs4_to_utf8(wc, out_utf8);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
collapseWS(CoreArray<Byte>& out_utf8)
{
  Impl::CoreArray<Byte> copy_utf8(out_utf8);
  Span<const Byte> utf8(copy_utf8);
  out_utf8.clear();
  Int64 i = 0;
  Int64 n = utf8.size();
  // Si la chaîne est vide, retourne une chaîne vide.
  if (n == 1) {
    out_utf8.add('\0');
    return;
  }
  bool old_is_space = true;
  bool has_spaces_only = true;
  for (; i < n;) {
    if (utf8[i] == 0)
      break;
    Int32 wc;
    i += utf8_to_ucs4(utf8, i, wc);
    if (_isSpace(wc)) {
      if (!old_is_space)
        out_utf8.add(' ');
      old_is_space = true;
    }
    else {
      old_is_space = false;
      ucs4_to_utf8(wc, out_utf8);
      has_spaces_only = false;
    }
  }
  if (old_is_space && (!has_spaces_only)) {
    if (out_utf8.size() > 0)
      out_utf8.back() = 0;
  }
  else {
    if (has_spaces_only)
      out_utf8.add(' ');
    out_utf8.add(0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
upperCase(CoreArray<Byte>& out_utf8)
{
  CoreArray<Byte> copy_utf8(out_utf8);
  Span<const Byte> utf8(copy_utf8.view());
  out_utf8.clear();
  for (Int64 i = 0, n = utf8.size(); i < n;) {
    Int32 wc;
    i += utf8_to_ucs4(utf8, i, wc);
    Int32 upper_wc = _toUpper(wc);
    ucs4_to_utf8(upper_wc, out_utf8);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
lowerCase(CoreArray<Byte>& out_utf8)
{
  CoreArray<Byte> copy_utf8(out_utf8);
  Span<const Byte> utf8(copy_utf8.view());
  out_utf8.clear();
  for (Int64 i = 0, n = utf8.size(); i < n;) {
    Int32 wc;
    i += utf8_to_ucs4(utf8, i, wc);
    Int32 upper_wc = _toLower(wc);
    ucs4_to_utf8(upper_wc, out_utf8);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
substring(CoreArray<Byte>& out_utf8, Span<const Byte> utf8, Int64 pos, Int64 len)
{
  // Copie les \a len caractères unicodes de \a utf8 à partir de la position \a pos
  Int64 current_pos = 0;
  for (Int64 i = 0, n = utf8.size(); i < n;) {
    Int32 wc;
    i += utf8_to_ucs4(utf8, i, wc);
    if (current_pos >= pos && current_pos < (pos + len)) {
      // Pour être sur de ne pas ajouter le 0 terminal
      if (wc != 0)
        ucs4_to_utf8(wc, out_utf8);
    }
    ++current_pos;
  }
  // Ajoute le 0 terminal
  ucs4_to_utf8(0, out_utf8);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
