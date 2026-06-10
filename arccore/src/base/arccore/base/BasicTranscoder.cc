// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicTranscoder.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Conversions between utf8 and utf16.                                       */
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
 * \brief Convert a Unicode character (UCS4) to utf8.
 *
 * Routine retrieved from libiconv.
 *
 * One ucs4 character generates between 1 and 6 utf8 characters.
 * The converted characters are added to the utf8 array.
 *
 * \param wc ucs4 value of the character to convert
 * \param utf8[out] Array containing the converted utf8 characters
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
 * \brief Convert a utf8 character to unicode (UCS4).
 *
 * Routine retrieved from libiconv.
 *
 * One ucs4 character is created from 1 to 6 utf8 characters.
 *
 * \param uchar Array containing the utf8 characters to convert
 * \param index index of the first element of the array to convert
 * \param wc [out] ucs4 value of the character.
 * \return the number of utf8 characters read.
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

  // We should never reach here
  // because there are no longer (since the 2003 standard)
  // UTF-8 characters encoded in 5 or 6 bytes.

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
 * \brief Convert a utf16 character to unicode (UCS4).
 *
 * Routine retrieved from libiconv.
 *
 * One ucs4 character is created from 1 or 2 utf16 characters.
 *
 * \param uchar Array containing the utf16 characters to convert
 * \param index index of the first element of the array to convert
 * \param wc [out] ucs4 value of the character.
 * \return the number of utf16 characters read.
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
 * \brief Convert a character (UCS4) to utf16 big-endian.
 *
 * Routine retrieved from libiconv.
 *
 * One ucs4 character generates 1 or 2 utf16 characters. The
 * converted characters are added to the uchar array
 *
 * \param wc ucs4 value of the character to convert
 * \param uchar[out] Array containing the converted utf16 characters
 */
void ucs4_to_utf16(Int32 wc, Impl::CoreArray<UChar>& uchar)
{
  if (wc < 0xd800) {
    uchar.add((UChar)wc);
    return;
  }
  if (wc < 0xe000) {
    std::cout << "WARNING: ucs4_to_utf16(): Invalid sequence in conversion input\n";
    uchar.add(0x1A);
    return;
  }
  if (wc < 0x10000) {
    uchar.add((UChar)wc);
    return;
  }
  if (wc < 0x110000) {
    uchar.add((UChar)((wc - 0x10000) / 0x400 + 0xd800));
    uchar.add((UChar)((wc - 0x10000) % 0x400 + 0xdc00));
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

//! Translates from UTF16 to UTF8
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
  // If the string is empty, return an empty string.
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
  // Copy the 'len' Unicode characters from the 'pos' position of the utf8 array
  Int64 current_pos = 0;
  for (Int64 i = 0, n = utf8.size(); i < n;) {
    Int32 wc;
    i += utf8_to_ucs4(utf8, i, wc);
    if (current_pos >= pos && current_pos < (pos + len)) {
      // To ensure the terminal null character is not added
      if (wc != 0)
        ucs4_to_utf8(wc, out_utf8);
    }
    ++current_pos;
  }
  // Add the terminal null character
  ucs4_to_utf8(0, out_utf8);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
