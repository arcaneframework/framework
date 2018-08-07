/*---------------------------------------------------------------------------*/
/* BasicTranscoder.cc                                          (C) 2000-2018 */
/*                                                                           */
/* Conversions utf8/utf16/iso-8859-1.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/CoreArray.h"
#include "arccore/base/BasicTranscoder.h"

#include <glib.h>

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
int
_invalidChar(Int32& wc)
{
  std::cout << "WARNING: Invalid sequence in conversion input\n";
  wc = '?';
  return 1;
}

int
_notEnoughChar(Int32& wc)
{
  std::cout << "WARNING: Invalid sequence in conversion input (unexpected eof)\n";
  wc = '?';
  return 1;
}
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
static void
ucs4_to_utf8(Int32 wc,CoreArray<Byte>& utf8)
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
  else if (wc <= 0x7fffffff)
    count = 6;
  else{
    utf8.add('?');
    return;
  }
  switch (count) { /* note: code falls through cases! */
  case 6: r[5] = 0x80 | (wc & 0x3f); wc = wc >> 6; wc |= 0x4000000;
  case 5: r[4] = 0x80 | (wc & 0x3f); wc = wc >> 6; wc |= 0x200000;
  case 4: r[3] = 0x80 | (wc & 0x3f); wc = wc >> 6; wc |= 0x10000;
  case 3: r[2] = 0x80 | (wc & 0x3f); wc = wc >> 6; wc |= 0x800;
  case 2: r[1] = 0x80 | (wc & 0x3f); wc = wc >> 6; wc |= 0xc0;
  case 1: r[0] = wc;
  }
  for( int i=0; i<count; ++i )
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
static Integer
utf8_to_ucs4(ByteConstArrayView uchar,Integer index,Int32& wc)
{
  const Byte* s = uchar.data()+index;
  unsigned char c = s[0];
  Integer n = uchar.size() - index;
  if (c < 0x80) {
    wc = c;
    return 1;
  }

  if (c < 0xc2)
    return _invalidChar(wc);
  
  if (c < 0xe0) {
    if (n < 2)
      return _notEnoughChar(wc);
    if (!((s[1] ^ 0x80) < 0x40))
      return _invalidChar(wc);
    wc = ((Int32) (c & 0x1f) << 6) | (Int32) (s[1] ^ 0x80);
    return 2;
  }

  if (c < 0xf0) {
    if (n < 3)
      return _notEnoughChar(wc);
    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40
          && (c >= 0xe1 || s[1] >= 0xa0)))
      return _invalidChar(wc);
    wc = ((Int32) (c & 0x0f) << 12)
    | ((Int32) (s[1] ^ 0x80) << 6)
    | (Int32) (s[2] ^ 0x80);
    return 3;
  }

  if (c < 0xf8) {
    if (n < 4)
      return _notEnoughChar(wc);
    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40
          && (s[3] ^ 0x80) < 0x40
          && (c >= 0xf1 || s[1] >= 0x90)))
      return _invalidChar(wc);
    wc = ((Int32) (c & 0x07) << 18)
    | ((Int32) (s[1] ^ 0x80) << 12)
    | ((Int32) (s[2] ^ 0x80) << 6)
    | (Int32) (s[3] ^ 0x80);
    return 4;
  }

  if (c < 0xfc) {
    if (n < 5)
      return _notEnoughChar(wc);
    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40
          && (s[3] ^ 0x80) < 0x40 && (s[4] ^ 0x80) < 0x40
          && (c >= 0xf9 || s[1] >= 0x88)))
      return _invalidChar(wc);
    wc = ((Int32) (c & 0x03) << 24)
    | ((Int32) (s[1] ^ 0x80) << 18)
    | ((Int32) (s[2] ^ 0x80) << 12)
    | ((Int32) (s[3] ^ 0x80) << 6)
    | (Int32) (s[4] ^ 0x80);
    return 5;
  }

  if (c < 0xfe) {
    if (n < 6)
      return _notEnoughChar(wc);
    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40
          && (s[3] ^ 0x80) < 0x40 && (s[4] ^ 0x80) < 0x40
          && (s[5] ^ 0x80) < 0x40
          && (c >= 0xfd || s[1] >= 0x84)))
      return _invalidChar(wc);
    wc = ((Int32) (c & 0x01) << 30)
    | ((Int32) (s[1] ^ 0x80) << 24)
    | ((Int32) (s[2] ^ 0x80) << 18)
    | ((Int32) (s[3] ^ 0x80) << 12)
    | ((Int32) (s[4] ^ 0x80) << 6)
    | (Int32) (s[5] ^ 0x80);
    return 6;
  }
  return _invalidChar(wc);
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
static Integer
utf16_to_ucs4(UCharConstArrayView uchar,Integer index,Int32& wc)
{
  wc = uchar[index];
  if (wc>=0xd800 && wc<0xdc00){
    if ((index+1)==uchar.size()){
      std::cout << "WARNING: utf16_to_ucs4(): Invalid sequence in conversion input (unexpected eof)\n";
      wc = 0x1A;
      return 1;
    }
    Int32 wc2 = uchar[index+1];
    if (!(wc2>=0xdc00 && wc2<0xe000)){
      std::cout << "WARNING: utf16_to_ucs4(): Invalid sequence in conversion input\n";
      wc = 0x1A;
      return 1;
    }
    wc = (0x10000 + ((wc-0xd800)<<10) + (wc2 - 0xdc00));
    return 2;
  }
  else if (wc>=0xdc00 && wc<0xe0000){
    std::cout << "WARNING: utf16_to_ucs4(): Invalid sequence in conversion input\n";
    wc = 0x1A;
    return 1;
  }
  return 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un caractère (UCS4) en utf16.
 *
 * Routine récupérée dans libiconv.
 *
 * Un caractère ucs4 est génère 1 ou 2 caractères utf16. Les
 * caractères convertis sont ajoutés au tableau \a uchar
 *
 * \param wc valeur ucs4 du caractère à convertir
 * \param uchar[out] Tableau contenant les caractères utf16 convertis
 */
static void
ucs4_to_utf16(Int32 wc,CoreArray<UChar>& uchar)
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
/*!
 * \brief Convertie iso-8859-1 vers utf16.
 */
void  BasicTranscoder::
transcodeFromISO88591ToUtf16(const std::string& s,CoreArray<UChar>& utf16)
{
  // La conversion ISO-8859-1 vers UTF-16 est simple
  // puisque les valeurs sont identiques. Il suffit
  // donc de copier le tableau
  Integer len = arccoreCheckArraySize(s.length());
  utf16.resize(len+1);
  UChar* ustr = utf16.data();
  const Byte* str = (const Byte*)s.c_str();
  for( int i=0; i<len+1; ++i )
    ustr[i] = str[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conversion UTF-16 vers iso-8859-1.
 *
 * Le tableau \a utf16 a un zéro terminal qu'il ne faut pas
 * mettre dans \a s.
 */
void BasicTranscoder::
transcodeFromUtf16ToISO88591(UCharConstArrayView utf16,std::string& s)
{
  const UChar* ustr = utf16.data();
  Integer len = utf16.size();

  s.clear();

  // Ne devrait pas arriver mais on ne sait jamais
  if (len==0){
    std::cerr << "WARNING: empty 'utf16' array\n";
    s = "";
    return;
  }
  --len;

  for( int i=0; i<len; ++i ){
    Int32 wc = ustr[i];
    if (wc>=0xd800 && wc<0xdc00){
      if ((i+1)==len){
        wc = '?';
      }
      else{
        ++i;
        Int32 wc2 = ustr[i];
        if (!(wc2>=0xdc00 && wc2<0xe000)){
          //cout << "WARNING: utf16_to_ucs4(): Invalid sequence in conversion input\n";
          wc = '?';
        }
        else{
          wc = (0x10000 + ((wc-0xd800)<<10) + (wc2 - 0xdc00));
        }
      }
    }
    else if (wc>=0xdc00 && wc<0xe0000){
      wc = '?';
    }
    s.push_back((Byte)wc);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer BasicTranscoder::
stringLen(const UChar* ustr)
{
  if (!ustr || ustr[0] == 0)
    return 0;
  const UChar* u = ustr + 1;
  while ((*u)!=0)
    ++u;
  return (Integer)(u - ustr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Traduit depuis ISO-8859-1 vers UTF8
void BasicTranscoder::
transcodeFromISO88591ToUtf8(const char* str,Integer len,CoreArray<Byte>& utf8)
{

  if (len<0)
    throw std::exception();
  for( Integer i=0; i<len; ++i ){
    Int32 w = (Byte)str[i];
    if (w<0x80){
      utf8.add((Byte)w);
    }
    else{
      Int32 r1 = 0x80 | (w & 0x3f);
      Int32 r0 = (w>>6) | 0xc0;
      utf8.add((Byte)r0);
      utf8.add((Byte)r1);
    }
  }
  utf8.add(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
transcodeFromUtf8ToISO88591(ByteConstArrayView utf8,std::string& s)
{
  // Caractère utilisé si la conversion échoue.
  const char fallback_char = '?';
  const Byte* ustr = utf8.data();
  Integer len = utf8.size();
  if (len==0){
    std::cerr << "empty 'utf8' array\n";
    s = "";
    return;
  }
  --len;
  //char* new_str = new char[(len+1)*2];
  //Integer index = 0;
  s.clear();
  for( int i=0; i<len; ++i ){
    Int32 w = ustr[i];
    if (w<0x80){
      s.push_back((char)w);
    }
    else if (w<0xc2){
      // caractère 'utf8' incorrect
      s.push_back(fallback_char);
    }
    else if (w<0xe0){
      if ((i+1)==len){
        s.push_back(fallback_char);
        continue;
      }
      char x = (char)( ((Int32)(w & 0x1f) << 6) | ((Int32)(ustr[i+1] ^ 0x80)) );
      s.push_back(x);
      ++i;
    }
    else{
      // caractère 'utf8' ne pouvant pas être convert en 'iso-8859-1'
      s.push_back(fallback_char);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Traduit depuis UTF16 vers UTF8
void BasicTranscoder::
transcodeFromUtf16ToUtf8(UCharConstArrayView utf16,CoreArray<Byte>& utf8)
{
  for( int i=0, is=utf16.size(); i<is; ){
    Int32 wc;
    i += utf16_to_ucs4(utf16,i,wc);
    ucs4_to_utf8(wc,utf8);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
transcodeFromUtf8ToUtf16(ByteConstArrayView utf8,CoreArray<UChar>& utf16)
{
  for( int i=0, is=utf8.size(); i<is; ){
    Int32 wc;
    i += utf8_to_ucs4(utf8,i,wc);
    ucs4_to_utf16(wc,utf16);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
replaceWS(CoreArray<Byte>& out_utf8)
{
  CoreArray<Byte> copy_utf8(out_utf8);
  ByteConstArrayView utf8(copy_utf8.view());
  out_utf8.clear();
  for( int i=0, is=utf8.size(); i<is; ){
    Int32 wc;
    i += utf8_to_ucs4(utf8,i,wc);
    if (g_unichar_isspace(wc))
      out_utf8.add(' ');
    else
      ucs4_to_utf8(wc,out_utf8);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
collapseWS(CoreArray<Byte>& out_utf8)
{
  CoreArray<Byte> copy_utf8(out_utf8);
  ByteConstArrayView utf8(copy_utf8.view());
  out_utf8.clear();
  Integer i = 0;
  Integer n = utf8.size();
  // Si la chaîne est vide, retourne une chaîne vide.
  if (n==1){
    out_utf8.add('\0');
    return;
  }
  bool old_is_space = true;
  bool has_spaces_only = true;
  for( ; i<n ; ){
    if (utf8[i] == 0)
      break;
    Int32 wc;
    i += utf8_to_ucs4(utf8,i,wc);
    if (g_unichar_isspace(wc)){
      if (!old_is_space)
        out_utf8.add(' ');
      old_is_space = true;
    }
    else{
      old_is_space = false;
      ucs4_to_utf8(wc,out_utf8);
      has_spaces_only = false;
    }
  }
  if (old_is_space && (!has_spaces_only)){
    if (out_utf8.size()>0)
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
  ByteConstArrayView utf8(copy_utf8.view());
  out_utf8.clear();
  for( int i=0, is=utf8.size(); i<is; ){
    Int32 wc;
    i += utf8_to_ucs4(utf8,i,wc);
    Int32 upper_wc = g_unichar_toupper(wc);
    ucs4_to_utf8(upper_wc,out_utf8);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
lowerCase(CoreArray<Byte>& out_utf8)
{
  CoreArray<Byte> copy_utf8(out_utf8);
  ByteConstArrayView utf8(copy_utf8.view());
  out_utf8.clear();
  for( int i=0, is=utf8.size(); i<is; ){
    Int32 wc;
    i += utf8_to_ucs4(utf8,i,wc);
    Int32 upper_wc = g_unichar_tolower(wc);
    ucs4_to_utf8(upper_wc,out_utf8);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicTranscoder::
substring(CoreArray<Byte>& out_utf8,ByteConstArrayView utf8,Integer pos,Integer len)
{
  // Copie les \a len caractères unicodes de \a utf8 à partir de la position \a pos
  int current_pos = 0;
  for( int i=0, is=utf8.size(); i<is; ){
    Int32 wc;
    i += utf8_to_ucs4(utf8,i,wc);
    if (current_pos>=pos && current_pos<(pos+len)){
      // Pour être sur de ne pas ajouter le 0 terminal
      if (wc!=0)
        ucs4_to_utf8(wc,out_utf8);
    }
    ++current_pos;
  }
  // Ajoute le 0 terminal
  ucs4_to_utf8(0,out_utf8);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
