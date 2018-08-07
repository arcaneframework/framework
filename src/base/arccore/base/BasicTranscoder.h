/*---------------------------------------------------------------------------*/
/* BasicTranscoder.h                                           (C) 2000-2018 */
/*                                                                           */
/* Conversions utf8/utf16/iso-8859-1.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_BASICTRANSCODER_H
#define ARCCORE_BASE_BASICTRANSCODER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> class CoreArray;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
class ARCCORE_BASE_EXPORT BasicTranscoder
{
 public:
  BasicTranscoder() = delete;
 public:

  static void transcodeFromISO88591ToUtf16(const std::string& s,CoreArray<UChar>& utf16);
  static void transcodeFromUtf16ToISO88591(UCharConstArrayView utf16,std::string& s);

  static void transcodeFromISO88591ToUtf8(const char* str,Integer len,CoreArray<Byte>& utf8);
  static void transcodeFromUtf8ToISO88591(ByteConstArrayView utf8,std::string& s);

  static void transcodeFromUtf16ToUtf8(UCharConstArrayView utf16,CoreArray<Byte>& utf8);
  static void transcodeFromUtf8ToUtf16(ByteConstArrayView utf8,CoreArray<UChar>& utf16);

  static Integer stringLen(const UChar* ustr);

  static void replaceWS(CoreArray<Byte>& ustr);
  static void collapseWS(CoreArray<Byte>& ustr);

  static void upperCase(CoreArray<Byte>& utf8);
  static void lowerCase(CoreArray<Byte>& utf8);

  static void substring(CoreArray<Byte>& utf8,ByteConstArrayView rhs,Integer pos,Integer len);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

