// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextReader.h                                                (C) 2000-2018 */
/*                                                                           */
/* Ecrivain de données.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_TEXTREADER_H
#define ARCANE_STD_TEXTREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"

#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDeflateService;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TextReader
{
 public:

  explicit TextReader(const String& filename,bool is_binary);
  ~TextReader();

 public:

  void readIntegers(Span<Integer> values);

  void read(Span<Int16> values);
  void read(Span<Int32> values);
  void read(Span<Int64> values);
  void read(Span<Real> values);
  void read(Span<Byte> values);

 public:
  const String& fileName() const { return m_filename; }
  void setFileOffset(Int64 v) { m_istream.seekg(v,ios::beg); }
  void setDeflater(IDeflateService* ds) { m_deflater = ds; }
 private:
  String m_filename;
  ifstream m_istream;
  Integer m_current_line;
  bool m_is_binary;
  IDeflateService* m_deflater;
 private:
  void _removeComments();
  Integer _getInteger();
  Int16 _getInt16();
  Int32 _getInt32();
  Int64 _getInt64();
  Real _getReal();
  void _binaryRead(void* bytes,Int64 len);
  void _checkStream(const char* type,Int64 nb_read_value);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
