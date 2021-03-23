// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextWriter.h                                                (C) 2000-2018 */
/*                                                                           */
/* Ecrivain de données.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_TEXTWRITER_H
#define ARCANE_STD_TEXTWRITER_H
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

class TextWriter
{
 public:

  TextWriter(const String& filename,bool is_binary);
  TextWriter();
  ~TextWriter();

 public:

  void open(const String& filename,bool is_binary);
  void write(const String& comment,Span<const Real> values);
  void write(const String& comment,Span<const Int16> values);
  void write(const String& comment,Span<const Int32> values);
  void write(const String& comment,Span<const Int64> values);
  void write(const String& comment,Span<const Byte> values);
  ostream& stream();
 public:
  const String& fileName() const;
  bool isBinary() const { return m_is_binary; }
  void setDeflater(IDeflateService* ds);
  Int64 fileOffset();
 private:
  String m_filename;
  ofstream m_ostream;
  bool m_is_binary;
  IDeflateService* m_deflater;
 private:
  void _writeComments(const String& comment);
  void _binaryWrite(const void* bytes,Int64 len);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
