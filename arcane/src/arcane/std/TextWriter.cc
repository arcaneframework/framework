// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextWriter.cc                                               (C) 2000-2018 */
/*                                                                           */
/* Ecrivain de types simples.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/TextWriter.h"

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/Array.h"

#include "arcane/IDeflateService.h"
#include "arcane/ArcaneException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextWriter::
TextWriter(const String& filename,bool is_binary)
: m_is_binary(is_binary)
, m_deflater(nullptr)
{
  open(filename,is_binary);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextWriter::
TextWriter()
: m_is_binary(false)
, m_deflater(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextWriter::
~TextWriter()
{
  delete m_deflater;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
open(const String& filename,bool is_binary)
{
  m_is_binary = is_binary;
  ios::openmode mode = ios::out;
  if (m_is_binary)
    mode |= ios::binary;
  m_ostream.open(filename.localstr(),mode);
  if (!m_ostream)
    ARCANE_THROW(ReaderWriterException,"Can not open file '{0}' for writing", filename);
  m_ostream.precision(FloatInfo<Real>::maxDigit() + 2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(const String& comment,Span<const Real> values)
{
  if (m_is_binary) {
    _binaryWrite(values.data(), values.size() * sizeof(Real));
  }
  else {
    _writeComments(comment);
    for( Real v : values ){
      m_ostream << v << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(const String &comment,Span<const Int16> values)
{
  if (m_is_binary) {
    _binaryWrite(values.data(), values.size() * sizeof(Int16));
  }
  else {
    _writeComments(comment);
    for( Int16 v : values ){
      m_ostream << v << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(const String& comment,Span<const Int32> values)
{
  if (m_is_binary) {
    _binaryWrite(values.data(), values.size() * sizeof(Int32));
  }
  else {
    _writeComments(comment);
    for( Int32 v : values ){
      m_ostream << v << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(const String& comment,Span<const Int64> values)
{
  if (m_is_binary) {
    _binaryWrite(values.data(), values.size() * sizeof(Int64));
  }
  else {
    _writeComments(comment);
    for( Int64 v : values ){
      m_ostream << v << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(const String& comment,Span<const Byte> values)
{
  if (m_is_binary) {
    _binaryWrite(values.data(), values.size());
  }
  else {
    _writeComments(comment);
    m_ostream.write((const char *) values.data(), values.size());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ostream& TextWriter::
stream()
{
  return m_ostream;
}

const String& TextWriter::
fileName() const
{
  return m_filename;
}

void TextWriter::
setDeflater(IDeflateService *ds)
{
  m_deflater = ds;
}

Int64 TextWriter::
fileOffset()
{
  return m_ostream.tellp();
}

void TextWriter::
_writeComments(const String& comment)
{
  m_ostream << "# " << comment << '\n';
}

void TextWriter::
_binaryWrite(const void* bytes,Int64 len)
{
  //cout << "** BINARY WRITE len=" << len << " deflater=" << m_deflater << '\n';
  if (m_deflater && len > DEFLATE_MIN_SIZE) {
    ByteUniqueArray compressed_values;
    Int32 small_len = arcaneCheckArraySize(len);
    m_deflater->compress(ByteConstArrayView(small_len,(const Byte*)bytes), compressed_values);
    Int64 compressed_size = compressed_values.largeSize();
    m_ostream.write((const char *) &compressed_size, sizeof(Int64));
    m_ostream.write((const char *) compressed_values.data(), compressed_size);
    //cout << "** BINARY WRITE len=" << len << " compressed_len=" << compressed_size << '\n';
  }
  else
    m_ostream.write((const char *) bytes, len);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
