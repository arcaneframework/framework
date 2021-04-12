// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextWriter.cc                                               (C) 2000-2021 */
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
#include "arcane/utils/Ref.h"

#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TextWriter::Impl
{
 public:
  String m_filename;
  ofstream m_ostream;
  bool m_is_binary = false;
  Ref<IDeflateService> m_deflater;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextWriter::
TextWriter(const String& filename,bool is_binary)
: m_p(new Impl())
{
  open(filename,is_binary);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextWriter::
TextWriter()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextWriter::
~TextWriter()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
open(const String& filename,bool is_binary)
{
  m_p->m_is_binary = is_binary;
  m_p->m_filename = filename;
  ios::openmode mode = ios::out;
  if (m_p->m_is_binary)
    mode |= ios::binary;
  m_p->m_ostream.open(filename.localstr(),mode);
  if (!m_p->m_ostream)
    ARCANE_THROW(ReaderWriterException,"Can not open file '{0}' for writing", filename);
  m_p->m_ostream.precision(FloatInfo<Real>::maxDigit() + 2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(const String& comment,Span<const Real> values)
{
  if (m_p->m_is_binary) {
    _binaryWrite(values.data(), values.size() * sizeof(Real));
  }
  else {
    _writeComments(comment);
    for( Real v : values ){
      m_p->m_ostream << v << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(const String &comment,Span<const Int16> values)
{
  if (m_p->m_is_binary) {
    _binaryWrite(values.data(), values.size() * sizeof(Int16));
  }
  else {
    _writeComments(comment);
    for( Int16 v : values ){
      m_p->m_ostream << v << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(const String& comment,Span<const Int32> values)
{
  if (m_p->m_is_binary) {
    _binaryWrite(values.data(), values.size() * sizeof(Int32));
  }
  else {
    _writeComments(comment);
    for( Int32 v : values ){
      m_p->m_ostream << v << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(const String& comment,Span<const Int64> values)
{
  if (m_p->m_is_binary) {
    _binaryWrite(values.data(), values.size() * sizeof(Int64));
  }
  else {
    _writeComments(comment);
    for( Int64 v : values ){
      m_p->m_ostream << v << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(const String& comment,Span<const Byte> values)
{
  if (m_p->m_is_binary) {
    _binaryWrite(values.data(), values.size());
  }
  else {
    _writeComments(comment);
    m_p->m_ostream.write((const char *) values.data(), values.size());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& TextWriter::
fileName() const
{
  return m_p->m_filename;
}

void TextWriter::
setDeflater(Ref<IDeflateService> ds)
{
  m_p->m_deflater = ds;
}

Int64 TextWriter::
fileOffset()
{
  return m_p->m_ostream.tellp();
}

void TextWriter::
_writeComments(const String& comment)
{
  m_p->m_ostream << "# " << comment << '\n';
}

void TextWriter::
_binaryWrite(const void* bytes,Int64 len)
{
  ostream& o = m_p->m_ostream;
  //cout << "** BINARY WRITE len=" << len << " deflater=" << m_deflater << '\n';
  if (m_p->m_deflater.get() && len > DEFLATE_MIN_SIZE) {
    ByteUniqueArray compressed_values;
    Int32 small_len = arcaneCheckArraySize(len);
    m_p->m_deflater->compress(ByteConstArrayView(small_len,(const Byte*)bytes), compressed_values);
    Int64 compressed_size = compressed_values.largeSize();
    o.write((const char *) &compressed_size, sizeof(Int64));
    o.write((const char *) compressed_values.data(), compressed_size);
    //cout << "** BINARY WRITE len=" << len << " compressed_len=" << compressed_size << '\n';
  }
  else
    o.write((const char *) bytes, len);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool TextWriter::
isBinary() const
{
  return m_p->m_is_binary;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
