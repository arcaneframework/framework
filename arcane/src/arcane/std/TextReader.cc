// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextReader.cc                                               (C) 2000-2021 */
/*                                                                           */
/* Lecteur simple.                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/TextReader.h"

#include "arcane/utils/IOException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IDataCompressor.h"

#include "arcane/ArcaneException.h"

#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TextReader::Impl
{
 public:
  Impl(const String& filename)
  : m_filename(filename) {}
 public:
  String m_filename;
  ifstream m_istream;
  Integer m_current_line = 0;
  Int64 m_file_length = 0;
  Ref<IDataCompressor> m_data_compressor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextReader::
TextReader(const String& filename)
: m_p(new Impl(filename))
{
  ios::openmode mode = ios::in | ios::binary;
  m_p->m_istream.open(filename.localstr(),mode);
  if (!m_p->m_istream)
    ARCANE_THROW(ReaderWriterException, "Can not read file '{0}' for reading", filename);
  m_p->m_file_length = platform::getFileLength(filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextReader::
~TextReader()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
_removeComments()
{
  int c = '\0';
  char bufline[4096];
  while ((c = m_p->m_istream.peek()) == '#') {
    ++m_p->m_current_line;
    m_p->m_istream.getline(bufline, 4000, '\n');
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer TextReader::
_getInteger()
{
  _removeComments();
  ++m_p->m_current_line;
  Integer value = 0;
  m_p->m_istream >> value >> std::ws;
  return value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
readIntegers(Span<Integer> values)
{
  read(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
_checkStream(const char* type, Int64 nb_read_value)
{
  istream& s = m_p->m_istream;
  if (!s)
    ARCANE_THROW(IOException, "Can not read '{0}' (nb_val={1} bad?={2} "
                 "fail?={3} eof?={4} pos={5}) file='{6}'",
                 type, nb_read_value, s.bad(), s.fail(),
                 s.eof(), s.tellg(), m_p->m_filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Byte> values)
{
  Int64 nb_value = values.size();
  // _removeComments() nécessaire pour compatibilité avec première version.
  // a supprimer par la suite
  _removeComments();
  _binaryRead(values.data(), nb_value);
  _checkStream("Byte[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Int64> values)
{
  Int64 nb_value = values.size();
  _binaryRead(values.data(), nb_value * sizeof(Int64));
  _checkStream("Int64[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Int16> values)
{
  Int64 nb_value = values.size();
  _binaryRead(values.data(), nb_value * sizeof(Int16));
  _checkStream("Int16[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Int32> values)
{
  Int64 nb_value = values.size();
  _binaryRead(values.data(), nb_value * sizeof(Int32));
  _checkStream("Int32[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Real> values)
{
  Int64 nb_value = values.size();
  _binaryRead(values.data(), nb_value * sizeof(Real));
  _checkStream("Real[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
_binaryRead(void* values, Int64 len)
{
  istream& s = m_p->m_istream;
  IDataCompressor* d = m_p->m_data_compressor.get();
  if (d && len > d->minCompressSize()) {
    UniqueArray<std::byte> compressed_values;
    Int64 compressed_size = 0;
    s.read((char*)&compressed_size, sizeof(Int64));
    compressed_values.resize(compressed_size);
    s.read((char*)compressed_values.data(), compressed_size);
    m_p->m_data_compressor->decompress(compressed_values, Span<std::byte>((std::byte*)values,len));
  }
  else {
    s.read((char*)values, len);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 TextReader::
_getInt32()
{
  _removeComments();
  ++m_p->m_current_line;
  Int32 value = 0;
  m_p->m_istream >> value >> ws;
  return value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int16 TextReader::
_getInt16()
{
  _removeComments();
  ++m_p->m_current_line;
  Int16 value = 0;
  m_p->m_istream >> value >> ws;
  return value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 TextReader::
_getInt64()
{
  _removeComments();
  ++m_p->m_current_line;
  Int64 value = 0;
  m_p->m_istream >> value >> ws;
  return value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TextReader::
_getReal()
{
  _removeComments();
  ++m_p->m_current_line;
  Real value = 0;
  m_p->m_istream >> value >> ws;
  return value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String TextReader::
fileName() const
{
  return m_p->m_filename;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
setFileOffset(Int64 v)
{
  m_p->m_istream.seekg(v,ios::beg);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
setDataCompressor(Ref<IDataCompressor> dc)
{
  m_p->m_data_compressor = dc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IDataCompressor> TextReader::
dataCompressor() const
{
  return m_p->m_data_compressor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ifstream& TextReader::
stream()
{
  return m_p->m_istream;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 TextReader::
fileLength() const
{
  return m_p->m_file_length;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
