// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
  std::ifstream m_istream;
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
  std::ios::openmode mode = std::ios::in | std::ios::binary;
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
readIntegers(Span<Integer> values)
{
  read(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
_checkStream(const char* type, Int64 nb_read_value)
{
  std::istream& s = m_p->m_istream;
  if (!s)
    ARCANE_THROW(IOException, "Can not read '{0}' (nb_val={1} bad?={2} "
                 "fail?={3} eof?={4} pos={5}) file='{6}'",
                 type, nb_read_value, s.bad(), s.fail(),
                 s.eof(), s.tellg(), m_p->m_filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<std::byte> values)
{
  Int64 nb_value = values.size();
  _binaryRead(values.data(), nb_value);
  _checkStream("byte[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Byte> values)
{
  read(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Int64> values)
{
  read(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Int16> values)
{
  read(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Int32> values)
{
  read(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Real> values)
{
  read(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
_binaryRead(void* values, Int64 len)
{
  std::istream& s = m_p->m_istream;
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
  m_p->m_istream.seekg(v,std::ios::beg);
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

std::ifstream& TextReader::
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
