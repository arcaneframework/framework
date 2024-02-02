// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextReader2.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Lecteur simple.                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/TextReader2.h"

#include "arcane/utils/IOException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IDataCompressor.h"
#include "arcane/utils/FixedArray.h"

#include "arcane/core/ArcaneException.h"

#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TextReader2::Impl
{
 public:

  Impl(const String& filename)
  : m_filename(filename)
  {}

 public:

  String m_filename;
  std::ifstream m_istream;
  Integer m_current_line = 0;
  Int64 m_file_length = 0;
  Ref<IDataCompressor> m_data_compressor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextReader2::
TextReader2(const String& filename)
: m_p(new Impl(filename))
{
  std::ios::openmode mode = std::ios::in | std::ios::binary;
  m_p->m_istream.open(filename.localstr(), mode);
  if (!m_p->m_istream)
    ARCANE_THROW(ReaderWriterException, "Can not read file '{0}' for reading", filename);
  m_p->m_file_length = platform::getFileLength(filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextReader2::
~TextReader2()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader2::
readIntegers(Span<Integer> values)
{
  read(asWritableBytes(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader2::
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

void TextReader2::
read(Span<std::byte> values)
{
  Int64 nb_value = values.size();
  _binaryRead(values);
  _checkStream("byte[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader2::
_binaryRead(Span<std::byte> values)
{
  std::istream& s = m_p->m_istream;
  IDataCompressor* d = m_p->m_data_compressor.get();
  if (d && values.size() > d->minCompressSize()) {
    UniqueArray<std::byte> compressed_values;
    FixedArray<Int64, 1> compressed_size;
    binaryRead(s, asWritableBytes(compressed_size.span()));
    compressed_values.resize(compressed_size[0]);
    binaryRead(s, asWritableBytes(compressed_values));
    m_p->m_data_compressor->decompress(compressed_values, values);
  }
  else {
    binaryRead(s, values);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String TextReader2::
fileName() const
{
  return m_p->m_filename;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader2::
setFileOffset(Int64 v)
{
  m_p->m_istream.seekg(v, std::ios::beg);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader2::
setDataCompressor(Ref<IDataCompressor> dc)
{
  m_p->m_data_compressor = dc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IDataCompressor> TextReader2::
dataCompressor() const
{
  return m_p->m_data_compressor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::istream& TextReader2::
stream()
{
  return m_p->m_istream;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 TextReader2::
fileLength() const
{
  return m_p->m_file_length;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
