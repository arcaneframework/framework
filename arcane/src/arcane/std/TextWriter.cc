// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/IDataCompressor.h"

#include "arcane/ArcaneException.h"

#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TextWriter::Impl
{
 public:
  String m_filename;
  std::ofstream m_ostream;
  Ref<IDataCompressor> m_data_compressor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextWriter::
TextWriter(const String& filename)
: m_p(new Impl())
{
  open(filename);
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
open(const String& filename)
{
  m_p->m_filename = filename;
  std::ios::openmode mode = std::ios::out | std::ios::binary;
  m_p->m_ostream.open(filename.localstr(),mode);
  if (!m_p->m_ostream)
    ARCANE_THROW(ReaderWriterException,"Can not open file '{0}' for writing", filename);
  m_p->m_ostream.precision(FloatInfo<Real>::maxDigit() + 2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(Span<const Real> values)
{
  _binaryWrite(values.data(), values.size() * sizeof(Real));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(Span<const Int16> values)
{
  _binaryWrite(values.data(), values.size() * sizeof(Int16));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(Span<const Int32> values)
{
  _binaryWrite(values.data(), values.size() * sizeof(Int32));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(Span<const Int64> values)
{
  _binaryWrite(values.data(), values.size() * sizeof(Int64));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(Span<const Byte> values)
{
  _binaryWrite(values.data(), values.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter::
write(Span<const std::byte> values)
{
  _binaryWrite(values.data(), values.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String TextWriter::
fileName() const
{
  return m_p->m_filename;
}

void TextWriter::
setDataCompressor(Ref<IDataCompressor> dc)
{
  m_p->m_data_compressor = dc;
}

Ref<IDataCompressor> TextWriter::
dataCompressor() const
{
  return m_p->m_data_compressor;
}

Int64 TextWriter::
fileOffset()
{
  return m_p->m_ostream.tellp();
}

void TextWriter::
_binaryWrite(const void* bytes,Int64 len)
{
  std::ostream& o = m_p->m_ostream;
  //cout << "** BINARY WRITE len=" << len << " deflater=" << m_data_compressor << '\n';
  IDataCompressor* d = m_p->m_data_compressor.get();
  if (d && len > d->minCompressSize()) {
    UniqueArray<std::byte> compressed_values;
    m_p->m_data_compressor->compress(Span<const std::byte>((const std::byte*)bytes,len), compressed_values);
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

std::ostream& TextWriter::
stream()
{
  return m_p->m_ostream;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
