// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextWriter2.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Ecrivain de types simples.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/TextWriter2.h"

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/IDataCompressor.h"

#include "arcane/core/ArcaneException.h"

#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TextWriter2::Impl
{
 public:

  String m_filename;
  std::ofstream m_ostream;
  Ref<IDataCompressor> m_data_compressor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextWriter2::
TextWriter2(const String& filename)
: m_p(new Impl())
{
  open(filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextWriter2::
TextWriter2()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextWriter2::
~TextWriter2()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter2::
open(const String& filename)
{
  m_p->m_filename = filename;
  std::ios::openmode mode = std::ios::out | std::ios::binary;
  m_p->m_ostream.open(filename.localstr(), mode);
  if (!m_p->m_ostream)
    ARCANE_THROW(ReaderWriterException, "Can not open file '{0}' for writing", filename);
  m_p->m_ostream.precision(FloatInfo<Real>::maxDigit() + 2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextWriter2::
write(Span<const std::byte> values)
{
  _binaryWrite(values.data(), values.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String TextWriter2::
fileName() const
{
  return m_p->m_filename;
}

void TextWriter2::
setDataCompressor(Ref<IDataCompressor> dc)
{
  m_p->m_data_compressor = dc;
}

Ref<IDataCompressor> TextWriter2::
dataCompressor() const
{
  return m_p->m_data_compressor;
}

Int64 TextWriter2::
fileOffset()
{
  return m_p->m_ostream.tellp();
}

void TextWriter2::
_binaryWrite(const void* bytes, Int64 len)
{
  std::ostream& o = m_p->m_ostream;
  //cout << "** BINARY WRITE len=" << len << " deflater=" << m_data_compressor << '\n';
  IDataCompressor* d = m_p->m_data_compressor.get();
  if (d && len > d->minCompressSize()) {
    UniqueArray<std::byte> compressed_values;
    m_p->m_data_compressor->compress(Span<const std::byte>((const std::byte*)bytes, len), compressed_values);
    Int64 compressed_size = compressed_values.largeSize();
    o.write((const char*)&compressed_size, sizeof(Int64));
    o.write((const char*)compressed_values.data(), compressed_size);
    //cout << "** BINARY WRITE len=" << len << " compressed_len=" << compressed_size << '\n';
  }
  else
    o.write((const char*)bytes, len);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& TextWriter2::
stream()
{
  return m_p->m_ostream;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
