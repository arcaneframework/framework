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

#include "arcane/ArcaneException.h"
#include "arcane/IDeflateService.h"
#include "arcane/utils/Ref.h"

#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TextReader::Impl
{
 public:
  Impl(const String& filename,bool is_binary)
  : m_filename(filename), m_is_binary(is_binary) {}
 public:
  String m_filename;
  ifstream m_istream;
  Integer m_current_line = 0;
  bool m_is_binary = false;
  Ref<IDeflateService> m_deflater;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TextReader::
TextReader(const String& filename,bool is_binary)
: m_p(new Impl(filename,is_binary))
{
  ios::openmode mode = ios::in;
  if (m_p->m_is_binary)
    mode |= ios::binary;
  m_p->m_istream.open(filename.localstr(),mode);
  if (!m_p->m_istream)
    ARCANE_THROW(ReaderWriterException, "Can not read file '{0}' for reading", filename);
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
  if (m_p->m_is_binary) {
    read(values);
  }
  else {
    for (Integer& v : values)
      v = _getInteger();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
_checkStream(const char* type, Int64 nb_read_value)
{
  istream& s = m_p->m_istream;
  if (!s)
    ARCANE_THROW(IOException, "Can not read '{0}' (nb_val={1} is_binary={2} bad?={3} "
                              "fail?={4} eof?={5} pos={6}) file='{7}'",
                 type, nb_read_value, m_p->m_is_binary, s.bad(), s.fail(),
                 s.eof(), s.tellg(), m_p->m_filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Byte> values)
{
  Int64 nb_value = values.size();
  if (m_p->m_is_binary) {
    // _removeComments() nécessaire pour compatibilité avec première version.
    // a supprimer par la suite
    _removeComments();
    _binaryRead(values.data(), nb_value);
  }
  else {
    _removeComments();
    m_p->m_istream.read((char*)values.data(), nb_value);
  }
  _checkStream("Byte[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Int64> values)
{
  Int64 nb_value = values.size();
  if (m_p->m_is_binary) {
    _binaryRead(values.data(), nb_value * sizeof(Int64));
  }
  else {
    for (Int64& v : values)
      v = _getInt64();
  }
  _checkStream("Int64[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Int16> values)
{
  Int64 nb_value = values.size();
  if (m_p->m_is_binary) {
    _binaryRead(values.data(), nb_value * sizeof(Int16));
  }
  else {
    for (Int16& v : values)
      v = _getInt16();
  }
  _checkStream("Int16[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Int32> values)
{
  Int64 nb_value = values.size();
  if (m_p->m_is_binary) {
    _binaryRead(values.data(), nb_value * sizeof(Int32));
  }
  else {
    for (Int32& v : values)
      v = _getInt32();
  }
  _checkStream("Int32[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
read(Span<Real> values)
{
  Int64 nb_value = values.size();
  if (m_p->m_is_binary) {
    _binaryRead(values.data(), nb_value * sizeof(Real));
  }
  else {
    for (Real& v : values) {
      v = _getReal();
    }
  }
  _checkStream("Real[]", nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TextReader::
_binaryRead(void* values, Int64 len)
{
  istream& s = m_p->m_istream;
  if (m_p->m_deflater.get() && len > DEFLATE_MIN_SIZE) {
    ByteUniqueArray compressed_values;
    Int64 compressed_size = 0;
    s.read((char*)&compressed_size, sizeof(Int64));
    compressed_values.resize(arcaneCheckArraySize(compressed_size));
    s.read((char*)compressed_values.data(), compressed_size);
    m_p->m_deflater->decompress(compressed_values, ByteArrayView(arcaneCheckArraySize(len), (Byte*)values));
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
setDeflater(Ref<IDeflateService> ds)
{
  m_p->m_deflater = ds;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class KeyValueTextReader::Impl
{
 public:
  Impl(const String& filename,bool is_binary,Int32 version)
  : m_reader(filename,is_binary), m_version(version){}
 public:
  TextReader m_reader;
  Int32 m_version;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KeyValueTextReader::
KeyValueTextReader(const String& filename,bool is_binary,Int32 version)
: m_p(new Impl(filename,is_binary,version))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KeyValueTextReader::
~KeyValueTextReader()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
getExtents(const String& key_name,Int64ArrayView extents)
{
  Integer dimension_array_size = extents.size();
  if (m_p->m_version==1 || m_p->m_version==2){
    //String true_key_name = "Extents:" + key_name;
    if (m_p->m_version==1){
      // Dans la version 1, les dimensions sont des 'Int32'
      IntegerUniqueArray dims;
      if (dimension_array_size>0){
        dims.resize(dimension_array_size);
        //extents.resize(dimension_array_size);
        m_p->m_reader.read(dims);
      }
      for( Integer i=0; i<dimension_array_size; ++i )
        extents[i] = dims[i];
    }
    else{
      if (dimension_array_size>0){
        //extents.resize(dimension_array_size);
        m_p->m_reader.read(extents);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
readIntegers(const String& key,Span<Integer> values)
{
  m_p->m_reader.readIntegers(values);
}

void KeyValueTextReader::
read(const String& key,Span<Int16> values)
{
  m_p->m_reader.read(values);
}

void KeyValueTextReader::
read(const String& key,Span<Int32> values)
{
  m_p->m_reader.read(values);
}

void KeyValueTextReader::
read(const String& key,Span<Int64> values)
{
  m_p->m_reader.read(values);
}

void KeyValueTextReader::
read(const String& key,Span<Real> values)
{
  m_p->m_reader.read(values);
}

void KeyValueTextReader::
read(const String& key,Span<Byte> values)
{
  m_p->m_reader.read(values);
}

String KeyValueTextReader::
fileName() const
{
  return m_p->m_reader.fileName();
}

void KeyValueTextReader::
setFileOffset(Int64 v)
{
  m_p->m_reader.setFileOffset(v);
}

void KeyValueTextReader::
setDeflater(Ref<IDeflateService> ds)
{
  m_p->m_reader.setDeflater(ds);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
