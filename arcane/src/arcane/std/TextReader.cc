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
#include "arcane/utils/JSONReader.h"

#include "arcane/ArcaneException.h"
#include "arcane/IDeflateService.h"

#include <fstream>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
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
  Int64 m_file_length = 0;
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

class KeyValueTextReader::Impl
{
 public:
  static constexpr int MAX_SIZE = 8;
  // TODO: a fusionner avec KeyValueTextWriter::Impl::ExtentsInfo
  struct ExtentsInfo
  {
    void fill(Int64ConstArrayView v)
    {
      if (v.size()>MAX_SIZE)
        ARCANE_FATAL("Number of extents ({0}) is greater than max allowed ({1})",v.size(),MAX_SIZE);
      nb = v.size();
      for( Int32 i=0; i<nb; ++i )
        sizes[i] = v[i];
    }
    Int64ConstArrayView view() const { return Int64ConstArrayView(nb,sizes); }
    Int32 size() const { return nb; }
   private:
    Int32 nb = 0;
   public:
    Int64 sizes[MAX_SIZE];
  };
  struct DataInfo
  {
    Int64 m_file_offset = 0;
    ExtentsInfo m_extents;
  };
 public:
  Impl(const String& filename,bool is_binary,Int32 version)
  : m_reader(filename,is_binary), m_version(version){}
 public:
  TextReader m_reader;
  std::map<String,DataInfo> m_data_infos;
  Int32 m_version;
 public:
  DataInfo& findData(const String& key_name)
  {
    auto x = m_data_infos.find(key_name);
    if (x==m_data_infos.end())
      ARCANE_FATAL("Can not find key '{0}' in database",key_name);
    return x->second;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KeyValueTextReader::
KeyValueTextReader(const String& filename,bool is_binary,Int32 version)
: m_p(new Impl(filename,is_binary,version))
{
  if (m_p->m_version>=3){
    _readHeader();
    _readJSON();
  }
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
_readDirect(Int64 offset,Span<std::byte> bytes)
{
  m_p->m_reader.setFileOffset(offset);
  ifstream& s = m_p->m_reader.stream();
  s.read((char*)bytes.data(),bytes.size());
  if (s.fail())
    ARCANE_FATAL("Can not read file part offset={0} length={1}",offset,bytes.length());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief En-tête du format de fichier.
 *
 * Toute modification dans cette en-tête doit être reportée dans la
 * classe KeyValueTextWriter.
 */
void KeyValueTextReader::
_readHeader()
{
  constexpr Integer header_size = 12;
  Byte header_buf[header_size];
  Span<Byte> header(header_buf,header_size);
  header.fill(0);
  _readDirect(0,asWritableBytes(header));
  if (header[0]!='A' || header[1]!='C' || header[2]!='R' || header[3]!=(Byte)39)
    ARCANE_FATAL("Bad header");
  if (header[5]!=0 || header[6]!=0 || header[7]!=0)
    ARCANE_FATAL("Bad header version");
  Integer version = header[4];
  if (version!=m_p->m_version)
    ARCANE_FATAL("Invalid version for ACR file version={0} expected={1}",version,m_p->m_version);
  // TODO: tester indianess
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
_readJSON()
{
  // Les informations sur la position dans le fichier et la longueur du
  // texte JSON sont sauvgegardées à la fin du fichier sous la forme
  // de deux Int64.
  Int64 file_length = m_p->m_reader.fileLength();
  // Ne devrait pas arriver vu qu'on a réussi à lire l'en-tête
  if (file_length<16)
    ARCANE_FATAL("File is too short");

  UniqueArray<std::byte> json_bytes;

  {
    Int64 read_info[2];
    // Doit toujours être la dernière écriture du fichier
    Span<Int64> ri(read_info,2);
    _readDirect(file_length-16,asWritableBytes(ri));
    Int64 file_offset = read_info[0];
    Int64 meta_data_size = read_info[1];
    //std::cout << "FILE_INFO: offset=" << file_offset << " meta_data_size=" << meta_data_size << "\n";
    json_bytes.resize(meta_data_size);
    _readDirect(file_offset,json_bytes);
  }

  {
    JSONDocument json_doc;
    json_doc.parse(json_bytes);
    JSONValue root = json_doc.root();
    JSONValue data = root.child("Data");
    UniqueArray<Int64> extents;
    extents.reserve(12);
    for( JSONValue v : data.valueAsArray() ){
      String name = v.child("Name").valueAsString();
      Int64 file_offset = v.child("FileOffset").valueAsInt64();
      //std::cout << "Name=" << name << "\n";
      //std::cout << "FileOffset=" << file_offset << "\n";
      JSONValueList extents_info = v.child("Extents").valueAsArray();
      extents.clear();
      for( JSONValue v2 : extents_info ){
        extents.add(v2.valueAsInt64());
      }
      Impl::DataInfo x;
      x.m_file_offset = file_offset;
      x.m_extents.fill(extents);
      m_p->m_data_infos.insert(std::make_pair(name,x));
    }
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
getExtents(const String& key_name,Int64ArrayView extents)
{
  Integer dimension_array_size = extents.size();
  if (m_p->m_version>=3){
    Impl::DataInfo& data = m_p->findData(key_name);
    Impl::ExtentsInfo& exi = data.m_extents;
    if (extents.size()!=exi.size())
      ARCANE_FATAL("Bad size for extents size={0} expected={1}",extents.size(),exi.size());
    extents.copy(exi.view());
  }
  else {
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
  _setFileOffset(key);
  m_p->m_reader.readIntegers(values);
}

void KeyValueTextReader::
read(const String& key,Span<Int16> values)
{
  _setFileOffset(key);
  m_p->m_reader.read(values);
}

void KeyValueTextReader::
read(const String& key,Span<Int32> values)
{
  _setFileOffset(key);
  m_p->m_reader.read(values);
}

void KeyValueTextReader::
read(const String& key,Span<Int64> values)
{
  _setFileOffset(key);
  m_p->m_reader.read(values);
}

void KeyValueTextReader::
read(const String& key,Span<Real> values)
{
  _setFileOffset(key);
  m_p->m_reader.read(values);
}

void KeyValueTextReader::
read(const String& key,Span<Byte> values)
{
  _setFileOffset(key);
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

void KeyValueTextReader::
_setFileOffset(const String& key_name)
{
  // Avec les versions antérieures à la version 3, c'est l'appelant qui
  // positionne l'offset car il est le seul à le connaitre.
  if (m_p->m_version>=3){
    Impl::DataInfo& data = m_p->findData(key_name);
    setFileOffset(data.m_file_offset);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
