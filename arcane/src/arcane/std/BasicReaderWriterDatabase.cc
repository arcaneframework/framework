// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicReaderWriterDatabase.cc                                (C) 2000-2021 */
/*                                                                           */
/* Base de donnée pour le service 'BasicReaderWriter'.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/BasicReaderWriterDatabase.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/JSONReader.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/IDataCompressor.h"

#include "arcane/ArcaneException.h"

#include "arcane/std/TextReader.h"
#include "arcane/std/TextWriter.h"

#include <fstream>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{
//! Structure pour gérer l'épilogue.
class BasicReaderWriterDatabaseEpilogFormat
{
 public:
  // La taille de cette structure ne doit pas être modifiée sous peine
  // de rendre le format incompatible. Pour supporter des évolutions, on fixe
  // une taille de 128 octets, soit 16 'Int64'
  static constexpr Int64 STRUCT_SIZE = 128;
 public:
  BasicReaderWriterDatabaseEpilogFormat()
  {
    checkStructureSize();
    m_version = 1;
    m_padding0 = 0;
    m_padding1 = 0;
    m_padding2 = 0;
    m_padding3 = 0;
    m_json_data_info_file_offset = -1;
    m_json_data_info_size = 0;
    for( int i=0; i<10; ++i )
      m_remaining_padding[i] = i;
  }
 public:
  void setJSONDataInfoOffsetAndSize(Int64 file_offset,Int64 data_size)
  {
    m_json_data_info_file_offset = file_offset;
    m_json_data_info_size = data_size;
  }
  Int32 version() const { return m_version; }
  Int64 jsonDataInfoFileOffset() const { return m_json_data_info_file_offset; }
  Int64 jsonDataInfoSize() const { return m_json_data_info_size; }
  Span<std::byte> bytes()
  {
    return { reinterpret_cast<std::byte*>(this), STRUCT_SIZE };
  }
 private:
  // Version de l'epilogue. A ne pas confondre avec la version du fichier
  // qui est dans l'en-tête.
  Int32 m_version;
  Int32 m_padding0;
  Int64 m_padding1;
  Int64 m_padding2;
  Int64 m_padding3;
  Int64 m_json_data_info_file_offset;
  Int64 m_json_data_info_size;
  Int64 m_remaining_padding[10];
 public:
  static void checkStructureSize()
  {
    Int64 s = sizeof(BasicReaderWriterDatabaseEpilogFormat);
    if (s!=STRUCT_SIZE)
      ARCANE_FATAL("Invalid size for epilog format size={0} expected={1}",s,STRUCT_SIZE);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Structure pour gérer l'en-tête
class BasicReaderWriterDatabaseHeaderFormat
{
 public:
  // La taille de cette structure ne doit pas être modifiée sous peine
  // de rendre le format incompatible. Pour supporter des évolutions, on fixe
  // une taille de 128 octets, soit 16 'Int64'
  static constexpr Int64 STRUCT_SIZE = 128;
 public:
  BasicReaderWriterDatabaseHeaderFormat()
  {
    checkStructureSize();
    // 4 premiers octets pour indiquer qu'il s'agit d'un fichier de protections
    // arcane (ACR pour Arcane Checkpoint Restart)
    m_header_begin[0] = 'A';
    m_header_begin[1] = 'C';
    m_header_begin[2] = 'R';
    m_header_begin[3] = (Byte)39;
    // 4 octets suivant pour indiquer l'indianness (pas encore utilisé)
    m_endian_int = 0x01020304;
    // Version du fichier (à modifier par l'appelant via setVersion())
    m_version = 0;
    m_padding0 = 0;
    m_padding1 = 0;
    m_padding2 = 0;
    for( int i=0; i<12; ++i )
      m_remaining_padding[i] = i;
  }
 public:
  Span<std::byte> bytes()
  {
    return { reinterpret_cast<std::byte*>(this), STRUCT_SIZE };
  }
  void setVersion(Int32 version) { m_version = version; }
  Int32 version() const { return m_version; }
  void checkHeader()
  {
    // Vérifie que le header est correct.
    if (m_header_begin[0]!='A' || m_header_begin[1]!='C' || m_header_begin[2]!='R' || m_header_begin[3]!=(Byte)39)
      ARCANE_FATAL("Bad header");
    // TODO: tester indianess
  }
 private:
  Byte m_header_begin[4];
  Int32 m_endian_int;
  Int32 m_version;
  Int32 m_padding0;
  Int64 m_padding1;
  Int64 m_padding2;
  Int64 m_remaining_padding[12];
 public:
  static void checkStructureSize()
  {
    Int64 s = sizeof(BasicReaderWriterDatabaseHeaderFormat);
    if (s!=STRUCT_SIZE)
      ARCANE_FATAL("Invalid size for header format size={0} expected={1}",s,STRUCT_SIZE);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicReaderWriterDatabaseCommon
{
 public:
  struct ExtentsInfo
  {
    static constexpr int MAX_SIZE = 8;
    void fill(SmallSpan<const Int64> v)
    {
      nb = v.size();
      if (nb>MAX_SIZE){
        m_large_extents = v;
        return;
      }
      for( Int32 i=0; i<nb; ++i )
        sizes[i] = v[i];
    }
    Int64ConstArrayView view() const
    {
      if (nb<=MAX_SIZE)
        return Int64ConstArrayView(nb,sizes);
      return m_large_extents.view();
    }
    Int32 size() const { return nb; }
   private:
    Int32 nb = 0;
   public:
    Int64 sizes[MAX_SIZE];
    UniqueArray<Int64> m_large_extents;
  };

  struct DataInfo
  {
    Int64 m_file_offset = 0;
    ExtentsInfo m_extents;
  };

 public:

  DataInfo& findData(const String& key_name)
  {
    auto x = m_data_infos.find(key_name);
    if (x==m_data_infos.end())
      ARCANE_FATAL("Can not find key '{0}' in database",key_name);
    return x->second;
  }

 public:

  std::map<String,DataInfo> m_data_infos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class KeyValueTextWriter::Impl
: public BasicReaderWriterDatabaseCommon
{
 public:
  Impl(const String& filename,Int32 version)
  : m_writer(filename), m_version(version)
  {}
 public:
  TextWriter m_writer;
  Int32 m_version;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KeyValueTextWriter::
KeyValueTextWriter(const String& filename,Int32 version)
: m_p(new Impl(filename,version))
{
  if (m_p->m_version>=3)
    _writeHeader();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KeyValueTextWriter::
~KeyValueTextWriter()
{
  if (m_p->m_version>=3)
    arcaneCallFunctionAndTerminateIfThrow([&]() { _writeEpilog(); });
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief En-tête du format de fichier.
 *
 * Toute modification dans cette en-tête doit être reportée dans la
 * classe KeyValueTextReader.
 */
void KeyValueTextWriter::
_writeHeader()
{
  BasicReaderWriterDatabaseHeaderFormat header;

  // Actuellement si on passe dans cette partie de code,
  // la version utilisée est 3 ou plus.
  header.setVersion(m_p->m_version);
  Span<std::byte> bytes(header.bytes());
  m_p->m_writer.stream().write((const char*)bytes.data(),bytes.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
_writeEpilog()
{
  // Ecrit les méta-données au format JSON.
  JSONWriter jsw;
  {
    JSONWriter::Object main_object(jsw);
    jsw.writeKey("Data");
    jsw.beginArray();
    for( auto& x : m_p->m_data_infos ){
      JSONWriter::Object o(jsw);
      jsw.write("Name",x.first);
      jsw.write("FileOffset",x.second.m_file_offset);
      jsw.write("Extents",x.second.m_extents.view());
    }
    jsw.endArray();
  }

  std::ostream& stream = m_p->m_writer.stream();

  // Conserve la position dans le fichier des méta-données
  // ainsi que leur taille.
  Int64 file_offset = fileOffset();
  StringView buf = jsw.getBuffer();
  Int64 meta_data_size = buf.size();

  stream.write((const char*)buf.bytes().data(),meta_data_size);

  {
    BasicReaderWriterDatabaseEpilogFormat epilog;
    epilog.setJSONDataInfoOffsetAndSize(file_offset,meta_data_size);
    // Doit toujours être la dernière écriture du fichier
    auto epilog_bytes = epilog.bytes();
    stream.write((const char*)epilog_bytes.data(),epilog_bytes.size());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
setExtents(const String& key_name,SmallSpan<const Int64> extents)
{
  if (m_p->m_version>=3){
    _addKey(key_name,extents);
  }
  else{
    // Versions 1 et 2.
    // On sauve directement dans le fichier à la position courante
    // les valeurs des dimentions. Le nombre de valeur est donné par la taille
    // de \a extents
    Integer dimension_array_size = extents.size();
    if (dimension_array_size!=0){
      String true_key_name = "Extents:" + key_name;
      String comment = String::format("Writing Dim1Size for '{0}'",key_name);
      if (m_p->m_version==1){
        // Sauve les dimensions comme un tableau de Int32
        UniqueArray<Integer> dims(dimension_array_size);
        for( Integer i=0; i<dimension_array_size; ++i )
          dims[i] = CheckedConvert::toInteger(extents[i]);
        m_p->m_writer.write(asBytes(dims.span()));
      }
      else
        // Sauve les dimensions comme un tableau de Int64
        m_p->m_writer.write(asBytes(Span<const Int64>(extents)));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
write(const String& key,Span<const std::byte> values)
{
  _writeKey(key);
  m_p->m_writer.write(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String KeyValueTextWriter::
fileName() const
{
  return m_p->m_writer.fileName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
setDataCompressor(Ref<IDataCompressor> ds)
{
  m_p->m_writer.setDataCompressor(ds);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IDataCompressor> KeyValueTextWriter::
dataCompressor() const
{
  return m_p->m_writer.dataCompressor();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 KeyValueTextWriter::
fileOffset()
{
  return m_p->m_writer.fileOffset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
_addKey(const String& key,SmallSpan<const Int64> extents)
{
  Impl::DataInfo d { -1, Impl::ExtentsInfo() };
  d.m_extents.fill(extents);
  m_p->m_data_infos.insert(std::make_pair(key,d));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
_writeKey(const String& key)
{
  if (m_p->m_version>=3){
    auto x = m_p->m_data_infos.find(key);
    if (x==m_p->m_data_infos.end())
      ARCANE_FATAL("Key '{0}' is not in map. You should call setExtents() before",key);
    x->second.m_file_offset = fileOffset();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class KeyValueTextReader::Impl
: public BasicReaderWriterDatabaseCommon
{
 public:
  Impl(const String& filename,Int32 version)
  : m_reader(filename), m_version(version){}
 public:
  TextReader m_reader;
  Int32 m_version;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KeyValueTextReader::
KeyValueTextReader(const String& filename,Int32 version)
: m_p(new Impl(filename,version))
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
  std::ifstream& s = m_p->m_reader.stream();
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
  BasicReaderWriterDatabaseHeaderFormat header;
  Span<std::byte> bytes(header.bytes());
  _readDirect(0,bytes);
  header.checkHeader();
  Int32 version = header.version();
  if (version!=m_p->m_version)
    ARCANE_FATAL("Invalid version for ACR file version={0} expected={1}",version,m_p->m_version);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
_readJSON()
{
  // Les informations sur la position dans le fichier et la longueur du
  // texte JSON sont sauvgegardées dans l'épilogue.
  Int64 file_length = m_p->m_reader.fileLength();
  // Vérifie la longueur du fichier par précaution
  Int64 struct_size = BasicReaderWriterDatabaseEpilogFormat::STRUCT_SIZE;
  if (file_length<struct_size)
    ARCANE_FATAL("File is too short length={0} minimum={1}",file_length,struct_size);

  BasicReaderWriterDatabaseEpilogFormat epilog;

  // Lit l'épilogue et verifie que la version est supportée.
  {
    _readDirect(file_length-struct_size,epilog.bytes());
    const int expected_version = 1;
    if (epilog.version()!=expected_version)
      ARCANE_FATAL("Bad version for epilog version={0} expected={1}",
                   epilog.version(),expected_version);
  }

  UniqueArray<std::byte> json_bytes;

  // Lit les données JSON
  {
    Int64 file_offset = epilog.jsonDataInfoFileOffset();
    Int64 meta_data_size = epilog.jsonDataInfoSize();
    //std::cout << "FILE_INFO: offset=" << file_offset << " meta_data_size=" << meta_data_size << "\n";
    json_bytes.resize(meta_data_size);
    _readDirect(file_offset,json_bytes);
  }

  // Remplit les infos de la base de données à partir du JSON
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
      x.m_extents.fill(extents.view());
      m_p->m_data_infos.insert(std::make_pair(name,x));
    }
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
getExtents(const String& key_name,SmallSpan<Int64> extents)
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
        m_p->m_reader.read(asWritableBytes(dims.span()));
      }
      for( Integer i=0; i<dimension_array_size; ++i )
        extents[i] = dims[i];
    }
    else{
      if (dimension_array_size>0){
        m_p->m_reader.read(asWritableBytes(Span<Int64>(extents)));
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
read(const String& key,Span<std::byte> values)
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
setDataCompressor(Ref<IDataCompressor> ds)
{
  m_p->m_reader.setDataCompressor(ds);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IDataCompressor> KeyValueTextReader::
dataCompressor() const
{
  return m_p->m_reader.dataCompressor();
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
