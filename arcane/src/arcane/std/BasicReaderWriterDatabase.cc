// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicReaderWriterDatabase.cc                                (C) 2000-2025 */
/*                                                                           */
/* Base de donnée pour le service 'BasicReaderWriter'.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicReaderWriterDatabase.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/JSONReader.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/IDataCompressor.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/ArcaneException.h"

#include "arcane/std/internal/TextReader2.h"
#include "arcane/std/internal/TextWriter2.h"
#include "arcane/std/internal/IHashDatabase.h"

#include <fstream>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Classe pour calculer le hash d'un tableau
class Hasher
{
 public:

  void computeHash(Span<const std::byte> bytes, Array<Byte>& hash_result)
  {
    ARCANE_CHECK_POINTER(m_hash_algorithm);
    {
      Real t1 = platform::getRealTime();
      m_nb_processed_bytes += bytes.size();
      m_hash_algorithm->computeHash64(bytes, hash_result);
      Real t2 = platform::getRealTime();
      m_hash_time += (t2 - t1);
    }
  }

  void setHashAlgorithm(IHashAlgorithm* algo)
  {
    m_hash_algorithm = algo;
  }

  void printStats(ITraceMng* tm)
  {
    if (m_nb_processed_bytes == 0)
      return;
    Real nb_byte_per_second = static_cast<Real>(m_nb_processed_bytes) / (m_hash_time + 1.0e-9);
    // Pour avoir en Mega-byte par seconde
    nb_byte_per_second /= 1.0e6;
    tm->info() << "Hasher:nb_processed=" << m_nb_processed_bytes
               << " hash_time=" << m_hash_time
               << " rate=" << static_cast<Int64>(nb_byte_per_second) << " MB/s"
               << "\n";
  }

 private:

  IHashAlgorithm* m_hash_algorithm = nullptr;
  Int64 m_nb_processed_bytes = 0;
  Real m_hash_time = 0.0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
    for (int i = 0; i < 10; ++i)
      m_remaining_padding[i] = i;
  }

 public:

  void setJSONDataInfoOffsetAndSize(Int64 file_offset, Int64 data_size)
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
  Int32 m_version = 1;
  Int32 m_padding0 = 0;
  Int64 m_padding1 = 0;
  Int64 m_padding2 = 0;
  Int64 m_padding3 = 0;
  Int64 m_json_data_info_file_offset = -1;
  Int64 m_json_data_info_size = 0;
  Int64 m_remaining_padding[10];

 public:

  static void checkStructureSize()
  {
    Int64 s = sizeof(BasicReaderWriterDatabaseEpilogFormat);
    if (s != STRUCT_SIZE)
      ARCANE_FATAL("Invalid size for epilog format size={0} expected={1}", s, STRUCT_SIZE);
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
    for (int i = 0; i < 12; ++i)
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
    if (m_header_begin[0] != 'A' || m_header_begin[1] != 'C' || m_header_begin[2] != 'R' || m_header_begin[3] != (Byte)39)
      ARCANE_FATAL("Bad header");
    // TODO: tester indianess
  }

 private:

  // 4 premiers octets pour indiquer qu'il s'agit d'un fichier de protections
  // arcane (ACR pour Arcane Checkpoint Restart)
  std::array<Byte, 4> m_header_begin = { 'A', 'C', 'R', 39 };
  // 4 octets suivant pour indiquer l'indianness (pas encore utilisé)
  Int32 m_endian_int = 0x01020304;
  // Version du fichier (à modifier par l'appelant via setVersion())
  Int32 m_version = 0;
  Int32 m_padding0 = 0;
  Int64 m_padding1 = 0;
  Int64 m_padding2 = 0;
  Int64 m_remaining_padding[12];

 public:

  static void checkStructureSize()
  {
    Int64 s = sizeof(BasicReaderWriterDatabaseHeaderFormat);
    if (s != STRUCT_SIZE)
      ARCANE_FATAL("Invalid size for header format size={0} expected={1}", s, STRUCT_SIZE);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicReaderWriterDatabaseCommon
: public TraceAccessor
{
 public:

  struct ExtentsInfo
  {
    static constexpr int MAX_SIZE = 8;
    void fill(SmallSpan<const Int64> v)
    {
      nb = v.size();
      if (nb > MAX_SIZE) {
        m_large_extents = v;
        return;
      }
      for (Int32 i = 0; i < nb; ++i)
        sizes[i] = v[i];
    }
    Int64ConstArrayView view() const
    {
      if (nb <= MAX_SIZE)
        return Int64ConstArrayView(nb, sizes);
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

  BasicReaderWriterDatabaseCommon(ITraceMng* tm, Int32 version)
  : TraceAccessor(tm)
  {
    if (version >= 3) {
      String hash_directory = platform::getEnvironmentVariable("ARCANE_HASHDATABASE_DIRECTORY");
      if (!hash_directory.null()) {
        info() << "Using Hash database at location '" << hash_directory << "'";
        m_hash_database = createFileHashDatabase(tm, hash_directory);
      }
      else {
        String redis_machine = platform::getEnvironmentVariable("ARCANE_HASHDATABASE_REDIS");
        if (!redis_machine.null()){
          info() << "Using Redis database at location '" << redis_machine << "'";
          m_hash_database = createRedisHashDatabase(tm, redis_machine, 6379);
        }
      }
    }
  }

 public:

  DataInfo& findData(const String& key_name)
  {
    auto x = m_data_infos.find(key_name);
    if (x == m_data_infos.end())
      ARCANE_FATAL("Can not find key '{0}' in database", key_name);
    return x->second;
  }

 public:

  std::map<String, DataInfo> m_data_infos;
  Ref<IDataCompressor> m_data_compressor;
  Ref<IHashAlgorithm> m_hash_algorithm;
  Ref<IHashDatabase> m_hash_database;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class KeyValueTextWriter::Impl
: public BasicReaderWriterDatabaseCommon
{
 public:

  Impl(ITraceMng* tm, const String& filename, Int32 version)
  : BasicReaderWriterDatabaseCommon(tm, version)
  , m_writer(filename)
  , m_version(version)
  {
    if (m_version >= 3)
      _writeHeader();
  }

  ~Impl()
  {
    if (m_version >= 3)
      arcaneCallFunctionAndTerminateIfThrow([&]() { _writeEpilog(); });
    m_hasher.printStats(traceMng());
  }

 public:

  Int64 fileOffset() { return m_writer.fileOffset(); }
  void setExtents(const String& key_name, SmallSpan<const Int64> extents);
  void write(const String& key, Span<const std::byte> values);

 private:

  void _addKey(const String& key, SmallSpan<const Int64> extents);
  void _writeKey(const String& key);
  void _writeHeader();
  void _writeEpilog();

 public:

  TextWriter2 m_writer;
  Int32 m_version;
  Hasher m_hasher;

 private:

  void _write2(const String& key, Span<const std::byte> values);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KeyValueTextWriter::
KeyValueTextWriter(ITraceMng* tm, const String& filename, Int32 version)
: TraceAccessor(tm)
, m_p(new Impl(tm, filename, version))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KeyValueTextWriter::
~KeyValueTextWriter()
{
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
void KeyValueTextWriter::Impl::
_writeHeader()
{
  BasicReaderWriterDatabaseHeaderFormat header;

  // Actuellement si on passe dans cette partie de code,
  // la version utilisée est 3 ou plus.
  header.setVersion(m_version);
  binaryWrite(m_writer.stream(), header.bytes());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::Impl::
_writeEpilog()
{
  // Ecrit les méta-données au format JSON.
  JSONWriter jsw;
  {
    JSONWriter::Object main_object(jsw);
    jsw.writeKey("Data");
    jsw.beginArray();
    for (auto& x : m_data_infos) {
      JSONWriter::Object o(jsw);
      jsw.write("Name", x.first);
      jsw.write("FileOffset", x.second.m_file_offset);
      jsw.write("Extents", x.second.m_extents.view());
    }
    jsw.endArray();
  }

  std::ostream& stream = m_writer.stream();

  // Conserve la position dans le fichier des méta-données
  // ainsi que leur taille.
  Int64 file_offset = m_writer.fileOffset();
  StringView buf = jsw.getBuffer();
  Int64 meta_data_size = buf.size();

  binaryWrite(stream, asBytes(buf.bytes()));

  {
    BasicReaderWriterDatabaseEpilogFormat epilog;
    epilog.setJSONDataInfoOffsetAndSize(file_offset, meta_data_size);
    // Doit toujours être la dernière écriture du fichier
    binaryWrite(stream, epilog.bytes());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::Impl::
setExtents(const String& key_name, SmallSpan<const Int64> extents)
{
  if (m_version >= 3) {
    _addKey(key_name, extents);
  }
  else {
    // Versions 1 et 2.
    // On sauve directement dans le fichier à la position courante
    // les valeurs des dimentions. Le nombre de valeur est donné par la taille
    // de \a extents
    Integer dimension_array_size = extents.size();
    if (dimension_array_size != 0) {
      String true_key_name = "Extents:" + key_name;
      String comment = String::format("Writing Dim1Size for '{0}'", key_name);
      if (m_version == 1) {
        // Sauve les dimensions comme un tableau de Int32
        UniqueArray<Integer> dims(dimension_array_size);
        for (Integer i = 0; i < dimension_array_size; ++i)
          dims[i] = CheckedConvert::toInteger(extents[i]);
        m_writer.write(asBytes(dims));
      }
      else
        // Sauve les dimensions comme un tableau de Int64
        m_writer.write(asBytes(extents));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::Impl::
write(const String& key, Span<const std::byte> values)
{
  _writeKey(key);

  IDataCompressor* d = m_data_compressor.get();
  Int64 len = values.size();
  if (d && len > d->minCompressSize()) {
    UniqueArray<std::byte> compressed_values;
    m_data_compressor->compress(values, compressed_values);
    Int64 compressed_size = compressed_values.largeSize();
    m_writer.write(asBytes(Span<const Int64>(&compressed_size, 1)));
    _write2(key, compressed_values);
  }
  else
    _write2(key, values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::Impl::
_write2(const String& key, Span<const std::byte> values)
{
  if (m_hash_database.get()) {
    IHashAlgorithm* hash_algo = m_hash_algorithm.get();
    if (!hash_algo)
      ARCANE_FATAL("Can not use hash database without hash algorithm");

    SmallArray<Byte, 1024> hash_result;
    m_hasher.computeHash(values, hash_result);
    String hash_value = Convert::toHexaString(hash_result);

    HashDatabaseWriteResult result;
    HashDatabaseWriteArgs args(values, hash_value);
    args.setKey(key);

    m_hash_database->writeValues(args, result);
    info(5) << "WRITE_KW_HASH key=" << key << " hash=" << hash_value << " len=" << values.size();
    m_writer.write(asBytes(hash_result));
  }
  else
    m_writer.write(values);
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
  m_p->m_data_compressor = ds;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IDataCompressor> KeyValueTextWriter::
dataCompressor() const
{
  return m_p->m_data_compressor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
setHashAlgorithm(Ref<IHashAlgorithm> v)
{
  m_p->m_hash_algorithm = v;
  m_p->m_hasher.setHashAlgorithm(v.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IHashAlgorithm> KeyValueTextWriter::
hashAlgorithm() const
{
  return m_p->m_hash_algorithm;
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

void KeyValueTextWriter::Impl::
_addKey(const String& key, SmallSpan<const Int64> extents)
{
  Impl::DataInfo d{ -1, Impl::ExtentsInfo() };
  d.m_extents.fill(extents);
  m_data_infos.insert(std::make_pair(key, d));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::Impl::
_writeKey(const String& key)
{
  if (m_version >= 3) {
    auto x = m_data_infos.find(key);
    if (x == m_data_infos.end())
      ARCANE_FATAL("Key '{0}' is not in map. You should call setExtents() before", key);
    x->second.m_file_offset = fileOffset();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
setExtents(const String& key_name, SmallSpan<const Int64> extents)
{
  m_p->setExtents(key_name, extents);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
write(const String& key, Span<const std::byte> values)
{
  m_p->write(key, values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class KeyValueTextReader::Impl
: public BasicReaderWriterDatabaseCommon
{
 public:

  Impl(ITraceMng* tm, const String& filename, Int32 version)
  : BasicReaderWriterDatabaseCommon(tm, version)
  , m_reader(filename)
  , m_version(version)
  {
    if (m_version >= 3) {
      _readHeader();
      _readJSON();
    }
  }

 public:

  void readIntegers(const String& key, Span<Integer> values);
  void read(const String& key, Span<std::byte> values);

 public:

  void _readHeader();
  void _readJSON();
  void _readDirect(Int64 offset, Span<std::byte> bytes);
  void _setFileOffset(const String& key_name);
  void _read2(const String& key_name, Span<std::byte> values);

 public:

  TextReader2 m_reader;
  Int32 m_version;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KeyValueTextReader::
KeyValueTextReader(ITraceMng* tm, const String& filename, Int32 version)
: TraceAccessor(tm)
, m_p(new Impl(tm, filename, version))
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

void KeyValueTextReader::Impl::
_readDirect(Int64 offset, Span<std::byte> bytes)
{
  m_reader.setFileOffset(offset);
  std::istream& s = m_reader.stream();
  binaryRead(s, bytes);
  if (s.fail())
    ARCANE_FATAL("Can not read file part offset={0} length={1}", offset, bytes.length());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief En-tête du format de fichier.
 *
 * Toute modification dans cette en-tête doit être reportée dans la
 * classe KeyValueTextWriter.
 */
void KeyValueTextReader::Impl::
_readHeader()
{
  BasicReaderWriterDatabaseHeaderFormat header;
  Span<std::byte> bytes(header.bytes());
  _readDirect(0, bytes);
  header.checkHeader();
  Int32 version = header.version();
  if (version != m_version)
    ARCANE_FATAL("Invalid version for ACR file version={0} expected={1}", version, m_version);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::Impl::
_readJSON()
{
  // Les informations sur la position dans le fichier et la longueur du
  // texte JSON sont sauvgegardées dans l'épilogue.
  Int64 file_length = m_reader.fileLength();
  // Vérifie la longueur du fichier par précaution
  Int64 struct_size = BasicReaderWriterDatabaseEpilogFormat::STRUCT_SIZE;
  if (file_length < struct_size)
    ARCANE_FATAL("File is too short length={0} minimum={1}", file_length, struct_size);

  BasicReaderWriterDatabaseEpilogFormat epilog;

  // Lit l'épilogue et verifie que la version est supportée.
  {
    _readDirect(file_length - struct_size, epilog.bytes());
    const int expected_version = 1;
    if (epilog.version() != expected_version)
      ARCANE_FATAL("Bad version for epilog version={0} expected={1}",
                   epilog.version(), expected_version);
  }

  UniqueArray<std::byte> json_bytes;

  // Lit les données JSON
  {
    Int64 file_offset = epilog.jsonDataInfoFileOffset();
    Int64 meta_data_size = epilog.jsonDataInfoSize();
    //std::cout << "FILE_INFO: offset=" << file_offset << " meta_data_size=" << meta_data_size << "\n";
    json_bytes.resize(meta_data_size);
    _readDirect(file_offset, json_bytes);
  }

  // Remplit les infos de la base de données à partir du JSON
  {
    JSONDocument json_doc;
    json_doc.parse(json_bytes);
    JSONValue root = json_doc.root();
    JSONValue data = root.child("Data");
    UniqueArray<Int64> extents;
    extents.reserve(12);
    for (JSONValue v : data.valueAsArray()) {
      String name = v.child("Name").value();
      Int64 file_offset = v.child("FileOffset").valueAsInt64();
      //std::cout << "Name=" << name << "\n";
      //std::cout << "FileOffset=" << file_offset << "\n";
      JSONValueList extents_info = v.child("Extents").valueAsArray();
      extents.clear();
      for (JSONValue v2 : extents_info) {
        extents.add(v2.valueAsInt64());
      }
      Impl::DataInfo x;
      x.m_file_offset = file_offset;
      x.m_extents.fill(extents.view());
      m_data_infos.insert(std::make_pair(name, x));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
getExtents(const String& key_name, SmallSpan<Int64> extents)
{
  Integer dimension_array_size = extents.size();
  if (m_p->m_version >= 3) {
    Impl::DataInfo& data = m_p->findData(key_name);
    Impl::ExtentsInfo& exi = data.m_extents;
    if (extents.size() != exi.size())
      ARCANE_FATAL("Bad size for extents size={0} expected={1}", extents.size(), exi.size());
    extents.copy(exi.view());
  }
  else {
    if (m_p->m_version == 1) {
      // Dans la version 1, les dimensions sont des 'Int32'
      IntegerUniqueArray dims;
      if (dimension_array_size > 0) {
        dims.resize(dimension_array_size);
        m_p->m_reader.read(asWritableBytes(dims));
      }
      for (Integer i = 0; i < dimension_array_size; ++i)
        extents[i] = dims[i];
    }
    else {
      if (dimension_array_size > 0) {
        m_p->m_reader.read(asWritableBytes(extents));
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::Impl::
readIntegers(const String& key, Span<Integer> values)
{
  _setFileOffset(key);
  m_reader.readIntegers(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::Impl::
read(const String& key, Span<std::byte> values)
{
  _setFileOffset(key);

  IDataCompressor* d = m_data_compressor.get();
  Int64 len = values.size();
  if (d && len > d->minCompressSize()) {
    UniqueArray<std::byte> compressed_values;
    Int64 compressed_size = 0;
    m_reader.read(asWritableBytes(Span<Int64>(&compressed_size, 1)));
    compressed_values.resize(compressed_size);
    _read2(key, compressed_values);
    m_data_compressor->decompress(compressed_values, values);
  }
  else {
    _read2(key, values);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::Impl::
_read2(const String& key, Span<std::byte> values)
{
  if (m_hash_database.get()) {
    IHashAlgorithm* hash_algo = m_hash_algorithm.get();
    if (!hash_algo)
      ARCANE_FATAL("Can not use hash database without hash algorithm");
    Int32 hash_size = hash_algo->hashSize();
    SmallArray<Byte, 1024> hash_as_bytes;
    hash_as_bytes.resize(hash_size);
    m_reader.read(asWritableBytes(hash_as_bytes));
    String hash_value = Convert::toHexaString(hash_as_bytes);
    info(5) << "READ_KW_HASH key=" << key << " hash=" << hash_value << " expected_len=" << values.size();
    HashDatabaseReadArgs args(hash_value, values);
    m_hash_database->readValues(args);

    // Vérifie le hash
    if (arcaneIsCheck()) {
      Hasher hasher;
      hasher.setHashAlgorithm(hash_algo);
      SmallArray<Byte, 1024> hash_result;
      hasher.computeHash(values, hash_result);
      String check_hash_value = Convert::toHexaString(hash_result);
      if (check_hash_value!=hash_value)
        ARCANE_FATAL("Invalid hash expected={0} read={1} key={2}",hash_value,check_hash_value,key);
    }
  }
  else
    m_reader.read(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::Impl::
_setFileOffset(const String& key_name)
{
  // Avec les versions antérieures à la version 3, c'est l'appelant qui
  // positionne l'offset car il est le seul à le connaitre.
  if (m_version >= 3) {
    Impl::DataInfo& data = findData(key_name);
    m_reader.setFileOffset(data.m_file_offset);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String KeyValueTextReader::
fileName() const
{
  return m_p->m_reader.fileName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
setFileOffset(Int64 v)
{
  m_p->m_reader.setFileOffset(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
setDataCompressor(Ref<IDataCompressor> ds)
{
  m_p->m_data_compressor = ds;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IDataCompressor> KeyValueTextReader::
dataCompressor() const
{
  return m_p->m_data_compressor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
setHashAlgorithm(Ref<IHashAlgorithm> v)
{
  m_p->m_hash_algorithm = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IHashAlgorithm> KeyValueTextReader::
hashAlgorithm() const
{
  return m_p->m_hash_algorithm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
readIntegers(const String& key, Span<Integer> values)
{
  m_p->readIntegers(key, values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextReader::
read(const String& key, Span<std::byte> values)
{
  m_p->read(key, values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
