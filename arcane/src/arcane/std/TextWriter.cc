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
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/JSONWriter.h"

#include "arcane/IDeflateService.h"
#include "arcane/ArcaneException.h"

#include <fstream>
#include <map>

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
  ofstream m_ostream;
  Ref<IDeflateService> m_deflater;
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
  ios::openmode mode = ios::out | ios::binary;
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

String TextWriter::
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

ostream& TextWriter::
stream()
{
  return m_p->m_ostream;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class KeyValueTextWriter::Impl
{
 public:
  static constexpr int MAX_SIZE = 8;
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
    Int32 nb = 0;
    Int64 sizes[MAX_SIZE];
  };
  struct DataInfo
  {
    Int64 m_file_offset = 0;
    ExtentsInfo m_extents;
  };
 public:
  Impl(const String& filename,Int32 version)
  : m_writer(filename), m_version(version)
  {}
 public:
  TextWriter m_writer;
  std::map<String,DataInfo> m_data_infos;
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
    _writeEpilog();
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
  Byte header[12];
  // 4 premiers octets pour indiquer qu'il s'agit d'un fichier de protections
  // arcane (ACR pour Arcane Checkpoint Restart)
  header[0] = 'A';
  header[1] = 'C';
  header[2] = 'R';
  header[3] = (Byte)39;
  // 4 octets suivant pour la version
  // Actuellement version 1 ou 2. La version 1 ne supporte que les fichiers
  // de taille sur 32 bits. La seule différence de la version 2 est que
  // les offsets et longueurs stoquées à la fin du fichier sont sur
  // 64bit au lieu de 32.
  header[4] = (Byte)m_p->m_version;
  header[5] = 0;
  header[6] = 0;
  header[7] = 0;
  // 4 octets suivant pour indiquer l'indianness.
  Int32 v = 0x01020304;
  Byte* ptr = (Byte*)(&v);
  for( Integer i=0; i<4; ++i )
    header[8+i] = ptr[i];
  m_p->m_writer.stream().write((const char*)header,12);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
_writeEpilog()
{
  std::cout << "EPILOG!\n";

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
  // Conserve la position dans le fichier des méta-données
  // ainsi que leur taille.
  Int64 file_offset = fileOffset();
  StringView buf = jsw.getBuffer();
  Int64 meta_data_size = buf.size();
  m_p->m_writer.stream().write((const char*)buf.bytes().data(),meta_data_size);

  {
    Int64 write_info[2];
    write_info[0] = file_offset;
    write_info[1] = meta_data_size;

    // Doit toujours être la dernière écriture du fichier
    Span<const Int64> wi(write_info,2);
    auto wi_bytes = asBytes(wi);
    m_p->m_writer.stream().write((const char*)wi_bytes.data(),wi_bytes.size());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
setExtents(const String& key_name,Int64ConstArrayView extents)
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
        m_p->m_writer.write(dims);
      }
      else
        // Sauve les dimensions comme un tableau de Int64
        m_p->m_writer.write(extents);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
write(const String& key,Span<const Real> values)
{
  _writeKey(key);
  m_p->m_writer.write(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
write(const String& key,Span<const Int16> values)
{
  _writeKey(key);
  m_p->m_writer.write(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
write(const String& key,Span<const Int32> values)
{
  _writeKey(key);
  m_p->m_writer.write(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
write(const String& key,Span<const Int64> values)
{
  _writeKey(key);
  m_p->m_writer.write(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KeyValueTextWriter::
write(const String& key,Span<const Byte> values)
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
setDeflater(Ref<IDeflateService> ds)
{
  m_p->m_writer.setDeflater(ds);
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
_addKey(const String& key,Int64ConstArrayView extents)
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

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
