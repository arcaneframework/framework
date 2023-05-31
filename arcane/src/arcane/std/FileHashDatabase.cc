// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FileHashDatabase.cc                                         (C) 2000-2023 */
/*                                                                           */
/* Base de données de hash gérée par le système de fichier.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/IHashDatabase.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/SHA3HashAlgorithm.h"
#include "arcane/utils/MD5HashAlgorithm.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/String.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
// TODO: ajouter information sur la version.
// TODO: ajouter verrou pour la création des répertoires
// TODO: ajouter support pour la compression

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class FileHashDatabase
: public IHashDatabase
{
 public:

  //! Nom du répertoire et du fichier contenant le hash
  struct DirFileInfo
  {
    String directory;
    String filename;
    String full_filename;
  };

 public:

  FileHashDatabase(const String& directory)
  : m_directory(directory)
  {
  }
  ~FileHashDatabase()
  {
    Real nb_byte_per_second = static_cast<Real>(m_nb_processed_bytes) / (m_hash_time + 1.0e-9);
    // Pour avoir en Mega-byte par seconde
    nb_byte_per_second /= 1.0e6;
    std::cout << "FileHashDatabase: nb_write_cache=" << m_nb_write_cache
              << " nb_write=" << m_nb_write << " nb_read=" << m_nb_read
              << " nb_processed=" << m_nb_processed_bytes
              << " hash_time=" << m_hash_time
              << " rate=" << static_cast<Int64>(nb_byte_per_second) << " MB/s"
              << "\n";
  }

 public:

  void writeValues(const HashDatabaseWriteArgs& args, HashDatabaseWriteResult& xresult) override
  {
    const String& key = args.key();
    Span<const std::byte> bytes = args.values();
    m_nb_processed_bytes += bytes.size();
    SHA3_256HashAlgorithm hash_algo;
    //MD5HashAlgorithm hash_algo;
    SmallArray<Byte, 1024> hash_result;
    {
      Real t1 = platform::getRealTime();
      hash_algo.computeHash64(bytes, hash_result);
      Real t2 = platform::getRealTime();
      m_hash_time += (t2 - t1);
    }
    String hash_value = Convert::toHexaString(hash_result);
    xresult.setHashValueAsString(hash_value);
    //std::cout << "COMPUTE_KEYVALUE_HASH hash=" << hash_value << " name=" << key << "\n";
    DirFileInfo dirfile_info = _getDirFileInfo(hash_value);
    String base_name = dirfile_info.directory;
    String full_filename = dirfile_info.full_filename;
    // TODO: sauver les répertoires créés dans une base pour ne pas le refaire à chaque fois.
    platform::recursiveCreateDirectory(base_name);
    // Si le hash est déjà sauvé, ne fait rien
    // TODO: il faudrait relire la valeur pour vérifier que tout est OK dans la base
    {
      std::ifstream ifile(full_filename.localstr());
      if (ifile.good()) {
        ++m_nb_write_cache;
        //std::cout << "FILE_FOUND hash=" << hash_value << " name=" << key << "\n";
        return;
      }
    }
    // TODO: Ajouter verrou pour l'écriture si plusieurs processus écrivent en même temps
    {
      String full_filename = dirfile_info.full_filename;
      ofstream ofile(full_filename.localstr());
      //std::cout << "WRITE_HASH hash=" << hash_value << " size=" << bytes.size() << "\n";
      ofile.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      ++m_nb_write;
      if (!ofile)
        ARCANE_FATAL("Can not write hash for filename '{0}'", full_filename);
    }
  }

  void readValues(const HashDatabaseReadArgs& args)
  {
    const String& hash_value = args.hashValueAsString();
    Span<std::byte> bytes = args.values();
    //std::cout << "READ_VALUE hash_value" << hash_value << " name=" << args.key() << "\n";
    DirFileInfo dirfile_info = _getDirFileInfo(hash_value);
    {
      String full_filename = dirfile_info.full_filename;
      std::ifstream ifile(full_filename.localstr());
      ifile.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
      ++m_nb_read;
      if (!ifile.good())
        ARCANE_FATAL("Can not read file '{0}' full_filename");
    }
  }

 private:

  DirFileInfo _getDirFileInfo(const String& hash_value)
  {
    char name1 = hash_value.bytes()[0];
    char name2 = hash_value.bytes()[1];
    char name3 = hash_value.bytes()[2];
    StringBuilder base_name_builder = m_directory;
    base_name_builder += '/';
    base_name_builder += name1;
    base_name_builder += '/';
    base_name_builder += name2;
    base_name_builder += name3;
    String base_name = base_name_builder.toString();
    String full_filename = String::format("{0}/{1}", base_name, hash_value);
    return DirFileInfo{ base_name, hash_value, full_filename };
  }

 private:

  String m_directory;
  Int64 m_nb_write_cache = 0;
  Int64 m_nb_write = 0;
  Int64 m_nb_read = 0;
  Int64 m_nb_processed_bytes = 0;
  Real m_hash_time = 0.0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IHashDatabase>
createFileHashDatabase(const String& directory)
{
  return makeRef<IHashDatabase>(new FileHashDatabase(directory));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
