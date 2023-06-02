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
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/String.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceAccessor.h"

#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
// TODO: ajouter information sur la version.
// TODO: ajouter verrou pour la création des répertoires

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class FileHashDatabase
: public TraceAccessor
, public IHashDatabase
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

  FileHashDatabase(ITraceMng* tm, const String& directory)
  : TraceAccessor(tm)
  , m_directory(directory)
  {
  }
  ~FileHashDatabase()
  {
    info() << "FileHashDatabase: nb_write_cache=" << m_nb_write_cache
           << " nb_write=" << m_nb_write << " nb_read=" << m_nb_read;
  }

 public:

  void writeValues(const HashDatabaseWriteArgs& args, HashDatabaseWriteResult& xresult) override
  {
    Span<const std::byte> bytes = args.values();
    String hash_value = args.hashValue();
    xresult.setHashValueAsString(hash_value);
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
      binaryWrite(ofile, bytes);
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
      binaryRead(ifile, bytes);
      ++m_nb_read;
      if (!ifile.good())
        ARCANE_FATAL("Can not read file '{0}'", full_filename);
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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IHashDatabase>
createFileHashDatabase(ITraceMng* tm, const String& directory)
{
  return makeRef<IHashDatabase>(new FileHashDatabase(tm, directory));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
