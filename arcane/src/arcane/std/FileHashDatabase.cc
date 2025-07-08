// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FileHashDatabase.cc                                         (C) 2000-2025 */
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
/*!
 * \brief Base de donnée de hashs sous la forme d'un système de fichier.
 *
 * L'implémentation est similaire à ce que peut faire 'git'.
 *
 * Pour chaque \a hash on écrit un fichier dont le nom sera le \a hash et
 * dont la valeur sera le tableau d'octets correspondant à ce \a hash.
 * Un fichier n'est donc écrit qu'une seule fois et sa valeur ne change
 * jamais: en écriture si le fichier existe déjà il n'est donc pas nécessaire
 * de l'écrire à nouveau.
 *
 * Afin d'éviter d'avoir trop de \a hash dans le même répertoire, on créé deux
 * sous-répertoires. Le premier avec la première lettre du hash et le second avec
 * les deux lettres suivantes. Donc par exemple si le \a hash est '0a239fb4', alors
 * il sera dans le répertoire '0/a2/0a239fb4'.
 */
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
    // TODO: utiliser des verrous lors de la création des sous-répertoires pour éviter
    // que deux processus le fassent en même temps
    platform::recursiveCreateDirectory(base_name);
    // Si le hash est déjà sauvé, ne fait rien
    {
      // TODO: il faudrait relire la valeur pour vérifier que tout est OK dans la base
      // et aussi récupérer la taille du fichier pour garantir la cohérence.
      std::ifstream ifile(full_filename.localstr(), ios::binary);
      if (ifile.good()) {
        ++m_nb_write_cache;
        //std::cout << "FILE_FOUND hash=" << hash_value << " name=" << key << "\n";
        return;
      }
    }
    // TODO: Ajouter verrou pour l'écriture si plusieurs processus écrivent en même temps
    {
      String full_filename = dirfile_info.full_filename;
      ofstream ofile(full_filename.localstr(), ios::binary);
      //info() << "WRITE_HASH hash=" << hash_value << " size=" << bytes.size() << " file=" << full_filename;
      binaryWrite(ofile, bytes);
      ++m_nb_write;
      if (!ofile)
        ARCANE_FATAL("Can not write hash for filename '{0}'", full_filename);
    }
  }

  void readValues(const HashDatabaseReadArgs& args) override
  {
    const String& hash_value = args.hashValueAsString();
    Span<std::byte> bytes = args.values();
    DirFileInfo dirfile_info = _getDirFileInfo(hash_value);
    {
      String full_filename = dirfile_info.full_filename;
      //info() << "READ_VALUE hash_value=" << hash_value << " name=" << args.key() << " file=" << full_filename;
      std::ifstream ifile(full_filename.localstr(), ios::binary);
      if (!ifile.good())
        ARCANE_FATAL("Can not open file '{0}'", full_filename);
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
