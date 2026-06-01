// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FileHashDatabase.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Hash database managed by the file system.                                 */
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
// TODO: add version information.
// TODO: add lock for directory creation

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Hash database implemented as a file system.
 *
 * The implementation is similar to what 'git' can do.
 *
 * For each hash, we write a file whose name will be the hash and
 * whose value will be the byte array corresponding to this hash.
 * A file is therefore written only once and its value never changes:
 * on write: if the file already exists, it is not necessary
 * to write it again.
 *
 * To avoid having too many hashes in the same directory, we create two
 * subdirectories. The first with the first letter of the hash and the second with
 * the next two letters. So for example, if the hash is '0a239fb4', then
 * it will be in the directory '0/a2/0a239fb4'.
 */
class FileHashDatabase
: public TraceAccessor
, public IHashDatabase
{
 public:

  //! Name of the directory and file containing the hash
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
    // TODO: save the created directories in a database so as not to do it every time.
    // TODO: use locks when creating subdirectories to prevent
    // two processes from doing it at the same time
    platform::recursiveCreateDirectory(base_name);
    // If the hash is already saved, do nothing
    {
      // TODO: we should reread the value to verify that everything is OK in the database
      // and also retrieve the file size to guarantee consistency.
      std::ifstream ifile(full_filename.localstr(), ios::binary);
      if (ifile.good()) {
        ++m_nb_write_cache;
        //std::cout << "FILE_FOUND hash=" << hash_value << " name=" << key << "\n";
        return;
      }
    }
    // TODO: Add lock for writing if multiple processes write at the same time
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
