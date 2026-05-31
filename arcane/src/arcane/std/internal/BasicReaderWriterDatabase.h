// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicReaderWriterDatabase.h                                 (C) 2000-2023 */
/*                                                                           */
/* Database for the 'BasicReaderWriter' service.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_BASICREADERWRITERDATABASE_H
#define ARCANE_STD_BASICREADERWRITERDATABASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Once we are certain that this file is not used outside of
// Arcane, we can merge these classes with their implementation

namespace Arcane
{
class IDataCompressor;
class IHashAlgorithm;
}

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * Using a TextWriter with (key,value) format writing.
 *
 * For each value to be written, setExtents() must first be called to
 * position the data dimensions, and then write() must be called to write the
 * values. This is necessary to maintain compatibility with versions 1 and 2 of the
 * format, where data was written sequentially.
 */
class KeyValueTextWriter
: public TraceAccessor
{
  class Impl;

 public:

  KeyValueTextWriter(ITraceMng* tm,const String& filename, Int32 version);
  KeyValueTextWriter(const KeyValueTextWriter& rhs) = delete;
  ~KeyValueTextWriter();
  KeyValueTextWriter& operator=(const KeyValueTextWriter& rhs) = delete;

 public:

  void setExtents(const String& key_name, SmallSpan<const Int64> extents);
  void write(const String& key, Span<const std::byte> values);
  Int64 fileOffset();

 public:

  String fileName() const;
  void setDataCompressor(Ref<IDataCompressor> dc);
  Ref<IDataCompressor> dataCompressor() const;
  void setHashAlgorithm(Ref<IHashAlgorithm> v);
  Ref<IHashAlgorithm> hashAlgorithm() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Text file writing class for backups/restorations
 */
class KeyValueTextReader
: public TraceAccessor
{
  class Impl;

 public:

  KeyValueTextReader(ITraceMng* tm,const String& filename, Int32 version);
  KeyValueTextReader(const KeyValueTextReader& rhs) = delete;
  ~KeyValueTextReader();
  KeyValueTextReader& operator=(const KeyValueTextReader& rhs) = delete;

 public:

  void setFileOffset(Int64 v);
  void getExtents(const String& key_name, SmallSpan<Int64> extents);
  void readIntegers(const String& key, Span<Integer> values);
  void read(const String& key, Span<std::byte> values);

 public:

  String fileName() const;
  void setDataCompressor(Ref<IDataCompressor> ds);
  Ref<IDataCompressor> dataCompressor() const;
  void setHashAlgorithm(Ref<IHashAlgorithm> v);
  Ref<IHashAlgorithm> hashAlgorithm() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
