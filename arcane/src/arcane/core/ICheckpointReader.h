// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICheckpointReader.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface for the protection/recovery reading service.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICHECKPOINTREADER_H
#define ARCANE_CORE_ICHECKPOINTREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup StandardService
 * \brief Interface for the protection/recovery reading service.
 */
class ARCANE_CORE_EXPORT ICheckpointReader
{
 public:

  //! Frees resources
  virtual ~ICheckpointReader() = default;

 public:

  //! Returns the associated reader
  virtual IDataReader* dataReader() = 0;

  //! Notifies that a protection will be read with current parameters
  virtual void notifyBeginRead() = 0;

  //! Notifies that a protection has just been read
  virtual void notifyEndRead() = 0;

  //! Sets the name of the protection file
  virtual void setFileName(const String& file_name) = 0;

  //! Name of the protection file
  virtual String fileName() const = 0;

  //! Sets the name of the protection base directory
  virtual void setBaseDirectoryName(const String& dirname) = 0;

  //! Name of the protection base directory
  virtual String baseDirectoryName() const = 0;

  //! Metadata associated with this reader.
  virtual void setReaderMetaData(const String&) = 0;

  //! Sets the time and index of the protection to read
  virtual void setCurrentTimeAndIndex(Real current_time, Integer current_index) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface for the protection/recovery reading service (V2)
 */
class ARCANE_CORE_EXPORT ICheckpointReader2
{
 public:

  //! Frees resources
  virtual ~ICheckpointReader2() = default;

 public:

  //! Returns the data reader associated with this protection reader
  virtual IDataReader2* dataReader() = 0;

  /*!
   * \brief Notifies that a protection will be read with information
   * from \a checkpoint_info.
   */
  virtual void notifyBeginRead(const CheckpointReadInfo& cri) = 0;

  //! Notifies the end of reading a protection.
  virtual void notifyEndRead() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
