// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICheckpointWriter.h                                         (C) 2000-2025 */
/*                                                                           */
/* Checkpoint/Recovery write service interface.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICHECKPOINTWRITER_H
#define ARCANE_CORE_ICHECKPOINTWRITER_H
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
 * \brief Interface of the checkpoint/recovery write service.
 *
 * The instance must return an IDataWriter (via dataWriter()) to
 * handle the writing.
 *
 * The sequence of functions is as follows:
 * \code
 * ICheckpointWriter* checkpoint_writer = ...;
 * checkpoint_writer->setCheckpointTimes();
 * checkpoint_writer->notifyBeginWrite();
 * checkpoint_writer->dataWriter();
 * // ...
 * // Writing with the IDataWriter
 * // ...
 * checkpoint_writer->notifyBeginWrite();
 * checkpoint_writer->readerServiceName();
 * checkpoint_writer->readerMetaData();
 * \endcode
 */
class ARCANE_CORE_EXPORT ICheckpointWriter
{
 public:

  //! Releases resources
  virtual ~ICheckpointWriter() = default;

 public:

  /*!
   * \brief Returns the associated writer.
   */
  virtual IDataWriter* dataWriter() = 0;

  //! Notifies that a checkpoint is going to be written with the current parameters
  virtual void notifyBeginWrite() = 0;

  //! Notifies that a checkpoint has just been written
  virtual void notifyEndWrite() = 0;

  //! Sets the name of the checkpoint file
  virtual void setFileName(const String& file_name) = 0;

  //! Name of the checkpoint file
  virtual String fileName() const = 0;

  //! Sets the name of the checkpoint base directory
  virtual void setBaseDirectoryName(const String& dirname) = 0;

  //! Name of the checkpoint base directory
  virtual String baseDirectoryName() const = 0;

  /*! \brief Sets the checkpoint times.
   *
   * The time of the current checkpoint is the last element of the array
   */
  virtual void setCheckpointTimes(RealConstArrayView times) = 0;

  //! Checkpoint times
  virtual ConstArrayView<Real> checkpointTimes() const = 0;

  //! Closes the checkpoints
  virtual void close() = 0;

  //! Name of the reader service associated with this writer
  virtual String readerServiceName() const = 0;

  //! Metadata for the reader associated with this writer
  virtual String readerMetaData() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
