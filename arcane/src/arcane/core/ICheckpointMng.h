// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICheckpointMng.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface of the checkpoint information manager.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICHECKPOINTMNG_H
#define ARCANE_CORE_ICHECKPOINTMNG_H
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
 * \brief Interface of the checkpoint information manager.
 *
 * This manager handles checkpoint information, namely the saved times, the
 * services used, and other information necessary for recovery. It does not
 * directly manage the writing or reading, which are delegated to an
 * ICheckpointReader or ICheckpointWriter.
 *
 * Reading a checkpoint causes the modification of all variables and meshes.
 */
class ARCANE_CORE_EXPORT ICheckpointMng
{
 public:

  virtual ~ICheckpointMng() = default; //!< Frees resources.

 public:

  /*!
   * \brief Reads a checkpoint.
   *
   * This operation is collective.
   *
   * \deprecated Use readDefaultCheckpoint() instead
   */
  ARCANE_DEPRECATED_122 virtual void readCheckpoint() = 0;

  /*!
   * \brief Reads a checkpoint.
   *
   * Reads a checkpoint from the \a reader.
   */
  virtual void readCheckpoint(ICheckpointReader* reader) = 0;

  /*!
   * \brief Reads a checkpoint.
   *
   * Reads a checkpoint whose reading information is in \a infos.
   *
   * \deprecated Instead, use the following code:
   * \code
   * ICheckpointMng* cm = ...;
   * Span<const Byte> buffer;
   * CheckpointInfo checkpoint_info = cm->readChekpointInfo(buffer);
   * cm->readChekpoint(checkpoint_info);
   * \endcode
   */
  virtual ARCANE_DEPRECATED_2018 void readCheckpoint(ByteConstArrayView infos) = 0;

  /*!
   * \brief Reads checkpoint information.
   *
   * Reads the information of a checkpoint contained in the \a infos buffer.
   * \a buf_name contains the name of the buffer used in displays in case of an error.
   */
  virtual CheckpointInfo readCheckpointInfo(Span<const Byte> infos, const String& buf_name) = 0;

  /*!
   * \brief Reads a checkpoint.
   *
   * Reads a checkpoint whose information is in \a checkpoint_infos.
   */
  virtual void readCheckpoint(const CheckpointInfo& checkpoint_info) = 0;

  /*!
   * \brief Reads a default checkpoint
   *
   * This operation is collective.
   *
   * In the default implementation, the information for resumption is stored in
   * a file named 'checkpoint_info.xml' located in the case's export directory
   * (ISubDomain::exportDirectory()).
   *
   * \deprecated Instead, use the following code:
   * \code
   * ICheckpointMng* cm = ...;
   * CheckpointInfo checkpoint_info = cm->readDefaultChekpointInfo();
   * cm->readChekpoint(checkpoint_info);
   * \endcode
   */
  virtual ARCANE_DEPRECATED_2018 void readDefaultCheckpoint() = 0;

  /*!
   * \brief Reads default checkpoint information.
   *
   * This operation is collective.
   *
   * In the default implementation, the information for resumption is stored
   * in a file named 'checkpoint_info.xml' located in the case's export
   * directory (ISubDomain::exportDirectory()).
   *
   * After reading the information, it is possible to call
   * readCheckpoint(const CheckpointInfo& checkpoint_info) to read the checkpoint.
   */
  virtual CheckpointInfo readDefaultCheckpointInfo() = 0;

  /*!
   * \brief Writes a default checkpoint using the \a writer.
   *
   * This operation is collective.
   *
   * \deprecated Use writeDefaultCheckpoint() instead.
   */
  ARCANE_DEPRECATED_122 virtual void writeCheckpoint(ICheckpointWriter* writer) = 0;

  /*!
   * \brief Writes a checkpoint using the \a writer.
   *
   * This operation is collective.
   *
   * The information required to read it back is stored in the \a infos array
   * passed as an argument. It is then possible to read a checkpoint back via
   * readCheckpoint(ByteConstArrayView).
   *
   * The default implementation stores in infos an XML file containing, among
   * other things, the name of the corresponding reader, the number of subdomains, ...
   */
  virtual void writeCheckpoint(ICheckpointWriter* writer, ByteArray& infos) = 0;

  /*!
   * \brief Writes a checkpoint using the \a writer.
   *
   * This operation is collective.
   *
   * This is a standard checkpoint that can be read back via readDefaultCheckpoint().
   *
   * \sa readDefaultCheckpoint
   */
  virtual void writeDefaultCheckpoint(ICheckpointWriter* writer) = 0;

  /*!
   * \brief Write observable.
   *
   * Observers registered in this observable are called before writing a checkpoint.
   */
  virtual IObservable* writeObservable() = 0;

  /*!
   * \brief Read observable.
   *
   * Observers registered in this observable are called after a complete checkpoint read.
   */
  virtual IObservable* readObservable() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
