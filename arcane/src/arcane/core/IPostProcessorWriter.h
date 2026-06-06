// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPostProcessorWriter.h                                      (C) 2000-2026 */
/*                                                                           */
/* Interface for a writer for post-processing information.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPOSTPROCESSORWRITER_H
#define ARCANE_CORE_IPOSTPROCESSORWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICaseOptionList;
class IDataWriter;
class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup IO
 * \brief Interface for a writer for post-processing information.
 *
 * The instance must return an IDataWriter (via dataWriter()) to
 * handle the writing.
 *
 * The caller must position the instant fields and start
 * the writing via the call IVariableMng::writePostProcessing(). For example
 * \code
 * IPostProcessorWriter* pp = ...;
 * pp->setBaseDirectoryName(...);
 * pp->setTimes(...);
 * pp->setVariables(...);
 * pp->setGroups(...);
 * IVariableMng* vm = ...;
 * vm->writerPostProcessing(pp);
 * \endcode
 *
 * Before writing the variables, the IVariableMng instance will call
 * notifyBeginWrite(). After writing, it calls notifyEndWrite().
 */
class ARCANE_CORE_EXPORT IPostProcessorWriter
{
 public:

  //! Releases resources
  virtual ~IPostProcessorWriter() = default;

 public:

  //! Constructs the instance
  virtual void build() = 0;

 public:

  /*!
   * \brief Returns the writer associated with this post-processor.
   *
   * The returned pointer is only valid between calls
   * to notifyBeginWrite() and notifyEndWrite().
   */
  virtual IDataWriter* dataWriter() = 0;

  /*!
   * \brief Positions the output directory name for files.
   *
   * This directory must exist.
   */
  virtual void setBaseDirectoryName(const String& dirname) = 0;

  // TODO: Deprecate in 2027: use getBaseDirectoryName() instead
  //! Name of the output directory for files.
  virtual const String& baseDirectoryName() = 0;

  //! Name of the output directory for files.
  virtual String getBaseDirectoryName();

  /*!
   * \brief Positions the name of the file containing the outputs.
   *
   * Not all writers support changing the file name.
   */
  virtual void setBaseFileName(const String& filename) = 0;

  // TODO: Deprecate in 2027: use getBaseFileName() instead
  //! Name of the file containing the outputs.
  virtual const String& baseFileName() = 0;

  //! Name of the file containing the outputs.
  virtual String getBaseFileName();

  /*!
   * \brief Positions the mesh.
   *
   * If not overloaded, this method does nothing.
   *
   * \deprecated This method is obsolete. It is no longer possible
   * to change the mesh of a service implementing this interface.
   * The choice of the mesh is made when creating the service via
   * ServiceBuilder by passing the desired mesh as an argument.
   */
  ARCANE_DEPRECATED_REASON("Y2022: Choose the mesh during service creation via ServiceBuilder")
  virtual void setMesh(IMesh* mesh);

  //! Positions the list of times
  virtual void setTimes(ConstArrayView<Real> times) = 0;

  //! List of saved times
  virtual ConstArrayView<Real> times() = 0;

  //! Positions the list of variables to output
  virtual void setVariables(VariableCollection variables) = 0;

  //! List of variables to save
  virtual VariableCollection variables() = 0;

  /*!
   * \brief Positions the list of groups to output.
   *
   * The collection passed as an argument is cloned.
   */
  virtual void setGroups(ItemGroupCollection groups) = 0;

  //! List of groups to save
  virtual ItemGroupCollection groups() = 0;

 public:

  //! Notifies that an output is going to be performed with the current parameters.
  virtual void notifyBeginWrite() = 0;

  //! Notifies that an output has just been performed.
  virtual void notifyEndWrite() = 0;

 public:

  //! Closes the writer. After closing, it can no longer be used
  virtual void close() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
