// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVerifierService.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface of the data verification service between two executions.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVERIFIERSERVICE_H
#define ARCANE_CORE_IVERIFIERSERVICE_H
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
 * \brief Interface of the data verification service
 * between two executions.
 */
class IVerifierService
{
 public:

  /*!
   * \brief Comparison mode
   */
  enum class eCompareMode
  {
    //! Compare all values
    Values,
    /*!
     * \brief Compares only the hashes of the values.
     *
     * This mode only allows detecting if two values are different
     * without knowing this difference. However, it is faster
     * than the \a Values mode and allows limiting the size of the comparison
     * files.
     */
    HashOnly
  };

 public:

  //! Frees resources
  virtual ~IVerifierService() = default;

 public:

  //! Writes the reference file
  virtual void writeReferenceFile() = 0;

  /*!
   * \brief Performs the verification from the reference file.
   *
   * \param parallel_sequential if true, indicates that the result
   * of a parallel execution is compared with that of a sequential execution. This
   * option is inactive if the execution is sequential.
   *
   * \param compare_ghost if true, indicates that the results are also compared
   * on ghost entities. It is generally normal for the results to be
   * different on ghost entities, because it is not necessary that
   * all variables are synchronized. This is why it is better
   * generally not to perform verification on ghost entities. This
   * option is inactive if the execution is sequential.
   */
  virtual void doVerifFromReferenceFile(bool parallel_sequential, bool compare_ghost) = 0;

 public:

  //! Sets the name of the file containing the reference values
  virtual void setFileName(const String& file_name) = 0;
  //! Name of the file containing the reference values
  virtual String fileName() const = 0;

 public:

  //! Name of the file containing the results
  virtual void setResultFileName(const String& file_name) = 0;
  virtual String resultfileName() const = 0;

  //! Desired comparison type
  virtual void setCompareMode(eCompareMode v) = 0;
  virtual eCompareMode compareMode() const = 0;

 public:

  //! Sets the name of the subdirectory containing the reference values
  virtual void setSubDir(const String& sub_dir) = 0;
  //! Name of the file containing the reference values
  virtual String subDir() const = 0;

 public:

  //! Method to use for calculating the difference between two values
  virtual void setComputeDifferenceMethod(eVariableComparerComputeDifferenceMethod v) = 0;
  virtual eVariableComparerComputeDifferenceMethod computeDifferenceMethod() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
