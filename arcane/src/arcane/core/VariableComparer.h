// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableComparer.h                                          (C) 2000-2025 */
/*                                                                           */
/* Class to perform comparisons between variables.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLECOMPARER_H
#define ARCANE_CORE_VARIABLECOMPARER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Comparison method to use
enum class eVariableComparerCompareMode
{
  //! Compares with a reference
  Same = 0,
  //! Checks that the variable is synchronized
  Sync = 1,
  //! Checks that the variable values are the same on all replicas
  SameOnAllReplica = 2
};

//! Method used to calculate the difference between two values \a v1 and \a v2.
enum class eVariableComparerComputeDifferenceMethod
{
  //! Uses (v1-v2) / v1
  Relative,
  /*!
   * \brief Uses (v1-v2) / local_norm_max.
   *
   * \a local_norm_max is the maximum of math::abs() of the values on the subdomain.
   */
  LocalNormMax,
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arguments for VariableComparer methods.
 */
class ARCANE_CORE_EXPORT VariableComparerArgs
{
 public:

  /*!
   * \brief Sets the number of errors to display in the listing.
   *
   * If 0, no elements are displayed. If positive, displays at most
   * \a v elements. If negative, all elements are displayed.
   */
  void setMaxPrint(Int32 v) { m_max_print = v; }
  Int32 maxPrint() const { return m_max_print; }

  /*!
   * \brief Indicates on which entities the comparison is performed.
   *
   * If \a v is true, compares the values both on the proper entities and
   * the ghost entities. Otherwise, it only performs the comparison on the
   * proper entities.
   *
   * This parameter is only used if compareMode() equals eCompareMode::Same.
   */
  void setCompareGhost(bool v) { m_is_compare_ghost = v; }
  bool isCompareGhost() const { return m_is_compare_ghost; }

  void setDataReader(IDataReader* v) { m_data_reader = v; }
  IDataReader* dataReader() const { return m_data_reader; }

  void setCompareMode(eVariableComparerCompareMode v) { m_compare_mode = v; }
  eVariableComparerCompareMode compareMode() const { return m_compare_mode; }

  void setComputeDifferenceMethod(eVariableComparerComputeDifferenceMethod v) { m_compute_difference_method = v; }
  eVariableComparerComputeDifferenceMethod computeDifferenceMethod() const { return m_compute_difference_method; }

 private:

  Int32 m_max_print = 0;
  bool m_is_compare_ghost = false;
  IDataReader* m_data_reader = nullptr;
  eVariableComparerCompareMode m_compare_mode = eVariableComparerCompareMode::Same;
  eVariableComparerComputeDifferenceMethod m_compute_difference_method = eVariableComparerComputeDifferenceMethod::Relative;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Results of a comparison operation.
 */
class ARCANE_CORE_EXPORT VariableComparerResults
{
 public:

  VariableComparerResults() = default;
  explicit VariableComparerResults(Int32 nb_diff)
  : m_nb_diff(nb_diff)
  {}

 public:

  void setNbDifference(Int32 v) { m_nb_diff = v; }
  Int32 nbDifference() const { return m_nb_diff; }

 public:

  Int32 m_nb_diff = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to perform comparisons between variables.
 *
 * To use this class, you must create an instance of
 * VariableComparerArgs via one of the following methods:
 *
 * - buildForCheckIfSame()
 * - buildForCheckIfSync()
 * - buildForCheckIfSameOnAllReplica()
 *
 * You must then call the apply() method with the created instance for
 * each variable you wish to compare.
 */
class ARCANE_CORE_EXPORT VariableComparer
{
 public:

  VariableComparer() = default;

 public:

  /*!
   * \brief Creates a comparison to verify that a variable is synchronized.
   *
   * This operation only works for mesh variables.
   *
   * A variable is synchronized when its values are the same across all
   * subdomains, both on proper elements and ghost elements.
   *
   * It is possible to call the methods
   * VariableComparerArgs::setMaxPrint(),
   * VariableComparerArgs::setCompareGhost()
   * or VariableComparerArgs::setComputeDifferenceMethod() on the returned
   * instance to modify the behavior.
   */
  VariableComparerArgs buildForCheckIfSync();

  /*!
   * \brief Creates a comparison to verify that a variable is identical on
   * all replicas.
   *
   * Compares the variable values with those of the same subdomain on other
   * replicas. For each differing element, a message is displayed.
   *
   * Using apply() for this type of comparison is a collective method on the
   * replica of the variable passed as an argument. Therefore, it should only
   * be called if the variable exists on all subdomains, otherwise it will cause
   * a blockage.
   *
   * This comparison only works for variables of numerical types. In this case,
   * it throws a NotSupportedException.
   *
   * It is possible to call the methods
   * VariableComparerArgs::setMaxPrint() or
   * VariableComparerArgs::setComputeDifferenceMethod() on the returned instance
   * to modify the behavior.
   */
  VariableComparerArgs buildForCheckIfSameOnAllReplica();

 public:

  /*!
   * \brief Creates a comparison to verify that a variable is identical to
   * a reference value.
   *
   * This operation verifies that the variable values are identical to a
   * reference value which will be read from the \a data_reader.
   *
   * It is possible to call the methods
   * VariableComparerArgs::setMaxPrint(),
   * VariableComparerArgs::setCompareGhost() or
   * VariableComparerArgs::setComputeDifferenceMethod() on the returned
   * instance to modify the behavior.
   *
   * It is then possible to call the apply() method on the returned instance
   * to perform comparisons on a variable.
   */
  VariableComparerArgs buildForCheckIfSame(IDataReader* data_reader);

 public:

  //! Applies the comparison \a compare_args to the variable \a var
  VariableComparerResults apply(IVariable* var, const VariableComparerArgs& compare_args);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
