// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMultiReduce.h                                              (C) 2000-2016 */
/*                                                                           */
/* Management of multiple reductions.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_IMULTIREDUCE_H
#define ARCANE_PARALLEL_IMULTIREDUCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class managing a reduction of a sum of values.
 *
 * Instances of this class must be created via IMultiReduce::getSumOfReal().
 * The user must accumulate values via the call to add(). After execution
 * of the reduction via IMultiReduce::execute(), it is possible
 * to retrieve the reduced value via reducedValue().
 * \sa IMultiReduce
 */
class ARCANE_CORE_EXPORT ReduceSumOfRealHelper
{
 public:

  ReduceSumOfRealHelper(bool is_strict)
  : m_reduced_value(0.0)
  , m_is_strict(is_strict)
  {
    if (!m_is_strict)
      m_values.add(0.0);
  }

 public:

  //! Adds the value \a v
  void add(Real v)
  {
    if (m_is_strict)
      m_values.add(v);
    else
      m_values[0] += v;
  }

  //! Clears the accumulated values.
  void clear()
  {
    m_values.clear();
  }

  //! List of accumulated values.
  RealConstArrayView values() const { return m_values; }

  //! Reduced value
  Real reducedValue() const { return m_reduced_value; }

  //! Positions the reduced value.
  void setReducedValue(Real v) { m_reduced_value = v; }

 private:

  SharedArray<Real> m_values;
  Real m_reduced_value;
  bool m_is_strict;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Management of multiple reductions.
 *
 * For now, only 'sum' type reductions on reals
 * are supported.
 *
 * It is possible to specify a strict mode, via setStrict(), which
 * allows these sums to be identical regardless of the order
 * of the operations. However, this requires storing all intermediate values
 * and is therefore memory-intensive and not scalable because
 * a single processor will handle the sum calculation.
 *
 * The strict mode must be specified before creating the reductions.
 * The strict mode is automatically active if the environment variable
 * ARCANE_STRICT_REDUCE is set.
 */
class ARCANE_CORE_EXPORT IMultiReduce
{
 public:

  virtual ~IMultiReduce() {} //!< Frees resources

 public:

  static IMultiReduce* create(IParallelMng* pm);

 public:

  //! Executes the reductions
  virtual void execute() = 0;

  //! Indicates if strict mode is used
  virtual bool isStrict() const = 0;

  //! Sets the strict mode
  virtual void setStrict(bool is_strict) = 0;

 public:

  /*!
   * \brief Returns the name manager \a name.
   * If a name manager \a name does not exist, it is created.
   * The returned object remains the property of this instance and must not
   * be explicitly destroyed. It will be when this instance is
   * destroyed.
   */
  virtual ReduceSumOfRealHelper* getSumOfReal(const String& name) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
