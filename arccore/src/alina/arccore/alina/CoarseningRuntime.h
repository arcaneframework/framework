// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CoarseningRuntime.h                                         (C) 2000-2026 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_COARSENINGRUNTIME_H
#define ARCCORE_ALINA_COARSENINGRUNTIME_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/ValueTypeInterface.h"
#include "arccore/alina/Coarsening.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum class eCoarserningType
{
  ruge_stuben, ///< Ruge-Stueben coarsening
  RugeStubenCoarsening = ruge_stuben,
  aggregation, ///< Aggregation
  AggregationCoarsening = aggregation,
  smoothed_aggregation, ///< Smoothed aggregation
  SmoothedAggregationCoarserning = smoothed_aggregation,
  smoothed_aggr_emin, ///< Smoothed aggregation with energy minimization
  SmoothedAggregationEnergyMinCoarsening = smoothed_aggr_emin
};

inline std::ostream& operator<<(std::ostream& os, eCoarserningType c)
{
  switch (c) {
  case eCoarserningType::ruge_stuben:
    return os << "ruge_stuben";
  case eCoarserningType::aggregation:
    return os << "aggregation";
  case eCoarserningType::smoothed_aggregation:
    return os << "smoothed_aggregation";
  case eCoarserningType::smoothed_aggr_emin:
    return os << "smoothed_aggr_emin";
  default:
    return os << "???";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::istream& operator>>(std::istream& in, eCoarserningType& c)
{
  std::string val;
  in >> val;

  if (val == "ruge_stuben")
    c = eCoarserningType::ruge_stuben;
  else if (val == "aggregation")
    c = eCoarserningType::aggregation;
  else if (val == "smoothed_aggregation")
    c = eCoarserningType::smoothed_aggregation;
  else if (val == "smoothed_aggr_emin")
    c = eCoarserningType::smoothed_aggr_emin;
  else
    throw std::invalid_argument("Invalid coarsening value. Valid choices are: "
                                "ruge_stuben, aggregation, smoothed_aggregation, smoothed_aggr_emin.");

  return in;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend>
struct CoarseningRuntime
{
  typedef Alina::PropertyTree params;
  eCoarserningType c;
  bool as_scalar;
  void* handle = nullptr;

  explicit CoarseningRuntime(params prm = params())
  : c(prm.get("type", eCoarserningType::smoothed_aggregation))
  {
    if (!prm.erase("type"))
      ARCCORE_ALINA_PARAM_MISSING("type");

    typedef typename backend::value_type<Backend>::type value_type;
    const bool block_value_type = math::static_rows<value_type>::value > 1;

    as_scalar = (block_value_type &&
                 c != eCoarserningType::ruge_stuben &&
                 prm.get("nullspace.cols", 0) > 0);
    std::cout << "PreconditionerCoarseningType=" << c << "\n";
    switch (c) {

#define ARCCORE_ALINA_RUNTIME_COARSENING(t) \
  case eCoarserningType::t: \
    if (as_scalar) { \
      handle = call_constructor<AsScalarCoarsening<t>::type>(prm); \
    } \
    else { \
      handle = call_constructor<t>(prm); \
    } \
    break

      ARCCORE_ALINA_RUNTIME_COARSENING(RugeStubenCoarsening);
      ARCCORE_ALINA_RUNTIME_COARSENING(AggregationCoarsening);
      ARCCORE_ALINA_RUNTIME_COARSENING(SmoothedAggregationCoarserning);
      ARCCORE_ALINA_RUNTIME_COARSENING(SmoothedAggregationEnergyMinCoarsening);

#undef ARCCORE_ALINA_RUNTIME_COARSENING

    default:
      throw std::invalid_argument("Unsupported coarsening type");
    }
  }

  ~CoarseningRuntime()
  {
    switch (c) {

#define ARCCORE_ALINA_RUNTIME_COARSENING(t) \
  case eCoarserningType::t: \
    if (as_scalar) { \
      call_destructor<AsScalarCoarsening<t>::type>(); \
    } \
    else { \
      call_destructor<t>(); \
    } \
    break

      ARCCORE_ALINA_RUNTIME_COARSENING(RugeStubenCoarsening);
      ARCCORE_ALINA_RUNTIME_COARSENING(AggregationCoarsening);
      ARCCORE_ALINA_RUNTIME_COARSENING(SmoothedAggregationCoarserning);
      ARCCORE_ALINA_RUNTIME_COARSENING(SmoothedAggregationEnergyMinCoarsening);

#undef ARCCORE_ALINA_RUNTIME_COARSENING
    }
  }

  template <class Matrix>
  std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>
  transfer_operators(const Matrix& A)
  {
    switch (c) {

#define ARCCORE_ALINA_RUNTIME_COARSENING(t) \
  case eCoarserningType::t: \
    if (as_scalar) { \
      return make_operators<AsScalarCoarsening<t>::type>(A); \
    } \
    return make_operators<t>(A)

      ARCCORE_ALINA_RUNTIME_COARSENING(RugeStubenCoarsening);
      ARCCORE_ALINA_RUNTIME_COARSENING(AggregationCoarsening);
      ARCCORE_ALINA_RUNTIME_COARSENING(SmoothedAggregationCoarserning);
      ARCCORE_ALINA_RUNTIME_COARSENING(SmoothedAggregationEnergyMinCoarsening);

#undef ARCCORE_ALINA_RUNTIME_COARSENING

    default:
      throw std::invalid_argument("Unsupported coarsening type");
    }
  }

  template <class Matrix> std::shared_ptr<Matrix>
  coarse_operator(const Matrix& A, const Matrix& P, const Matrix& R) const
  {
    switch (c) {

#define ARCCORE_ALINA_RUNTIME_COARSENING(t) \
      case eCoarserningType::t:            \
    if (as_scalar) { \
      return make_coarse<AsScalarCoarsening<t>::type>(A, P, R); \
    } \
    return make_coarse<t>(A, P, R)

      ARCCORE_ALINA_RUNTIME_COARSENING(RugeStubenCoarsening);
      ARCCORE_ALINA_RUNTIME_COARSENING(AggregationCoarsening);
      ARCCORE_ALINA_RUNTIME_COARSENING(SmoothedAggregationCoarserning);
      ARCCORE_ALINA_RUNTIME_COARSENING(SmoothedAggregationEnergyMinCoarsening);

#undef ARCCORE_ALINA_RUNTIME_COARSENING

    default:
      throw std::invalid_argument("Unsupported coarsening type");
    }
  }

  template <template <class> class Coarsening>
  std::enable_if_t<backend::coarsening_is_supported<Backend, Coarsening>::value, void*>
  call_constructor(const params& prm)
  {
    return static_cast<void*>(new Coarsening<Backend>(prm));
  }

  template <template <class> class Coarsening>
  std::enable_if_t<!backend::coarsening_is_supported<Backend, Coarsening>::value, void*>
  call_constructor(const params&)
  {
    throw std::logic_error("The coarsening is not supported by the backend");
  }

  template <template <class> class Coarsening>
  std::enable_if_t<backend::coarsening_is_supported<Backend, Coarsening>::value, void>
  call_destructor()
  {
    delete static_cast<Coarsening<Backend>*>(handle);
  }

  template <template <class> class Coarsening>
  std::enable_if_t<!backend::coarsening_is_supported<Backend, Coarsening>::value, void>
  call_destructor()
  {
  }

  template <template <class> class Coarsening, class Matrix>
  std::enable_if_t<backend::coarsening_is_supported<Backend, Coarsening>::value,
                          std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>>
  make_operators(const Matrix& A) const
  {
    return static_cast<Coarsening<Backend>*>(handle)->transfer_operators(A);
  }

  template <template <class> class Coarsening, class Matrix>
  std::enable_if_t<!backend::coarsening_is_supported<Backend, Coarsening>::value,
                          std::tuple<
                          std::shared_ptr<Matrix>,
                          std::shared_ptr<Matrix>>>
  make_operators(const Matrix&)
  {
    throw std::logic_error("The coarsening is not supported by the backend");
  }

  template <template <class> class Coarsening, class Matrix>
  std::enable_if_t<backend::coarsening_is_supported<Backend, Coarsening>::value,
                          std::shared_ptr<Matrix>>
  make_coarse(const Matrix& A, const Matrix& P, const Matrix& R) const
  {
    return static_cast<Coarsening<Backend>*>(handle)->coarse_operator(A, P, R);
  }

  template <template <class> class Coarsening, class Matrix>
  std::enable_if_t<!backend::coarsening_is_supported<Backend, Coarsening>::value,
                          std::shared_ptr<Matrix>>
  make_coarse(const Matrix&, const Matrix&, const Matrix&) const
  {
    throw std::logic_error("The coarsening is not supported by the backend");
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
