// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <boost/multi_index_container.hpp>
#include <boost/numeric/ublas/lu.hpp>

// LU factorization with partial pivoting

namespace Numerics {

  using namespace boost::numeric::ublas;

  template<class M, class PM>
  typename M::size_type lu_factorize (M &m,
                                      PM &pm,
                                      const typename M::value_type &eps = typename M::value_type()/*zero*/) {
    typedef M matrix_type;
    typedef typename M::size_type size_type;
    typedef typename M::value_type value_type;

#if BOOST_UBLAS_TYPE_CHECK
    matrix_type cm (m);
#endif
    int singular = 0;
    size_type size1 = m.size1 ();
    size_type size2 = m.size2 ();
    size_type size = (std::min) (size1, size2);
    value_type max_pivot = typename M::value_type()/*zero*/;
    for (size_type i = 0; i < size; ++ i) {
      matrix_column<M> mci (column (m, i));
      matrix_row<M> mri (row (m, i));
      size_type i_norm_inf = i + index_norm_inf (project (mci, range (i, size1)));
      BOOST_UBLAS_CHECK (i_norm_inf < size1, external_logic ());
      max_pivot = std::max (max_pivot, std::fabs (m (i_norm_inf, i)));
      if (std::fabs (m (i_norm_inf, i)) > max_pivot*eps &&
          m (i_norm_inf, i) != value_type/*zero*/()) {
        if (i_norm_inf != i) {
          pm (i) = i_norm_inf;
          row (m, i_norm_inf).swap (mri);
        } else {
          BOOST_UBLAS_CHECK (pm (i) == i_norm_inf, external_logic ());
        }
        project (mci, range (i + 1, size1)) *= value_type (1) / m (i, i);
      } else if (singular == 0) {
        singular = i + 1;
      }
      project (m, range (i + 1, size1), range (i + 1, size2)).minus_assign (
                                                                            outer_prod (project (mci, range (i + 1, size1)),
                                                                                        project (mri, range (i + 1, size2))));
    }
#if BOOST_UBLAS_TYPE_CHECK
    swap_rows (pm, cm);
    BOOST_UBLAS_CHECK (singular != 0 ||
                       boost::numeric::ublas::detail::expression_type_check (prod (triangular_adaptor<matrix_type, unit_lower> (m),
                                                            triangular_adaptor<matrix_type, upper> (m)), cm), internal_logic ());
#endif
    return singular;
  }

}

#endif
