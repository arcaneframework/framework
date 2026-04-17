// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CuthillMcKeeReorderer h                                     (C) 2000-2026 */
/*                                                                           */
/* (Reverse) Cuthill-McKee matrix reorder algorithm.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_CUTHILLMCKEEREORDERER_H
#define ARCCORE_ALINA_CUTHILLMCKEEREORDERER_H
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
/*
The code is adopted from Kratos project http://www.cimne.com/kratos. The
original code came with the following copyright notice:
\verbatim
Kratos Multi-Physics

Copyright (c) 2012, Pooyan Dadvand, Riccardo Rossi, CIMNE (International Center for Numerical Methods in Engineering)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
    All advertising materials mentioning features or use of this software must
    display the following acknowledgement:
    This product includes Kratos Multi-Physics technology.
    Neither the name of the CIMNE nor the names of its contributors may be used
    to endorse or promote products derived from this software without specific
    prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED ANDON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THISSOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\endverbatim
*/

#include <vector>
#include <algorithm>

#include "arccore/alina/BackendInterface.h"
#include "arccore/alina/AlinaUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief (Reverse) Cuthill-McKee matrix reorder algorithm.
 */
template <bool reverse = false>
struct CuthillMcKeeReorderer
{
  template <class Matrix, class Vector>
  static void get(const Matrix& A, Vector& perm)
  {
    const ptrdiff_t n = backend::nbRow(A);

    /* The data structure used to sort and traverse the level sets:
     *
     * The current level set is currentLevelSet;
     * In this level set, there are nodes with degrees from 0 (not really
     * useful) to maxDegreeInCurrentLevelSet.
     * firstWithDegree[i] points to a node with degree i, or to -1 if it
     * does not exist. nextSameDegree[firstWithDegree[i]] points to the
     * second node with that degree, etc.
     * While the level set is being traversed, the structure for the next
     * level set is generated; nMDICLS will be the next
     * maxDegreeInCurrentLevelSet and nFirstWithDegree will be
     * firstWithDegree.
     */
    ptrdiff_t initialNode = 0; // node to start search
    ptrdiff_t maxDegree = 0;

    std::vector<ptrdiff_t> degree(n);
    std::vector<ptrdiff_t> levelSet(n, 0);
    std::vector<ptrdiff_t> nextSameDegree(n, -1);

#pragma omp parallel
    {
      ptrdiff_t maxd = 0;
#pragma omp for
      for (ptrdiff_t i = 0; i < n; ++i) {
        ptrdiff_t row_width = 0;
        for (auto a = backend::row_begin(A, i); a; ++a, ++row_width)
          ;
        degree[i] = row_width;
        maxd = std::max(maxd, degree[i]);
      }
#pragma omp critical
      {
        maxDegree = std::max(maxDegree, maxd);
      }
    }

    std::vector<ptrdiff_t> firstWithDegree(maxDegree + 1, -1);
    std::vector<ptrdiff_t> nFirstWithDegree(maxDegree + 1);

    // Initialize the first level set, made up by initialNode alone
    perm[0] = initialNode;
    ptrdiff_t currentLevelSet = 1;
    levelSet[initialNode] = currentLevelSet;
    ptrdiff_t maxDegreeInCurrentLevelSet = degree[initialNode];
    firstWithDegree[maxDegreeInCurrentLevelSet] = initialNode;

    // Main loop
    for (ptrdiff_t next = 1; next < n;) {
      ptrdiff_t nMDICLS = 0;
      std::fill(nFirstWithDegree.begin(), nFirstWithDegree.end(), -1);
      bool empty = true; // used to detect different connected components

      ptrdiff_t firstVal = reverse ? maxDegreeInCurrentLevelSet : 0;
      ptrdiff_t finalVal = reverse ? -1 : maxDegreeInCurrentLevelSet + 1;
      ptrdiff_t increment = reverse ? -1 : 1;

      for (ptrdiff_t soughtDegree = firstVal; soughtDegree != finalVal; soughtDegree += increment) {
        ptrdiff_t node = firstWithDegree[soughtDegree];
        while (node > 0) {
          // Visit neighbors
          for (auto a = backend::row_begin(A, node); a; ++a) {
            ptrdiff_t c = a.col();
            if (levelSet[c] == 0) {
              levelSet[c] = currentLevelSet + 1;
              perm[next] = c;
              ++next;
              empty = false; // this level set is not empty
              nextSameDegree[c] = nFirstWithDegree[degree[c]];
              nFirstWithDegree[degree[c]] = c;
              nMDICLS = std::max(nMDICLS, degree[c]);
            }
          }
          node = nextSameDegree[node];
        }
      }

      ++currentLevelSet;
      maxDegreeInCurrentLevelSet = nMDICLS;
      for (ptrdiff_t i = 0; i <= nMDICLS; ++i)
        firstWithDegree[i] = nFirstWithDegree[i];

      if (empty) {
        // The graph contains another connected component that we
        // cannot reach.  Search for a node that has not yet been
        // included in a level set, and start exploring from it.
        bool found = false;
        for (ptrdiff_t i = 0; i < n; ++i) {
          if (levelSet[i] == 0) {
            perm[next] = i;
            ++next;
            levelSet[i] = currentLevelSet;
            maxDegreeInCurrentLevelSet = degree[i];
            firstWithDegree[maxDegreeInCurrentLevelSet] = i;
            found = true;
            break;
          }
        }
        precondition(found, "Internal consistency error at skyline_lu");
      }
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
