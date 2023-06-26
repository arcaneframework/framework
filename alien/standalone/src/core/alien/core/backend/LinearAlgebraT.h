/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file LinearAlgebraT.h
 * \brief LinearAlgebraT.h
 */

#pragma once

#include <string>

#include <alien/core/backend/LinearAlgebra.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/data/IMatrix.h>
#include <alien/data/IVector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
LinearAlgebra<Tag, TagV>::~LinearAlgebra() {}

/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
Arccore::Real
LinearAlgebra<Tag, TagV>::norm0(const IVector& x) const
{
  const auto& vx = x.impl()->get<TagV>();
  return m_algebra->norm0(vx);
}

/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
Arccore::Real
LinearAlgebra<Tag, TagV>::norm1(const IVector& x) const
{
  const auto& vx = x.impl()->get<TagV>();
  return m_algebra->norm1(vx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
Arccore::Real
LinearAlgebra<Tag, TagV>::norm2(const IVector& x) const
{
  const auto& vx = x.impl()->get<TagV>();
  return m_algebra->norm2(vx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebra<Tag, TagV>::mult(const IMatrix& a, const IVector& x, IVector& r) const
{
  const auto& ma = a.impl()->get<Tag>();
  const auto& vx = x.impl()->get<TagV>();
  auto& vr = r.impl()->get<TagV>(true);
  ALIEN_ASSERT((ma.colSpace() == vx.space() && ma.rowSpace() == vr.space()),
               ("Incompatible spaces"));
  m_algebra->mult(ma, vx, vr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebra<Tag, TagV>::axpy(Real alpha, const IVector& x, IVector& r) const
{
  const auto& vx = x.impl()->get<TagV>();
  auto& vr = r.impl()->get<TagV>(true);
  ALIEN_ASSERT((vx.space() == vr.space()), ("Incompatible spaces"));
  m_algebra->axpy(alpha, vx, vr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebra<Tag, TagV>::aypx(Real alpha, IVector& y, const IVector& x) const
{
  const auto& vx = x.impl()->get<TagV>();
  auto& vy = y.impl()->get<TagV>(true);
  ALIEN_ASSERT((vx.space() == vy.space()), ("Incompatible spaces"));
  m_algebra->aypx(alpha, vy, vx);
}

/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebra<Tag, TagV>::copy(const IVector& x, IVector& r) const
{
  const auto& vx = x.impl()->get<TagV>();
  auto& vr = r.impl()->get<TagV>(true);
  ALIEN_ASSERT((vx.space() == vr.space()), ("Incompatible spaces"));
  m_algebra->copy(vx, vr);
}

/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
Real LinearAlgebra<Tag, TagV>::dot(const IVector& x, const IVector& y) const
{
  const auto& vx = x.impl()->get<TagV>();
  const auto& vy = y.impl()->get<TagV>();
  ALIEN_ASSERT((vx.space() == vy.space()), ("Incompatible space"));
  return m_algebra->dot(vx, vy);
}

/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebra<Tag, TagV>::diagonal(const IMatrix& a, IVector& x) const
{
  auto& vx = x.impl()->get<TagV>(true);
  const auto& ma = a.impl()->get<Tag>();
  ALIEN_ASSERT((ma.rowSpace() == ma.colSpace()), ("Matrix not square"));
  ALIEN_ASSERT((ma.rowSpace() == vx.space()), ("Incompatible space"));
  m_algebra->diagonal(ma, vx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebra<Tag, TagV>::reciprocal(IVector& x) const
{
  auto& vx = x.impl()->get<TagV>(true);
  m_algebra->reciprocal(vx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebra<Tag, TagV>::scal(Real alpha, IVector& x) const
{
  auto& vx = x.impl()->get<TagV>(true);
  m_algebra->scal(alpha, vx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebra<Tag, TagV>::pointwiseMult(
const IVector& x, const IVector& y, IVector& w) const
{
  auto& vw = w.impl()->get<TagV>(true);
  const auto& vx = x.impl()->get<TagV>();
  const auto& vy = y.impl()->get<TagV>();
  ALIEN_ASSERT((vy.space() == vx.space()), ("Incompatible space"));
  ALIEN_ASSERT((vw.space() == vx.space()), ("Incompatible space"));
  m_algebra->pointwiseMult(vx, vy, vw);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebra<Tag, TagV>::dump(const IMatrix& a, std::string const& filename) const
{
  auto const& ma = a.impl()->get<Tag>();
  m_algebra->dump(ma, filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebra<Tag, TagV>::dump(const IVector& x, std::string const& filename) const
{
  auto const& vx = x.impl()->get<TagV>();
  m_algebra->dump(vx, filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
