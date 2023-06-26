/*

Copyright 2020 IFPEN-CEA

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

/*!
 * \file LinearAlgebraExprT.h
 * \brief LinearAlgebraExprT.h
 */

#pragma once

#include <alien/core/backend/LinearAlgebraExpr.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/data/IMatrix.h>
#include <alien/data/IVector.h>
#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
LinearAlgebraExpr<Tag, TagV>::~LinearAlgebraExpr() {}

/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
Arccore::Real
LinearAlgebraExpr<Tag, TagV>::norm0(const IVector& x) const
{
  const auto& vx = x.impl()->get<TagV>();
  return m_algebra->norm0(vx);
}

/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
Arccore::Real
LinearAlgebraExpr<Tag, TagV>::norm1(const IVector& x) const
{
  const auto& vx = x.impl()->get<TagV>();
  return m_algebra->norm1(vx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
Arccore::Real
LinearAlgebraExpr<Tag, TagV>::norm2(const IVector& x) const
{
  const auto& vx = x.impl()->get<TagV>();
  return m_algebra->norm2(vx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
Arccore::Real
LinearAlgebraExpr<Tag, TagV>::norm2(const IMatrix& x) const
{
  const auto& mx = x.impl()->get<Tag>();
  return m_algebra->norm2(mx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::mult(const IMatrix& a, const IVector& x, IVector& r) const
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
void LinearAlgebraExpr<Tag, TagV>::axpy(Real alpha, const IVector& x, IVector& r) const
{
  const auto& vx = x.impl()->get<TagV>();
  auto& vr = r.impl()->get<TagV>(true);
  ALIEN_ASSERT((vx.space() == vr.space()), ("Incompatible spaces"));
  m_algebra->axpy(alpha, vx, vr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::aypx(Real alpha, IVector& y, const IVector& x) const
{
  const auto& vx = x.impl()->get<TagV>();
  auto& vy = y.impl()->get<TagV>(true);
  ALIEN_ASSERT((vx.space() == vy.space()), ("Incompatible spaces"));
  m_algebra->aypx(alpha, vy, vx);
}

/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::copy(const IVector& x, IVector& r) const
{
  const auto& vx = x.impl()->get<TagV>();
  auto& vr = r.impl()->get<TagV>(true);
  ALIEN_ASSERT((vx.space() == vr.space()), ("Incompatible spaces"));
  m_algebra->copy(vx, vr);
}

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::copy(const IMatrix& x, IMatrix& r) const
{
  const auto& mx = x.impl()->get<TagV>();
  auto& mr = r.impl()->get<TagV>(true);
  ALIEN_ASSERT((mx.rowSpace() == mr.rowSpace()), ("Incompatible spaces"));
  m_algebra->copy(mx, mr);
}

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::add(const IMatrix& a, IMatrix& b) const
{
  const auto& ma = a.impl()->get<Tag>();
  auto& mb = b.impl()->get<Tag>(true);
  ALIEN_ASSERT((ma.rowSpace() == mb.rowSpace()), ("Incompatible spaces"));
  m_algebra->add(ma, mb);
}

/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
Real LinearAlgebraExpr<Tag, TagV>::dot(const IVector& x, const IVector& y) const
{
  const auto& vx = x.impl()->get<TagV>();
  const auto& vy = y.impl()->get<TagV>();
  ALIEN_ASSERT((vx.space() == vy.space()), ("Incompatible space"));
  return m_algebra->dot(vx, vy);
}

/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::diagonal(const IMatrix& a, IVector& x) const
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
void LinearAlgebraExpr<Tag, TagV>::reciprocal(IVector& x) const
{
  auto& vx = x.impl()->get<TagV>(true);
  m_algebra->reciprocal(vx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::scal(Real alpha, IVector& x) const
{
  auto& vx = x.impl()->get<TagV>(true);
  m_algebra->scal(alpha, vx);
}

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::scal(Real alpha, IMatrix& A) const
{
  auto& ma = A.impl()->get<Tag>(true);
  m_algebra->scal(alpha, ma);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::pointwiseMult(
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

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::mult(
const IMatrix& a, const UniqueArray<Real>& x, UniqueArray<Real>& r) const
{
  auto const& ma = a.impl()->get<Tag>();
  m_algebra->mult(ma, x, r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::axpy(Real alpha, const UniqueArray<Real>& x, UniqueArray<Real>& r) const
{
  m_algebra->axpy(alpha, x, r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::aypx(Real alpha, UniqueArray<Real>& y, UniqueArray<Real> const& x) const
{
  m_algebra->aypx(alpha, y, x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::copy(
const Alien::UniqueArray<Real>& x, Alien::UniqueArray<Real>& r) const
{
  m_algebra->copy(x, r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
Real LinearAlgebraExpr<Tag, TagV>::dot(
Integer local_size, const UniqueArray<Real>& x, const UniqueArray<Real>& y) const
{
  return m_algebra->dot(local_size, x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::scal(Real alpha, Alien::UniqueArray<Real>& x) const
{
  m_algebra->scal(alpha, x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::dump(const IMatrix& a, std::string const& filename) const
{
  auto const& ma = a.impl()->get<Tag>();
  m_algebra->dump(ma, filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag, class TagV>
void LinearAlgebraExpr<Tag, TagV>::dump(const IVector& x, std::string const& filename) const
{
  auto const& vx = x.impl()->get<TagV>();
  m_algebra->dump(vx, filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
