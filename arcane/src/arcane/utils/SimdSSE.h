// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdSSE.h                                                   (C) 2000-2016 */
/*                                                                           */
/* Vectorisation pour le SSE.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SIMDSSE_H
#define ARCANE_UTILS_SIMDSSE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Ce fichier ne doit pas être inclus directement.
 * Utiliser 'Simd.h' à la place.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Vectorisation des entiers en utilisant SSE.
 * \todo Normalement il faudrait mettre l'alignement sur 32 octets mais
 * cela rend sur CentOS 6 les compilations entre gcc 4.4 (le défaut sur
 * CentOS 6) et gcc 4.6+ incompatibles.
 */
class ARCANE_ALIGNAS_PACKED(16) SSESimdX4Int32
{
 public:
  static const int BLOCK_SIZE = 4;
  enum
  {
    Length = 4,
    Alignment = 16
  };
 public:
  __m128i v0;
  SSESimdX4Int32(){}
  SSESimdX4Int32(__m128i _v0) : v0(_v0) {}
  explicit SSESimdX4Int32(Int32 a) : v0(_mm_set1_epi32(a)){}
 private:
  SSESimdX4Int32(Int32 a3,Int32 a2,Int32 a1,Int32 a0)
  : v0(_mm_set_epi32(a3,a2,a1,a0)){}
 public:
  SSESimdX4Int32(const Int32* base,const Int32* idx)
  : v0(_mm_set_epi32(base[idx[3]],base[idx[2]],base[idx[1]],base[idx[0]])) {}
  // TODO: faire la version non alignée
  explicit SSESimdX4Int32(const Int32* base)
  : v0(_mm_load_si128((const __m128i*)base)){}

  Int32 operator[](Integer i) const { return ((const Int32*)&v0)[i]; }
  Int32& operator[](Integer i) { return ((Int32*)&v0)[i]; }

  void set(ARCANE_RESTRICT Int32* base,const ARCANE_RESTRICT Int32* idx) const
  {
    const Int32* x = (const Int32*)(this);
    base[idx[0]] = x[0];
    base[idx[1]] = x[1];
    base[idx[2]] = x[2];
    base[idx[3]] = x[3];
  }

  void set(ARCANE_RESTRICT Int32* base) const
  {
    // TODO: faire la version non alignée
    _mm_store_si128((__m128i*)base,v0);
  }

  static SSESimdX4Int32 fromScalar(Int32 a0,Int32 a1,Int32 a2,Int32 a3)
  {
    return SSESimdX4Int32(a3,a2,a1,a0);
  }

 private:
  void operator=(Int32 _v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Vectorisation des réels en utilisant SSE.
 * \todo Normalement il faudrait mettre l'alignement sur 32 octets mais
 * cela rend sur CentOS 6 les compilations entre gcc 4.4 (le défaut sur
 * CentOS 6) et gcc 4.6+ incompatibles.
 */
class ARCANE_ALIGNAS_PACKED(16) SSESimdX2Real
{
 public:
  static const int BLOCK_SIZE = 2;
  enum
  {
    Length = 2
  };
  typedef EMULSimdX2Int32 Int32IndexType;
 public:
  __m128d v0;
  SSESimdX2Real(){}
  SSESimdX2Real(__m128d _v0)
  : v0(_v0){}
  explicit SSESimdX2Real(Real r)
  : v0(_mm_set1_pd(r)){}
 private:
  SSESimdX2Real(Real a1,Real a0)
  : v0(_mm_set_pd(a1,a0)){}
 public:
  SSESimdX2Real(const Real* base,const Int32* idx)
  : v0(_mm_set_pd(base[idx[1]],base[idx[0]])) { }
  SSESimdX2Real(const Real* base,const Int32IndexType* simd_idx)
  {
    const Int32* idx = (const Int32*)simd_idx;
    v0 = _mm_set_pd(base[idx[1]],base[idx[0]]);
  }
  SSESimdX2Real(const Real* base,const Int32IndexType& simd_idx)
  {
    const Int32* idx = (const Int32*)&simd_idx;
    v0 = _mm_set_pd(base[idx[1]],base[idx[0]]);
  }
  SSESimdX2Real(const Real* base)
  {
    v0 = _mm_load_pd(base);
  }

  Real operator[](Integer i) const { return ((const Real*)&v0)[i]; }
  Real& operator[](Integer i) { return ((Real*)&v0)[i]; }

  void set(ARCANE_RESTRICT Real* base,const ARCANE_RESTRICT Int32* idx) const
  {
    const Real* x = (const Real*)(this);
    base[idx[0]] = x[0];
    base[idx[1]] = x[1];
  }

  void set(ARCANE_RESTRICT Real* base,const ARCANE_RESTRICT Int32IndexType& simd_idx) const
  {
    this->set(base,&simd_idx);
  }

  void set(ARCANE_RESTRICT Real* base,const ARCANE_RESTRICT Int32IndexType* simd_idx) const
  {
    const Int32* idx = (const ARCANE_RESTRICT Int32*)simd_idx;
    const Real* x = (const Real*)(this);
    base[idx[0]] = x[0];
    base[idx[1]] = x[1];
  }

  void set(ARCANE_RESTRICT Real* base) const
  {
    _mm_store_pd(base,v0);
  }

  static SSESimdX2Real fromScalar(Real a0,Real a1)
  {
    return SSESimdX2Real(a1,a0);
  }

  // Unary operation operator-
  inline SSESimdX2Real operator- () const
  {
    return SSESimdX2Real(_mm_sub_pd(_mm_setzero_pd(),v0));
  }

 private:
  void operator=(Real _v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Vectorisation des réels en utilisant SSE.
 * \todo Normalement il faudrait mettre l'alignement sur 32 octets mais
 * cela rend sur CentOS 6 les compilations entre gcc 4.4 (le défaut sur
 * CentOS 6) et gcc 4.6+ incompatibles.
 */
class ARCANE_ALIGNAS_PACKED(16) SSESimdX4Real
{
 public:
  static const int BLOCK_SIZE = 4;
  enum
  {
    Length = 4
  };
  // NOTE: utiliser EMULSimd au lieu de SSE est beaucoup plus performant
  // avec gcc 4.9 et gcc 6.1. Avec Intel 16, c'est le contraire mais la
  // différence n'est pas énorme.
  // typedef SSESimdX4Int32 Int32IndexType;
  typedef EMULSimdX4Int32 Int32IndexType;
 public:
  __m128d v0;
  __m128d v1;
  SSESimdX4Real(){}
  SSESimdX4Real(__m128d _v0,__m128d _v1)
  : v0(_v0), v1(_v1) {}
  explicit SSESimdX4Real(Real r)
  : v0(_mm_set1_pd(r)), v1(_mm_set1_pd(r)){}
 private:
  SSESimdX4Real(Real a3,Real a2,Real a1,Real a0)
  : v0(_mm_set_pd(a1,a0)), v1(_mm_set_pd(a3,a2)){}
 public:
  SSESimdX4Real(const Real* base,const Int32* idx)
  : v0(_mm_set_pd(base[idx[1]],base[idx[0]]))
  , v1(_mm_set_pd(base[idx[3]],base[idx[2]])){}

  SSESimdX4Real(const Real* base,const Int32IndexType* simd_idx)
  : SSESimdX4Real(base,(const Int32*)simd_idx) {}

  SSESimdX4Real(const Real* base,const Int32IndexType& simd_idx)
  : SSESimdX4Real(base,(const Int32*)&simd_idx) {}

  SSESimdX4Real(const Real* base)
  : v0(_mm_load_pd(base)), v1(_mm_load_pd(base+2)) {}

  Real operator[](Integer i) const { return ((const Real*)&v0)[i]; }
  Real& operator[](Integer i) { return ((Real*)&v0)[i]; }

  void set(ARCANE_RESTRICT Real* base,const ARCANE_RESTRICT Int32* idx) const
  {
    const Real* x = (const Real*)(this);
    base[idx[0]] = x[0];
    base[idx[1]] = x[1];
    base[idx[2]] = x[2];
    base[idx[3]] = x[3];
  }

  void set(ARCANE_RESTRICT Real* base,const ARCANE_RESTRICT Int32IndexType& simd_idx) const
  {
    this->set(base,(const Int32*)&simd_idx);
  }

  void set(ARCANE_RESTRICT Real* base,const ARCANE_RESTRICT Int32IndexType* simd_idx) const
  {
    this->set(base,(const Int32*)simd_idx);
  }

  void set(ARCANE_RESTRICT Real* base) const
  {
    _mm_store_pd(base,v0);
    _mm_store_pd(base+2,v1);
  }

  static SSESimdX4Real fromScalar(Real a0,Real a1,Real a2,Real a3)
  {
    return SSESimdX4Real(a3,a2,a1,a0);
  }

  // Unary operation operator-
  inline SSESimdX4Real operator- () const
  {
    return SSESimdX4Real(_mm_sub_pd(_mm_setzero_pd(),v0),
                         _mm_sub_pd(_mm_setzero_pd(),v1));
  }
 private:
  void operator=(Real _v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Vecteur de 8 doubles avec implémentation SSE.
 */
class ARCANE_ALIGNAS_PACKED(64) SSESimdX8Real
{
 public:
  static const int BLOCK_SIZE = 8;
  enum
  {
    Length = 8
  };
 public:
  __m128d v0;
  __m128d v1;
  __m128d v2;
  __m128d v3;
  SSESimdX8Real(){}
  SSESimdX8Real(__m128d _v0,__m128d _v1,__m128d _v2,__m128d _v3)
  : v0(_v0), v1(_v1), v2(_v2), v3(_v3) {}
  explicit SSESimdX8Real(Real r)
  : v0(_mm_set1_pd(r)), v1(_mm_set1_pd(r)), v2(_mm_set1_pd(r)), v3(_mm_set1_pd(r)){}
 private:
  SSESimdX8Real(Real a7,Real a6,Real a5,Real a4,Real a3,Real a2,Real a1,Real a0)
  : v0(_mm_set_pd(a1,a0)), v1(_mm_set_pd(a3,a2)),
    v2(_mm_set_pd(a5,a4)), v3(_mm_set_pd(a7,a6)){}
 public:
  SSESimdX8Real(const Real* base,const Int32* idx)
  {
    v0 = _mm_set_pd(base[idx[1]],base[idx[0]]);
    v1 = _mm_set_pd(base[idx[3]],base[idx[2]]);
    v2 = _mm_set_pd(base[idx[5]],base[idx[4]]);
    v3 = _mm_set_pd(base[idx[7]],base[idx[6]]);
  }

  Real operator[](Integer i) const { return ((const Real*)&v0)[i]; }
  Real& operator[](Integer i) { return ((Real*)&v0)[i]; }

  void set(ARCANE_RESTRICT Real* base,const ARCANE_RESTRICT Int32* idx) const
  {
    const Real* x = (const Real*)(this);
    base[idx[0]] = x[0];
    base[idx[1]] = x[1];
    base[idx[2]] = x[2];
    base[idx[3]] = x[3];
    base[idx[4]] = x[4];
    base[idx[5]] = x[5];
    base[idx[6]] = x[6];
    base[idx[7]] = x[7];
  }

  static SSESimdX8Real fromScalar(Real a0,Real a1,Real a2,Real a3,Real a4,Real a5,Real a6,Real a7)
  {
    return SSESimdX8Real(a7,a6,a5,a4,a3,a2,a1,a0);
  }

  // Unary operation operator-
  inline SSESimdX8Real operator- () const
  {
    return SSESimdX8Real(_mm_sub_pd(_mm_setzero_pd(),v0),
                         _mm_sub_pd(_mm_setzero_pd(),v1),
                         _mm_sub_pd(_mm_setzero_pd(),v2),
                         _mm_sub_pd(_mm_setzero_pd(),v3));
  }
 private:
  void operator=(Real _v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vecteur de 'double' en implémentation par SSE.
 *
 * Utilise le vecteur de 4 éléments comme vecteur par défaut en SSE.
 * Les différents tests montrent que c'est la meilleur taille. Avec une
 * taille de deux les boucles sont trop petites et avec une taille de 8
 * le compilateur a souvent trop de temporaires à gérer ce qui limite
 * l'optimisation.
 */
typedef SSESimdX4Real SSESimdReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SSESimdInfo
{
 public:
  static const char* name() { return "SSE"; }
  enum
    {
      Int32IndexSize = SSESimdReal::Length
    };
  typedef SSESimdReal SimdReal;
  typedef SSESimdReal::Int32IndexType SimdInt32IndexType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_UTILS_EXPORT std::ostream&
operator<<(std::ostream& o,const SSESimdReal& s);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
