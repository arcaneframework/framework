// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdAVX.h                                                   (C) 2000-2016 */
/*                                                                           */
/* Vectorisation pour AVX et AVX2.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SIMDAVX_H
#define ARCANE_UTILS_SIMDAVX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Ce fichier ne doit pas être inclus directement.
 * Utiliser 'Simd.h' à la place.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! A définir si on souhaite utiliser le gather de l'AVX2.
// #define ARCANE_USE_AVX2_GATHER

// Le gather n'est disponible que avec l'AVX2
#ifndef __AVX2__
#undef ARCANE_USE_AVX2_GATHER
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Vectorisation des entiers Int32 en utilisant AVX.
 */
class ARCANE_ALIGNAS_PACKED(32) AVXSimdX8Int32
{
 public:
  static const int BLOCK_SIZE = 8;
  enum
  {
    Length = 8,
    Alignment = 32
  };
 public:
  __m256i v0;
  AVXSimdX8Int32(){}
  AVXSimdX8Int32(__m256i _v0) : v0(_v0) {}
  explicit AVXSimdX8Int32(Int32 a) : v0(_mm256_set1_epi32(a)){}
 private:
  AVXSimdX8Int32(Int32 a7,Int32 a6,Int32 a5,Int32 a4,Int32 a3,Int32 a2,Int32 a1,Int32 a0)
  : v0(_mm256_set_epi32(a7,a6,a5,a4,a3,a2,a1,a0)){}
 public:
  AVXSimdX8Int32(const Int32* base,const Int32* idx)
  : v0(_mm256_set_epi32(base[idx[7]],base[idx[6]],base[idx[5]],base[idx[4]],
                        base[idx[3]],base[idx[2]],base[idx[1]],base[idx[0]])) {}
  explicit AVXSimdX8Int32(const Int32* base)
  : v0(_mm256_load_si256((const __m256i*)base)){}

  Int32 operator[](Integer i) const { return ((const Int32*)&v0)[i]; }
  Int32& operator[](Integer i) { return ((Int32*)&v0)[i]; }

  void set(ARCANE_RESTRICT Int32* base,const ARCANE_RESTRICT Int32* idx) const
  {
    const Int32* x = (const Int32*)(this);
    base[idx[0]] = x[0];
    base[idx[1]] = x[1];
    base[idx[2]] = x[2];
    base[idx[3]] = x[3];
    base[idx[4]] = x[4];
    base[idx[5]] = x[5];
    base[idx[6]] = x[6];
    base[idx[7]] = x[7];
  }

  void set(ARCANE_RESTRICT Int32* base) const
  {
    _mm256_store_si256((__m256i*)base,v0);
  }

  void load(const AVXSimdX8Int32* base)
  {
    v0 = _mm256_load_si256((const __m256i*)base);
  }

  static AVXSimdX8Int32 fromScalar(Int32 a0,Int32 a1,Int32 a2,Int32 a3,
                                   Int32 a4,Int32 a5,Int32 a6,Int32 a7)
  {
    return AVXSimdX8Int32(a7,a6,a5,a4,a3,a2,a1,a0);
  }

 private:
  void operator=(Int32 _v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Vectorisation des réels en utilisant AVX.
 * \note Cette classe doit être alignée sur 32 octets.
 */
class ARCANE_ALIGNAS_PACKED(32) AVXSimdX4Real
{
 public:
  static const int BLOCK_SIZE = 4;
  enum
  {
    Length = 4
  };
  typedef SSESimdX4Int32 Int32IndexType;
 public:
  __m256d v0;
  AVXSimdX4Real(){}
  AVXSimdX4Real(__m256d _v0)
  : v0(_v0) {}
  explicit AVXSimdX4Real(Real r)
  : v0(_mm256_set1_pd(r)) { }
 private:
  AVXSimdX4Real(Real a3,Real a2,Real a1,Real a0)
  : v0(_mm256_set_pd(a3,a2,a1,a0)) { }
 public:
  AVXSimdX4Real(const Real* base,const Int32* idx)
  : v0(_mm256_set_pd(base[idx[3]],base[idx[2]],base[idx[1]],base[idx[0]])) {}

  AVXSimdX4Real(const Real* base,const Int32IndexType& simd_idx)
#ifdef ARCANE_USE_AVX2_GATHER
  : v0(_mm256_i32gather_pd(base,simd_idx.v0,8)){}
#else
  : AVXSimdX4Real(base,(const Int32*)&simd_idx){}
#endif

  AVXSimdX4Real(const Real* base,const Int32IndexType* simd_idx)
#ifdef ARCANE_USE_AVX2_GATHER
  : v0(_mm256_i32gather_pd((Real*)base,simd_idx->v0,8)){}
#else
  : AVXSimdX4Real(base,(const Int32*)simd_idx){}
#endif

  //! Charge les valeurs continues situées à l'adresse \a base qui doit être alignée.
  explicit AVXSimdX4Real(const Real* base)
  : v0(_mm256_load_pd(base)) { }

  Real operator[](Integer i) const { return ((const Real*)&v0)[i]; }
  Real& operator[](Integer i) { return ((Real*)&v0)[i]; }

  void set(ARCANE_RESTRICT Real* base,const Int32* idx) const
  {
#if 1
    const Real* x = (const Real*)(this);
    base[idx[0]] = x[0];
    base[idx[1]] = x[1];
    base[idx[2]] = x[2];
    base[idx[3]] = x[3];
#else
    // Ces méthodes de scatter ne sont disponibles que
    // pour l'AVX512VL
    __m128i idx0 = _mm_load_si128((__m128i*)idx);
    _mm256_i32scatter_pd(base,idx0,v0, 8);
#endif
  }

  void set(ARCANE_RESTRICT Real* base,const Int32IndexType& simd_idx) const
  {
    this->set(base,&simd_idx);
  }

  void set(ARCANE_RESTRICT Real* base,const Int32IndexType* simd_idx) const
  {
    this->set(base,(const Int32*)simd_idx);
  }
  //! Stocke les valeurs de l'instance à l'adresse \a base qui doit être alignée.
  void set(ARCANE_RESTRICT Real* base) const
  {
    _mm256_store_pd(base,v0);
  }

  static AVXSimdX4Real fromScalar(Real a0,Real a1,Real a2,Real a3)
  {
    return AVXSimdX4Real(a3,a2,a1,a0);
  }

  // Unary operation operator-
  inline AVXSimdX4Real operator- () const
  {
    return AVXSimdX4Real(_mm256_sub_pd(_mm256_setzero_pd(),v0));
  }

 private:
  void operator=(Real _v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Vectorisation des réels en utilisant AVX avec des blocs de 8 reels.
 * \note Cette classe doit être alignée sur 32 octets.
 */
class ARCANE_ALIGNAS_PACKED(64) AVXSimdX8Real
{
 public:
  static const int BLOCK_SIZE = 8;
  enum
  {
    Length = 8
  };
 public:
  __m256d v0;
  __m256d v1;
  AVXSimdX8Real(){}
  AVXSimdX8Real(__m256d _v0,__m256d _v1)
  : v0(_v0), v1(_v1){}
  explicit AVXSimdX8Real(Real r)
  {
    v0 = _mm256_set1_pd(r);
    v1 = _mm256_set1_pd(r);
  }
 private:
  AVXSimdX8Real(Real a7,Real a6,Real a5,Real a4,Real a3,Real a2,Real a1,Real a0)
  {
    v0 = _mm256_set_pd(a3,a2,a1,a0);
    v1 = _mm256_set_pd(a7,a6,a5,a4);
  }
 public:
  AVXSimdX8Real(const Real* base,const Int32* idx)
  {
    //TODO Avec AVX2, utiliser vgather mais pour l'instant on ne le détecte pas
    // et les tests montrent que ce n'est pas toujours le plus
    // performant (peut-être avec des indices alignés).
#if 1
    v0 = _mm256_set_pd(base[idx[3]],base[idx[2]],base[idx[1]],base[idx[0]]);
    v1 = _mm256_set_pd(base[idx[7]],base[idx[6]],base[idx[5]],base[idx[4]]);
#else
    __m128i idx0 = _mm_loadu_si128((__m128i*)idx);
    __m128i idx1 = _mm_loadu_si128((__m128i*)(idx+4));
    v0 = _mm256_i32gather_pd((Real*)base,idx0,8);
    v1 = _mm256_i32gather_pd((Real*)base,idx1,8);
#endif
  }

  //! Charge les valeurs continues situées à l'adresse \a base qui doit être alignée.
  explicit AVXSimdX8Real(const Real* base)
  {
    v0 = _mm256_load_pd(base);
    v1 = _mm256_load_pd(base+4);
  }

  Real operator[](Integer i) const { return ((const Real*)&v0)[i]; }
  Real& operator[](Integer i) { return ((Real*)&v0)[i]; }

  void set(ARCANE_RESTRICT Real* base,const ARCANE_RESTRICT Int32* idx) const
  {
#if 1
    const Real* x = (const Real*)(this);
    base[idx[0]] = x[0];
    base[idx[1]] = x[1];
    base[idx[2]] = x[2];
    base[idx[3]] = x[3];
    base[idx[4]] = x[4];
    base[idx[5]] = x[5];
    base[idx[6]] = x[6];
    base[idx[7]] = x[7];
#else
    // Ces méthodes de scatter ne sont disponibles que
    // pour l'AVX512VL
    __m128i idx0 = _mm_loadu_si128((__m128i*)idx);
    __m128i idx1 = _mm_loadu_si128((__m128i*)(idx+4));
    _mm256_i32scatter_pd(base,idx0,v0, 8);
    _mm256_i32scatter_pd(base,idx1,v1, 8);
#endif
  }

  //! Stocke les valeurs de l'instance à l'adresse \a base qui doit être alignée.
  void set(ARCANE_RESTRICT Real* base) const
  {
    _mm256_store_pd(base,v0);
    _mm256_store_pd(base+4,v1);
  }

  static AVXSimdX8Real fromScalar(Real a0,Real a1,Real a2,Real a3,
                                Real a4,Real a5,Real a6,Real a7)
  {
    return AVXSimdX8Real(a7,a6,a5,a4,a3,a2,a1,a0);
  }

  // Unary operation operator-
  inline AVXSimdX8Real operator- () const
  {
    return AVXSimdX8Real(_mm256_sub_pd(_mm256_setzero_pd(),v0),
                         _mm256_sub_pd(_mm256_setzero_pd(),v1));
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
typedef AVXSimdX4Real AVXSimdReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AVXSimdInfo
{
 public:
  static const char* name() { return "AVX"; }
  enum
    {
      Int32IndexSize = AVXSimdReal::Length
    };
  typedef AVXSimdReal SimdReal;
  typedef AVXSimdReal::Int32IndexType SimdInt32IndexType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_UTILS_EXPORT std::ostream& operator<<(std::ostream& o,const AVXSimdReal& s);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
