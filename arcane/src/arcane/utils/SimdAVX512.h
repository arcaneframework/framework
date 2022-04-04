// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdAVX512.h                                                (C) 2000-2016 */
/*                                                                           */
/* Vectorisation pour l'AVX512.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SIMDAVX512_H
#define ARCANE_UTILS_SIMDAVX512_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Ce fichier ne doit pas être inclus directement.
 * Utiliser 'Simd.h' à la place.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// A définir si on souhaite utiliser directement les instructions
// de scatter/gather de l'AVX512.
// Il n'y a priori aucune raison de ne pas le faire.
#define ARCANE_USE_AVX512_SCATTERGATHER

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Vectorisation des réels en utilisant la vectorisation du AVX512.
 */
class ARCANE_ALIGNAS_PACKED(64) AVX512SimdReal
{
 public:
  static const int BLOCK_SIZE = 8;
  enum
    {
      Length = 8
    };
  typedef AVXSimdX8Int32 Int32IndexType;
 public:
  __m512d v0;
  AVX512SimdReal(){}
  AVX512SimdReal(__m512d _v0) : v0(_v0){}
  explicit AVX512SimdReal(Real r)
  : v0(_mm512_set1_pd(r)){}
 protected:
  AVX512SimdReal(Real a7,Real a6,Real a5,Real a4,Real a3,Real a2,Real a1,Real a0)
  : v0(_mm512_set_pd(a7,a6,a5,a4,a3,a2,a1,a0)){}
 public:
  AVX512SimdReal(const Real* base,const Int32* idx)
  {
#ifdef ARCANE_USE_AVX512_SCATTERGATHER
    __m256i idx2 = _mm256_loadu_si256((__m256i*)idx);
    v0 = _mm512_i32gather_pd(idx2,(Real*)base, 8);
#else
    v0 = _mm512_set_pd(base[idx[7]],base[idx[6]],base[idx[5]],base[idx[4]],
                       base[idx[3]],base[idx[2]],base[idx[1]],base[idx[0]]);
#endif
  }
  AVX512SimdReal(const Real* base,const Int32IndexType* simd_idx)
  : AVX512SimdReal(base,(const Int32*)simd_idx) { }

  AVX512SimdReal(const Real* base,const Int32IndexType& simd_idx)
#ifdef ARCANE_USE_AVX512_SCATTERGATHER
  : v0(_mm512_i32gather_pd(simd_idx.v0,base,8))
#else
  : AVX512SimdReal(base,(const Int32*)simd_idx)
#endif
  {
  }

  //! Charge les valeurs continues situées à l'adresse \a base qui doit être alignée.
  explicit AVX512SimdReal(const Real* base)
  : v0(_mm512_load_pd(base)) { }

  Real operator[](Integer i) const { return ((const Real*)&v0)[i]; }
  Real& operator[](Integer i) { return ((Real*)&v0)[i]; }

  // TODO: faire une surcharge qui prend directement un vecteur de Int32 avec alignement.
  void set(ARCANE_RESTRICT Real* base,const ARCANE_RESTRICT Int32* idx) const
  {
#ifdef ARCANE_USE_AVX512_SCATTERGATHER
    __m256i idx2 = _mm256_loadu_si256((__m256i*)idx);
    _mm512_i32scatter_pd(base,idx2,v0, 8);
#else
    const Real* x = (const Real*)(this);
    base[idx[0]] = x[0];
    base[idx[1]] = x[1];
    base[idx[2]] = x[2];
    base[idx[3]] = x[3];
    base[idx[4]] = x[4];
    base[idx[5]] = x[5];
    base[idx[6]] = x[6];
    base[idx[7]] = x[7];
#endif
  }

  void set(ARCANE_RESTRICT Real* base,const Int32IndexType* simd_idx) const
  {
    this->set(base,(const Int32*)simd_idx);
  }

  void set(ARCANE_RESTRICT Real* base,const Int32IndexType& simd_idx) const
  {
#ifdef ARCANE_USE_AVX512_SCATTERGATHER
    _mm512_i32scatter_pd(base,simd_idx.v0,v0, 8);
#else
    this->set(base,&simd_idx);
#endif
  }
  
  //! Stocke les valeurs de l'instance à l'adresse \a base qui doit être alignée.
  void set(ARCANE_RESTRICT Real* base) const
  {
    _mm512_store_pd(base,v0);
  }

  static AVX512SimdReal fromScalar(Real a0,Real a1,Real a2,Real a3,Real a4,Real a5,Real a6,Real a7)
  {
    return AVX512SimdReal(a7,a6,a5,a4,a3,a2,a1,a0);
  }

  // Unary operation operator-
  inline AVX512SimdReal operator- () const
  {
    return AVX512SimdReal(_mm512_sub_pd(_mm512_setzero_pd(),v0));
  }

 private:
  void operator=(Real _v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AVX512SimdInfo
{
 public:
  static const char* name() { return "AVX512"; }
  enum
    {
      Int32IndexSize = 8
    };
  typedef AVX512SimdReal SimdReal;
  typedef AVX512SimdReal::Int32IndexType SimdInt32IndexType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_UTILS_EXPORT std::ostream& operator<<(std::ostream& o,const AVX512SimdReal& s);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
