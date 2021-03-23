// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Simd.h                                                      (C) 2000-2017 */
/*                                                                           */
/* Types pour la vectorisation.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SIMD_H
#define ARCANE_UTILS_SIMD_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/SimdCommon.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file Simd.h
 *
 * Ce fichier contient les déclarations des types utilisés pour gérer
 * la vectorisation. Comme il existe plusieurs mécanismes possibles, il
 * faut faire un choix en fonction de l'architecture machine et des
 * options de compilation. Chaque mécanisme utilise des vecteurs
 * de taille différente. Dans notre cas comme la vectorisation est
 * utilisée principalement pour les calculs sur des doubles, la
 * taille d'un vecteur sera égale au nombre de double qu'on peut mettre
 * dans un vecteur.
 *
 * Actuellement, on supporte les modes suivants par ordre de priorité. Si
 * un mode est supporté, les autres ne sont pas utilisés.
 * - AVX512 pour les architectures de type Intel Knight Landing (KNL) ou
 * Xeon Skylake. La taille des vecteurs est de 8 dans ce mode.
 * - AVX. Pour ce mode, il faut compiler Arcane avec l'option '--with-avx'. Il
 * existe deux modes, l'AVX classique et l'AVX2. Pour l'instant, seul le
 * premier est utilisé, faute de machines pour tester le second. La taille
 * des vecteurs est de 4 dans ce mode.
 * - SSE. Ce mode est disponible par défaut car il existe sur toutes les
 * plateformes x64. La aussi il existe plusieurs versions et on se limite
 * à la version 2. La taille des vecteurs est de 2 dans ce mode
 * - aucun mode. Dans ce cas il n'y a pas de vectorisation spécifique.
 * Néanmoins pour tester le code, on permet une émulation avec 
 * des vecteurs de taille de 2.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Les compilateurs (gcc et icc) définissent les macros suivantes
 * sur x64 suivant le support de la vectorisation
 * : avx: __AVX__
 * : avx2: __AVX2__
 * : avx512f: __AVX512F__
 * : sse2: __SSE2__
 *
 * A noter que pour l'avx2 avec gcc, il n'y a pas par défaut le FMA.
 * Par exemple:
 * - gcc -mavx2 : pas de fma
 * - gcc -mavx2 -mfma : fma actif
 * - gcc -march=haswell : fma actif
 */

// Simd via émulation.
#include "arcane/utils/SimdEMUL.h"

// Ajoute support de SSE s'il existe
#if (defined(_M_X64) || defined(__x86_64__)) && !defined(ARCANE_NO_SSE)
// SSE2 est disponible sur tous les CPU x64
// La macro __x64_64__ est définie pour les machines linux
// La macro _M_X64 est définie pour les machines Windows
#define ARCANE_HAS_SSE
#include <emmintrin.h>
#include "arcane/utils/SimdSSE.h"
#endif

// Ajoute support de AVX si dispo
#if defined(ARCANE_HAS_AVX) || defined(ARCANE_HAS_AVX512)
#include <immintrin.h>
#include "arcane/utils/SimdAVX.h"
#endif

// Ajoute support de l'AVX512 si dispo
#if defined(ARCANE_HAS_AVX512)
#include <immintrin.h>
#include "arcane/utils/SimdAVX512.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Macro pour itérer sur les index d'un vecteur
 * Simd de réel ou dérivé (Real2, Real3, ...).
 */
#define ENUMERATE_SIMD_REAL(_iter) \
  for( ::Arcane::Integer _iter(0); _iter < SimdReal ::BLOCK_SIZE; ++ _iter )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Définit le type SimdInfo en fonction de la vectorisation disponible.
 * On prend le type qui permet le plus de vectorisation.
 */
#if defined(ARCANE_HAS_AVX512)
typedef AVX512SimdInfo SimdInfo;
#elif defined(ARCANE_HAS_AVX)
typedef AVXSimdInfo SimdInfo;
#elif defined(ARCANE_HAS_SSE)
typedef SSESimdInfo SimdInfo;
#else
typedef EMULSimdInfo SimdInfo;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Vecteur SIMD de réel.
 */
typedef SimdInfo::SimdReal SimdReal;
const int SimdSize = SimdReal::Length;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Représente un Real3 vectoriel.
 */
class ARCANE_UTILS_EXPORT SimdReal3
{
 public:
  typedef SimdReal::Int32IndexType Int32IndexType;
 public:
  SimdReal x;
  SimdReal y;
  SimdReal z;
  SimdReal3() {}
  SimdReal3(SimdReal _x,SimdReal _y,SimdReal _z) : x(_x), y(_y), z(_z){}
  SimdReal3(const Real3* base,const Int32IndexType& idx)
  {
    for( Integer i=0, n=SimdReal::BLOCK_SIZE; i<n; ++i ){
      Real3 v = base[idx[i]];
      this->set(i,v);
    }
  }
  const Real3 operator[](Integer i) const { return Real3(x[i],y[i],z[i]); }

  void set(Real3* base,const Int32IndexType& idx) const
  {
    for( Integer i=0, n=SimdReal::BLOCK_SIZE; i<n; ++i ){
      base[idx[i]] = this->get(i);
    }
  }

  // TODO: renommer cette méthode
  void set(Integer i,Real3 r)
  {
    x[i] = r.x;
    y[i] = r.y;
    z[i] = r.z;
  }
  Real3 get(Integer i) const
  {
    return Real3(x[i],y[i],z[i]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Représente un Real2 vectoriel.
 */
class ARCANE_UTILS_EXPORT SimdReal2
{
 public:
  typedef SimdReal::Int32IndexType Int32IndexType;
 public:
  SimdReal x;
  SimdReal y;
  SimdReal2() {}
  SimdReal2(SimdReal _x,SimdReal _y) : x(_x), y(_y){}
  SimdReal2(const Real2* base,const Int32IndexType& idx)
  {
    for( Integer i=0, n=SimdReal::BLOCK_SIZE; i<n; ++i ){
      Real2 v = base[idx[i]];
      this->set(i,v);
    }
  }
  const Real2 operator[](Integer i) const { return Real2(x[i],y[i]); }

  void set(Real2* base,const Int32IndexType& idx) const
  {
    for( Integer i=0, n=SimdReal::BLOCK_SIZE; i<n; ++i ){
      base[idx[i]] = this->get(i);
    }
  }

  void set(Integer i,Real2 r)
  {
    x[i] = r.x;
    y[i] = r.y;
  }
  Real2 get(Integer i) const
  {
    return Real2(x[i],y[i]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Charactéristiques des types vectoriels.
 *
 * Instantiation par défaut pour les types n'ayant pas de type vectoriel
 * correspondant. Actuellement, seuls les types 'Real', 'Real2' et 'Real3'
 * en ont un.
 */
template<typename DataType>
class SimdTypeTraits
{
 public:
  typedef void SimdType;
};

template<>
class SimdTypeTraits<Real>
{
 public:
  typedef SimdReal SimdType;
};

template<>
class SimdTypeTraits<Real2>
{
 public:
  typedef SimdReal SimdType;
};

template<>
class SimdTypeTraits<Real3>
{
 public:
  typedef SimdReal3 SimdType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Classe de base des énumérateurs vectoriels avec indirection.
 *
 * \warning Les tableaux des indices locaux (\a local_ids) passés aux
 * constructeurs doivent être alignés.
 */
class ARCANE_UTILS_EXPORT SimdEnumeratorBase
{
 public:

  typedef SimdInfo::SimdInt32IndexType SimdIndexType;

 public:

  SimdEnumeratorBase()
  : m_local_ids(nullptr), m_index(0), m_count(0) { }
  SimdEnumeratorBase(const Int32* local_ids,Integer n)
  : m_local_ids(local_ids), m_index(0), m_count(n)
  { _checkValid(); }
  explicit SimdEnumeratorBase(Int32ConstArrayView local_ids)
  : m_local_ids(local_ids.data()), m_index(0), m_count(local_ids.size())
  { _checkValid(); }

 public:

  bool hasNext() { return m_index<m_count; }

  //! Indices locaux
  const Int32* unguardedLocalIds() const { return m_local_ids; }

  void operator++() { m_index += SimdSize; }

  /*!
   * \brief Nombre de valeurs valides pour l'itérateur courant.
   * \pre hasNext()==true
   */
  inline Integer nbValid() const
  {
    Integer nb_valid = (m_count-m_index);
    if (nb_valid>SimdSize)
      nb_valid = SimdSize;
    return nb_valid;
  }

  Integer count() const { return m_count; }

 protected:

  const Int32* ARCANE_RESTRICT m_local_ids;
  Integer m_index;
  Integer m_count;

  const SimdIndexType* ARCANE_RESTRICT
  _currentSimdIndex() const
  {
    return (const SimdIndexType*)(m_local_ids+m_index);
  }

 private:

  // Vérifie que m_local_ids est correctement aligné.
  void _checkValid()
  {
#ifdef ARCANE_SIMD_BENCH
    Int64 modulo = (Int64)(m_local_ids) % SimdIndexType::Alignment;
    if (modulo!=0){
      throw BadAlignmentException();
    }
#else
    _checkValidHelper();
#endif
  }
  void _checkValidHelper();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
