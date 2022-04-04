// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdEmulated.h                                              (C) 2000-2016 */
/*                                                                           */
/* Emulation de la vectorisation lorsque aucune mécanisme n'est disponible.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SIMDEMULATED_H
#define ARCANE_UTILS_SIMDEMULATED_H
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Vectorisation des entiers en utilisant une émulation.
 */
class EMULSimdX2Int32
{
 public:
  static const int BLOCK_SIZE = 2;
  enum
  {
    Length = 2,
    Alignment = 4
  };
 public:
  Int32 v0;
  Int32 v1;
  EMULSimdX2Int32(){}
  explicit EMULSimdX2Int32(Int32 a) : v0(a), v1(a){}
 private:
  EMULSimdX2Int32(Int32 a1,Int32 a0)
  : v0(a0), v1(a1){}
 public:
  EMULSimdX2Int32(const Int32* base,const Int32* idx)
  : v0(base[idx[0]]), v1(base[idx[1]]){}
  explicit EMULSimdX2Int32(const Int32* base)
  : v0(base[0]), v1(base[1]){}

  Int32 operator[](Integer i) const { return (&v0)[i]; }
  Int32& operator[](Integer i) { return (&v0)[i]; }

  void set(ARCANE_RESTRICT Int32* base,const ARCANE_RESTRICT Int32* idx) const
  {
    base[idx[0]] = v0;
    base[idx[1]] = v1;
  }

  void set(ARCANE_RESTRICT Int32* base) const
  {
    base[0] = v0;
    base[1] = v1;
  }

  static EMULSimdX2Int32 fromScalar(Int32 a0,Int32 a1)
  {
    return EMULSimdX2Int32(a1,a0);
  }

 private:
  void operator=(Int32 _v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Vectorisation des entiers en utilisant une émulation.
 */
class EMULSimdX4Int32
{
 public:
  static const int BLOCK_SIZE = 4;
  enum
  {
    Length = 4,
    Alignment = 4
  };
 public:
  Int32 v0;
  Int32 v1;
  Int32 v2;
  Int32 v3;
  EMULSimdX4Int32(){}
  explicit EMULSimdX4Int32(Int32 a)
  : v0(a), v1(a), v2(a), v3(a){}
 private:
  EMULSimdX4Int32(Int32 a3,Int32 a2,Int32 a1,Int32 a0)
  : v0(a0), v1(a1), v2(a2), v3(a3){}
 public:
  EMULSimdX4Int32(const Int32* base,const Int32* idx)
  : v0(base[idx[0]]), v1(base[idx[1]]), v2(base[idx[2]]), v3(base[idx[3]]){}
  explicit EMULSimdX4Int32(const Int32* base)
  : v0(base[0]), v1(base[1]), v2(base[2]), v3(base[3]){}
  explicit EMULSimdX4Int32(const EMULSimdX4Int32* base)
  : v0(base->v0), v1(base->v1), v2(base->v2), v3(base->v3){}

  Int32 operator[](Integer i) const { return (&v0)[i]; }
  Int32& operator[](Integer i) { return (&v0)[i]; }

  void set(ARCANE_RESTRICT Int32* base,const ARCANE_RESTRICT Int32* idx) const
  {
    base[idx[0]] = v0;
    base[idx[1]] = v1;
    base[idx[2]] = v2;
    base[idx[3]] = v3;
  }

  void set(ARCANE_RESTRICT Int32* base) const
  {
    base[0] = v0;
    base[1] = v1;
    base[2] = v2;
    base[3] = v3;
  }

  static EMULSimdX4Int32 fromScalar(Int32 a0,Int32 a1,Int32 a2,Int32 a3)
  {
    return EMULSimdX4Int32(a3,a2,a1,a0);
  }

 private:
  void operator=(Int32 _v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vectorisation des réels par émulation.
 *
 * Cette classe est utilisée lorsqu'aucun mécanisme de vectorisation
 * n'est disponible. Il s'agit juste d'une encapsulation de 2 réels. On
 * aurait pu prendre d'autres tailles de vecteurs (par exemple 4 ou 8) mais
 * les tests effectués (en 2013) montrent que les performances se dégradent
 * au dela de 2.
 */
class EMULSimdReal
{
 public:
  static const int BLOCK_SIZE = 2;
  enum : Int32
  {
    Length = 2
  };
  typedef EMULSimdX2Int32 Int32IndexType;
  Real v0;
  Real v1;
  //NOTE: il est normal que ce constructeur ne fasse pas d'initialisation.
  EMULSimdReal(){}
  explicit EMULSimdReal(Real a)
  : v0(a), v1(a){}
 private:
  EMULSimdReal(Real a0,Real a1)
  : v0(a0), v1(a1){}
 public:
  EMULSimdReal(const Real* base)
  : v0(base[0]), v1(base[1]) {}
  EMULSimdReal(const Real* base,const Int32* idx)
  : v0(base[idx[0]]), v1(base[idx[1]]) {}
  EMULSimdReal(const Real* base,const Int32IndexType* idx)
  : v0(base[idx->v0]), v1(base[idx->v1]) {}
  EMULSimdReal(const Real* base,const Int32IndexType& idx)
  : v0(base[idx.v0]), v1(base[idx.v1]) {}
  const Real& operator[](Integer i) const { return ((Real*)this)[i]; }
  Real& operator[](Integer i) { return ((Real*)this)[i]; }
  void set(ARCANE_RESTRICT Real* base) const
  {
    base[0] = v0;
    base[1] = v1;
  }
  void set(ARCANE_RESTRICT Real* base,const Int32* idx) const
  {
    base[idx[0]] = v0;
    base[idx[1]] = v1;
  }
  void set(ARCANE_RESTRICT Real* base,const Int32IndexType* idx) const
  {
    base[idx->v0] = v0;
    base[idx->v1] = v1;
  }
  void set(ARCANE_RESTRICT Real* base,const Int32IndexType& idx) const
  {
    base[idx.v0] = v0;
    base[idx.v1] = v1;
  }
  static EMULSimdReal fromScalar(Real a0,Real a1)
  {
    return EMULSimdReal(a0,a1);
  }
 private:
  void operator=(Real _v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EMULSimdInfo
{
 public:
  static const char* name() { return "EMUL"; }
  enum : Int32
    {
      Int32IndexSize = 2 
    };
  typedef EMULSimdReal SimdReal;
  typedef EMULSimdReal::Int32IndexType SimdInt32IndexType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_UTILS_EXPORT std::ostream&
operator<<(std::ostream& o,const EMULSimdReal& s);

// Unary operation operator-
inline EMULSimdReal operator- (EMULSimdReal a)
{
  Real* za = (Real*)(&a);
  return EMULSimdReal::fromScalar(-(za[0]),-(za[1]));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
