// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MersenneTwister.h                                           (C) 2000-2006 */
/*                                                                           */
/* Ce fichier definit le patron de classe  MersenneTwister ainsi que deux    */
/* classes  associees mt19937 et mt11213b. Il est une version adapte a TROLL */
/* du fichier MersenneTwister.hpp provenant de la bibliotheque BOOST         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_RANDOM_MERSENNE_TWISTER_H
#define ARCANE_RANDOM_MERSENNE_TWISTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/random/RandomGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RANDOM_BEGIN_NAMESPACE

/*! Patron de classe MersenneTwister. Il permet de définir des classes de 
 * générateurs de type Mersenne Twister en fonction des paramètres w,n,m,r,a,u
 * s,b,t,c et l. Les nombres pseudo-aléatoires générés sont de type UIntType.
 * La génération de ces nombres s'effectue par l'appel de l'opérateur \c (). L'état
 * du genérateur est définie par un membre private x[] de la classe  qui est un
 * tableau de 2*n dimensions. La graine (état initiale du générateur) peut etre
 * initialisée par l'appel des constructeurs ou les différentes méthodes \c seed
 * disponibles.
*/
template<class UIntType, Integer w, Integer n, Integer m, Integer r, UIntType a, Integer u,
  Integer s, UIntType b, Integer t, UIntType c, Integer l, UIntType val>
class MersenneTwister
{
public:
  typedef UIntType result_type;
  static const Integer word_size = w;
  static const Integer state_size = n;
  static const Integer shift_size = m;
  static const Integer mask_bits = r;
  static const UIntType parameter_a = a;
  static const Integer output_u = u;
  static const Integer output_s = s;
  static const UIntType output_b = b;
  static const Integer output_t = t;
  static const UIntType output_c = c;
  static const Integer output_l = l;

  static const bool has_fixed_range = false;
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Constructeur avec initialisation de la graine à partir de la 
   * méthode seed() 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */  
  MersenneTwister() { seed(); }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Constructeur avec initialisation du tableau de graines à partir de
   *         la valeur \c value. L'appel à la méthode \c seed(value) est réalisé.
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  explicit MersenneTwister(UIntType value)
  { seed(value); }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Constructeur avec initialisation du tableau de graines à partir de 
   *         la méthode \c seed(first,last) . 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date   28/07/2006
   */
  template<class It> MersenneTwister(It& first, It last) { seed(first,last); }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Constructeur avec initialisation du tableau de graines à partir du 
   *         générateur \c gen. \c gen doit contenir l'opérateur () qui doit  
   *         retourner une valeur de type UIntType. 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date   28/07/2006
   */
  template<class Generator>
  explicit MersenneTwister(Generator & gen) { seed(gen); }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Initialisation du tableau de graines. L'appel à la méthode \c 
   *         seed(5489) est réalisé. 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date   28/07/2006
   */
  void seed() { seed(UIntType(5489)); }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Initialisation du tableau de graines à partir de la valeur \c value. 
   *         Le tableau de graines de ce générateur est composé de n éléments. 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date   28/07/2006
   */
  void seed(UIntType value)
  {
    // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html
    const UIntType mask = ~0u;
    x[0] = value & mask;
    for (i = 1; i < n; i++) {
      // See Knuth "The Art of Computer Programming" Vol. 2, 3rd ed., page 106
      x[i] = (1812433253UL * (x[i-1] ^ (x[i-1] >> (w-2))) + i) & mask;
    }
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Initialisation du tableau de graines à partir du tableau \c state.
   *         \c state doit être un tableau de n éléments. 
   *
   * \author Patrick Rathouit 
   * \date 28/07/2006
   */
  void seed(UIntType *  state)
  { for (Integer i=0;i<n;i++) x[i] = state[i];}
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Initialisation du tableau de graines à partir du générateur \c gen. 
   *         \c gen est une classe qui doit contenir l'opérateur () retournant  
   *         une valeur de type UIntType. 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  template<class Generator>
  void seed(Generator & gen)
  {
 // For GCC, moving this function out-of-line prevents inlining, which may
  // reduce overall object code size.  However, MSVC does not grok
  // out-of-line definitions of member function templates.
    for(Integer j = 0; j < n; j++)
      x[j] = gen();
    i = n;
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Méthode qui retourne l'état du générateur pour l'index \c j. L'état
   *         complet du générateur est donnée par les valeurs d'index  \c j
   *         comprises   entre 0 et n (0 < \c j <= n)
   *
   * \author Patrick Rathouit 
   * \date 28/07/2006
   */
  UIntType getState(Integer j) 
  {
    return x[j];
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief min() retourne la valeur minimum possible d'une séquence. 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  result_type min() const { return 0; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief max() retourne la valeur maximum possible d'une séquence. 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  result_type max() const
  {
    // avoid "left shift count >= with of type" warning
    result_type res = 0;
    for(Integer i = 0; i < w; ++i)
      res |= (1u << i);
    return res;
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /* déclaration de l'opérateur () */
  result_type operator()();
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Fonction de validation (je ne sais pas trop a quoi elle sert!)
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  static bool validation(result_type v) { return val == v; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Surdéfinition de l'opérateur ==
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  bool operator==(const MersenneTwister& rhs) const
  {
  // Use a member function; Streamable concept not supported.
    for(Integer j = 0; j < state_size; ++j)
      if(compute(j) != rhs.compute(j))
        return false;
    return true;
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Surdéfinition de l'opérateur !=
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  bool operator!=(const MersenneTwister& rhs) const
  { return !(*this == rhs); }

private:
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Méthode privée qui retourne l'état du générateur pour l'index 
   *         \c index.  
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date   28/07/2006
   */
  // returns x(i-n+index), where index is in 0..n-1
  UIntType compute(UIntType index) const
  {
    // equivalent to (i-n+index) % 2n, but doesn't produce negative numbers
    return x[ (i + n + index) % (2*n) ];
  }
  void twist(Integer block);
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /* Membres du patron de classe */
  // state representation: next output is o(x(i))
  //   x[0]  ... x[k] x[k+1] ... x[n-1]     x[n]     ... x[2*n-1]   represents
  //  x(i-k) ... x(i) x(i+1) ... x(i-k+n-1) x(i-k-n) ... x[i(i-k-1)]
  // The goal is to always have x(i-n) ... x(i-1) available for
  // operator== and save/restore.
  UIntType x[2*n]; 
  Integer i;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! \brief Réalisation de l'opération "twist" associée au Mersenne Twister.  
 *         L'état du générateur est modifié.
 *
 * \author Patrick Rathouit (origine bibliotheque BOOST)
 * \date   28/07/2006
 */
template<class UIntType, Integer w, Integer n, Integer m, Integer r, UIntType a, Integer u,
  Integer s, UIntType b, Integer t, UIntType c, Integer l, UIntType val>
void MersenneTwister<UIntType,w,n,m,r,a,u,s,b,t,c,l,val>::twist(int block)
{
  const UIntType upper_mask = (~0u) << r;
  const UIntType lower_mask = ~upper_mask;

  if(block == 0) {
    for(Integer j = n; j < 2*n; j++) {
      UIntType y = (x[j-n] & upper_mask) | (x[j-(n-1)] & lower_mask);
      x[j] = x[j-(n-m)] ^ (y >> 1) ^ (y&1 ? a : 0);
    }
  } else if (block == 1) {
    // split loop to avoid costly modulo operations
    {  // extra scope for MSVC brokenness w.r.t. for scope
      for(Integer j = 0; j < n-m; j++) {
        UIntType y = (x[j+n] & upper_mask) | (x[j+n+1] & lower_mask);
        x[j] = x[j+n+m] ^ (y >> 1) ^ (y&1 ? a : 0);
      }
    }
    
    for(Integer j = n-m; j < n-1; j++) {
      UIntType y = (x[j+n] & upper_mask) | (x[j+n+1] & lower_mask);
      x[j] = x[j-(n-m)] ^ (y >> 1) ^ (y&1 ? a : 0);
    }
    // last iteration
    UIntType y = (x[2*n-1] & upper_mask) | (x[0] & lower_mask);
    x[n-1] = x[m-1] ^ (y >> 1) ^ (y&1 ? a : 0);
    i = 0;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! \brief Surdéfinition de l'opérateur () qui retourne la valeur pseudo 
 *         aléatoire du générateur. L'état du générateur est modifié. 
 *
 * \author Patrick Rathouit (origine bibliotheque BOOST)
 * \date   28/07/2006
 */
template<class UIntType, Integer w, Integer n, Integer m, Integer r, UIntType a, Integer u,
  Integer s, UIntType b, Integer t, UIntType c, Integer l, UIntType val>
inline typename MersenneTwister<UIntType,w,n,m,r,a,u,s,b,t,c,l,val>::result_type
MersenneTwister<UIntType,w,n,m,r,a,u,s,b,t,c,l,val>::operator()()
{
  if(i == n)
    twist(0);
  else if(i >= 2*n)
    twist(1);
  // Step 4
  UIntType z = x[i];
  ++i;
  z ^= (z >> u);
  z ^= ((z << s) & b);
  z ^= ((z << t) & c);
  z ^= (z >> l);
  return z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/* Définition de la classe mt11213b*/
typedef MersenneTwister<UInt32,32,351,175,19,0xccab8ee7,11,
  7,0x31b6ab00,15,0xffe50000,17, 0xa37d3c92> Mt11213b;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/* Définition de la classe Mt19937*/
typedef MersenneTwister<UInt32,32,624,397,31,0x9908b0df,11,
  7,0x9d2c5680,15,0xefc60000,18, 3346425566U> Mt19937;

RANDOM_END_NAMESPACE
ARCANE_END_NAMESPACE


#endif // ARCANE_RANDOM_MERSENNE_TWISTER_H
